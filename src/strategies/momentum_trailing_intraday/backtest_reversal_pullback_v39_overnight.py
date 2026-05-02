"""v39: overnight / multi-day reversal pullback strategy.

This is a SECOND strategy, not a replacement for the intraday v38 strategy.

Entry:
- Reuses v38 optimizer-winner entry profiles.

Exit:
- No forced same-day exit.
- Keeps stop-loss, take-profit and trailing logic across later 1m bars.
- Forces exit after max hold days.

Run:
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v39_overnight --preset quality
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v39_overnight --preset quality --profile sample20
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v39_overnight --preset quality --profile sample20 --max-hold-days 5 --stop-loss 1.5
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades
from src.strategies.momentum_trailing_intraday.backtest import BacktestTrade
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v17_1m_entry import summarize_research
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit import (
    build_candidate_session_map,
)
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v38_optimizer_winner import (
    PROFILES,
    V38Profile,
    passes_profile,
)
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryCandidate,
    load_all_data,
    scan_entries,
)

DEFAULT_STOP_LOSS_PCT = 1.00
DEFAULT_TAKE_PROFIT_PCT = 3.00
DEFAULT_TRAILING_ACTIVATION_PCT = 1.50
DEFAULT_TRAILING_STOP_PCT = 1.00
DEFAULT_MAX_HOLD_DAYS = 3


def parse_time(value: str) -> pd.Timestamp:
    return pd.to_datetime(value)


def multi_day_window(symbol_1m: pd.DataFrame, entry_time: str, max_hold_days: int) -> pd.DataFrame:
    df = symbol_1m.sort_values("date").reset_index(drop=True).copy()
    entry_ts = parse_time(entry_time)
    entry_day = entry_ts.date()

    sessions = sorted(pd.to_datetime(df["date"]).dt.date.unique())
    future_sessions = [session for session in sessions if session >= entry_day][:max_hold_days]
    if not future_sessions:
        return pd.DataFrame()

    window = df[pd.to_datetime(df["date"]).dt.date.isin(future_sessions)].copy()
    window = window[pd.to_datetime(window["date"]) >= entry_ts]
    return window.sort_values("date").reset_index(drop=True)


def simulate_overnight_exit(
    symbol: str,
    symbol_1m: pd.DataFrame,
    candidate: SimpleEntryCandidate,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_activation_pct: float,
    trailing_stop_pct: float,
    max_hold_days: int,
) -> BacktestTrade | None:
    bars = multi_day_window(symbol_1m, candidate.entry_time, max_hold_days)
    if bars.empty:
        return None

    # Prefer the candidate price instead of the possibly rounded 1m close.
    entry_price = float(candidate.entry_price)
    entry_time = str(candidate.entry_time)
    session_date = str(candidate.session_date)

    stop_price = entry_price * (1.0 - stop_loss_pct / 100.0)
    take_profit_price = entry_price * (1.0 + take_profit_pct / 100.0)
    trailing_activation_price = entry_price * (1.0 + trailing_activation_pct / 100.0)

    highest_price = entry_price
    trailing_activated = False
    last_bar = bars.iloc[-1]

    for _, bar in bars.iloc[1:].iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_time = str(bar["date"])

        highest_price = max(highest_price, bar_high)

        if bar_low <= stop_price:
            pnl_pct = (stop_price - entry_price) / entry_price * 100.0
            return BacktestTrade(
                symbol=symbol,
                session_date=session_date,
                entry_time=entry_time,
                exit_time=bar_time,
                entry_price=entry_price,
                exit_price=stop_price,
                pnl_pct=pnl_pct,
                exit_reason="v39 overnight stop-loss",
                breakout_pct=float(candidate.pullback_from_recent_5m_high_pct),
                close_strength=float(candidate.entry_close_strength_1m),
                entry_risk_pct=float(candidate.entry_risk_pct_1m),
                daily_trend_pct=float(candidate.daily_trend_pct),
                setup_type="v39_overnight_reversal_pullback",
            )

        if bar_high >= take_profit_price:
            pnl_pct = (take_profit_price - entry_price) / entry_price * 100.0
            return BacktestTrade(
                symbol=symbol,
                session_date=session_date,
                entry_time=entry_time,
                exit_time=bar_time,
                entry_price=entry_price,
                exit_price=take_profit_price,
                pnl_pct=pnl_pct,
                exit_reason="v39 overnight take-profit",
                breakout_pct=float(candidate.pullback_from_recent_5m_high_pct),
                close_strength=float(candidate.entry_close_strength_1m),
                entry_risk_pct=float(candidate.entry_risk_pct_1m),
                daily_trend_pct=float(candidate.daily_trend_pct),
                setup_type="v39_overnight_reversal_pullback",
            )

        if highest_price >= trailing_activation_price:
            trailing_activated = True

        if trailing_activated:
            trailing_stop_price = highest_price * (1.0 - trailing_stop_pct / 100.0)
            if bar_low <= trailing_stop_price:
                pnl_pct = (trailing_stop_price - entry_price) / entry_price * 100.0
                return BacktestTrade(
                    symbol=symbol,
                    session_date=session_date,
                    entry_time=entry_time,
                    exit_time=bar_time,
                    entry_price=entry_price,
                    exit_price=trailing_stop_price,
                    pnl_pct=pnl_pct,
                    exit_reason="v39 overnight trailing stop",
                    breakout_pct=float(candidate.pullback_from_recent_5m_high_pct),
                    close_strength=float(candidate.entry_close_strength_1m),
                    entry_risk_pct=float(candidate.entry_risk_pct_1m),
                    daily_trend_pct=float(candidate.daily_trend_pct),
                    setup_type="v39_overnight_reversal_pullback",
                )

        last_bar = bar

    exit_price = float(last_bar["close"])
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    return BacktestTrade(
        symbol=symbol,
        session_date=session_date,
        entry_time=entry_time,
        exit_time=str(last_bar["date"]),
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        exit_reason=f"v39 max-hold {max_hold_days}d exit",
        breakout_pct=float(candidate.pullback_from_recent_5m_high_pct),
        close_strength=float(candidate.entry_close_strength_1m),
        entry_risk_pct=float(candidate.entry_risk_pct_1m),
        daily_trend_pct=float(candidate.daily_trend_pct),
        setup_type="v39_overnight_reversal_pullback",
    )


def run_backtest(
    preset_name: str,
    profile: V38Profile,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_activation_pct: float,
    trailing_stop_pct: float,
    max_hold_days: int,
):
    _counters, raw_candidates, _candidates_by_day = scan_entries(PRESETS[preset_name])
    _data_15m, _data_5m, data_1m, _daily_data = load_all_data()

    counters = Counter()
    candidates: list[SimpleEntryCandidate] = []
    for candidate in raw_candidates:
        counters["raw_candidates"] += 1
        if passes_profile(candidate, profile, data_1m):
            counters["passed_v39_profile"] += 1
            candidates.append(candidate)
        else:
            counters["rejected_v39_profile"] += 1

    candidates_by_symbol_day = build_candidate_session_map(candidates)
    trades = []
    for (symbol, _session_date), day_candidates in candidates_by_symbol_day.items():
        if symbol not in data_1m:
            continue
        symbol_1m = data_1m[symbol].sort_values("date").reset_index(drop=True)
        for candidate in day_candidates:
            trade = simulate_overnight_exit(
                symbol,
                symbol_1m,
                candidate,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                trailing_activation_pct=trailing_activation_pct,
                trailing_stop_pct=trailing_stop_pct,
                max_hold_days=max_hold_days,
            )
            if trade is not None:
                trades.append(trade)

    return counters, candidates, apply_costs_to_trades(trades)


def summarize_inputs(counters: Counter, candidates: list[SimpleEntryCandidate], profile: V38Profile) -> None:
    print("\nFunnel counts:")
    for name, count in counters.most_common():
        print(f"{name:<28} {count:>6}")

    print("\nEntry candidate pool")
    print(f"Candidates after filters: {len(candidates)}")
    active_days = len(set(candidate.session_date for candidate in candidates))
    print(f"Active days: {active_days}")
    if candidates and active_days:
        by_day = Counter(candidate.session_date for candidate in candidates)
        print(f"Max candidates on one day: {max(by_day.values())}")
        print(f"Avg candidates per active day: {len(candidates) / active_days:.2f}")

    print("v39 entry profile:")
    print(f"- source profile: v38 {profile.name} - {profile.description}")
    print(f"- daily_trend: {profile.trend_min:.2f}% to {profile.trend_max:.2f}%")
    print(f"- pullback proxy: {profile.pullback_min:.2f}% to {profile.pullback_max:.2f}%")
    print(f"- 1m close_strength: {profile.cs_min:.2f} to {profile.cs_max:.2f}")
    print(f"- entry_risk: {profile.risk_min:.2f}% to {profile.risk_max:.2f}%")
    print(f"- pre_15m_low_to_entry <= {profile.pre15_bounce_max:.2f}%")
    print(f"- 5m close_strength <= {profile.cs5_max:.2f}")
    print(f"- excluded symbols: {sorted(profile.excluded_symbols)}")

    print("\nTop symbols by candidate count")
    for symbol, count in Counter(candidate.symbol for candidate in candidates).most_common(30):
        print(f"{symbol:<6} {count:>4}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    parser.add_argument("--profile", default="strict", choices=sorted(PROFILES))
    parser.add_argument("--max-hold-days", type=int, default=DEFAULT_MAX_HOLD_DAYS)
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_STOP_LOSS_PCT)
    parser.add_argument("--take-profit", type=float, default=DEFAULT_TAKE_PROFIT_PCT)
    parser.add_argument("--trailing-activation", type=float, default=DEFAULT_TRAILING_ACTIVATION_PCT)
    parser.add_argument("--trailing-stop", type=float, default=DEFAULT_TRAILING_STOP_PCT)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    print("\nExperiment: v39 overnight reversal pullback")
    print(f"Entry preset: {args.preset} - {PRESETS[args.preset].description}")
    print(f"Entry profile: v38 {profile.name} - {profile.description}")
    print("Exit model:")
    print(f"- max_hold_days={args.max_hold_days}")
    print(f"- stop_loss={args.stop_loss:.2f}%")
    print(f"- take_profit={args.take_profit:.2f}%")
    print(f"- trailing_activation={args.trailing_activation:.2f}%")
    print(f"- trailing_stop={args.trailing_stop:.2f}%")
    print("- no forced same-day exit")

    counters, candidates, trades = run_backtest(
        args.preset,
        profile,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        trailing_activation_pct=args.trailing_activation,
        trailing_stop_pct=args.trailing_stop,
        max_hold_days=args.max_hold_days,
    )
    summarize_inputs(counters, candidates, profile)
    summarize_research(trades)
    df = export_trades(trades)
    analyze(df)


if __name__ == "__main__":
    main()
