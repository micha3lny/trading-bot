"""v30: simple entry scanner + v22 smart exit.

Purpose:
- use the new v29 simple 5m/1m entry algorithm
- do NOT force a daily trade limit
- simulate the same smart exit family as v22
- evaluate whether the higher-throughput entry has edge after costs

Run:
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit --preset quality
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit --preset balanced
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit --preset balanced --exclude-noisy
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryCandidate,
    SimpleEntryPreset,
    load_all_data,
    scan_entries,
)

TAKE_PROFIT_PCT = 3.0
STOP_LOSS_PCT = 1.0
TIME_EXIT_BARS = 60
TIME_EXIT_MIN_PNL_PCT = 0.50
TRAILING_ACTIVATION_PROFIT_PCT = 1.50
TRAILING_STOP_PCT = 1.00

NOISY_SYMBOLS = {
    "SOXL",
    "SOXS",
    "TQQQ",
    "SQQQ",
}


class V30Trade(bt.BacktestTrade):
    pass


def build_candidate_session_map(candidates: list[SimpleEntryCandidate]):
    by_symbol_day: dict[tuple[str, str], list[SimpleEntryCandidate]] = {}
    for candidate in candidates:
        by_symbol_day.setdefault((candidate.symbol, candidate.session_date), []).append(candidate)
    for key in by_symbol_day:
        by_symbol_day[key] = sorted(by_symbol_day[key], key=lambda candidate: candidate.entry_time)
    return by_symbol_day


def find_entry_position(session_1m: pd.DataFrame, entry_time: str) -> int | None:
    ts = pd.Timestamp(entry_time)
    matches = session_1m.index[session_1m["date"] == ts].tolist()
    if matches:
        return int(matches[0])
    after = session_1m.index[session_1m["date"] >= ts].tolist()
    if after:
        return int(after[0])
    return None


def simulate_v22_exit(symbol: str, session: pd.DataFrame, candidate: SimpleEntryCandidate):
    entry_position = find_entry_position(session, candidate.entry_time)
    if entry_position is None:
        return None

    entry_bar = session.iloc[entry_position]
    entry_price = float(entry_bar["close"])
    entry_time = str(entry_bar["date"])
    session_date = str(entry_bar["date"].date())

    stop_price = entry_price * (1.0 - STOP_LOSS_PCT / 100.0)
    take_profit_price = entry_price * (1.0 + TAKE_PROFIT_PCT / 100.0)
    trailing_activation_price = entry_price * (1.0 + TRAILING_ACTIVATION_PROFIT_PCT / 100.0)

    highest_price = entry_price
    trailing_activated = False
    bars_after_entry = session.iloc[entry_position + 1 :]

    setup_type = "v30_simple_entry"
    breakout_proxy = candidate.pullback_from_recent_5m_high_pct
    close_strength = candidate.entry_close_strength_1m
    entry_risk = candidate.entry_risk_pct_1m
    daily_trend = candidate.daily_trend_pct

    if bars_after_entry.empty:
        return bt.BacktestTrade(
            symbol,
            session_date,
            entry_time,
            entry_time,
            entry_price,
            entry_price,
            0.0,
            "v30 no bars after entry",
            breakout_proxy,
            close_strength,
            entry_risk,
            daily_trend,
            setup_type,
        )

    last_bar = bars_after_entry.iloc[-1]
    for bars_held, (_, bar) in enumerate(bars_after_entry.iterrows(), start=1):
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_time = str(bar["date"])
        highest_price = max(highest_price, bar_high)

        if bar_low <= stop_price:
            pnl_pct = (stop_price - entry_price) / entry_price * 100.0
            return bt.BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, stop_price, pnl_pct, "v30 stop-loss", breakout_proxy, close_strength, entry_risk, daily_trend, setup_type)

        if bar_high >= take_profit_price:
            pnl_pct = (take_profit_price - entry_price) / entry_price * 100.0
            return bt.BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, take_profit_price, pnl_pct, "v30 take-profit", breakout_proxy, close_strength, entry_risk, daily_trend, setup_type)

        if highest_price >= trailing_activation_price:
            trailing_activated = True

        if trailing_activated:
            trailing_stop_price = highest_price * (1.0 - TRAILING_STOP_PCT / 100.0)
            if bar_low <= trailing_stop_price:
                pnl_pct = (trailing_stop_price - entry_price) / entry_price * 100.0
                return bt.BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, trailing_stop_price, pnl_pct, "v30 trailing stop", breakout_proxy, close_strength, entry_risk, daily_trend, setup_type)

        if bars_held >= TIME_EXIT_BARS:
            pnl_pct = (bar_close - entry_price) / entry_price * 100.0
            if pnl_pct < TIME_EXIT_MIN_PNL_PCT:
                return bt.BacktestTrade(symbol, session_date, entry_time, bar_time, entry_price, bar_close, pnl_pct, "v30 time exit", breakout_proxy, close_strength, entry_risk, daily_trend, setup_type)

    exit_price = float(last_bar["close"])
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    return bt.BacktestTrade(symbol, session_date, entry_time, str(last_bar["date"]), entry_price, exit_price, pnl_pct, "v30 end of session", breakout_proxy, close_strength, entry_risk, daily_trend, setup_type)


def run_backtest(preset: SimpleEntryPreset, exclude_noisy: bool):
    counters, candidates, candidates_by_day = scan_entries(preset)
    if exclude_noisy:
        candidates = [candidate for candidate in candidates if candidate.symbol not in NOISY_SYMBOLS]

    data_15m, data_5m, data_1m, daily_data = load_all_data()
    candidates_by_symbol_day = build_candidate_session_map(candidates)

    trades = []
    for (symbol, session_date), day_candidates in candidates_by_symbol_day.items():
        if symbol not in data_1m:
            continue
        session_1m = data_1m[symbol][data_1m[symbol]["session_date"].astype(str) == session_date].sort_values("date").reset_index(drop=True)
        if session_1m.empty:
            continue
        for candidate in day_candidates:
            trade = simulate_v22_exit(symbol, session_1m, candidate)
            if trade is not None:
                trades.append(trade)

    net_trades = apply_costs_to_trades(trades)
    return counters, candidates, candidates_by_day, net_trades


def summarize_inputs(counters: Counter, candidates: list[SimpleEntryCandidate], candidates_by_day, exclude_noisy: bool) -> None:
    print("\nEntry candidate pool")
    print(f"Candidates after symbol filter: {len(candidates)}")
    print(f"Active days: {len(set(candidate.session_date for candidate in candidates))}")
    if candidates:
        max_per_day = max(Counter(candidate.session_date for candidate in candidates).values())
        avg_per_active_day = len(candidates) / len(set(candidate.session_date for candidate in candidates))
        print(f"Max candidates on one day: {max_per_day}")
        print(f"Avg candidates per active day: {avg_per_active_day:.2f}")
    print(f"Exclude noisy ETFs: {exclude_noisy}")

    by_symbol = Counter(candidate.symbol for candidate in candidates)
    print("\nTop symbols by candidate count")
    for symbol, count in by_symbol.most_common(30):
        print(f"{symbol:<6} {count:>4}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    parser.add_argument("--exclude-noisy", action="store_true", help="Exclude SOXL/SOXS/TQQQ/SQQQ")
    args = parser.parse_args()

    preset = PRESETS[args.preset]

    print("\nExperiment: v30 simple entry + v22 smart exit")
    print(f"Entry preset: {preset.name} - {preset.description}")
    print("Exit model:")
    print(f"- take_profit={TAKE_PROFIT_PCT:.2f}%")
    print(f"- stop_loss={STOP_LOSS_PCT:.2f}%")
    print(f"- time_exit_bars={TIME_EXIT_BARS}, min_pnl={TIME_EXIT_MIN_PNL_PCT:.2f}%")
    print(f"- trailing_activation={TRAILING_ACTIVATION_PROFIT_PCT:.2f}%, trailing_stop={TRAILING_STOP_PCT:.2f}%")

    counters, candidates, candidates_by_day, trades = run_backtest(preset, args.exclude_noisy)
    summarize_inputs(counters, candidates, candidates_by_day, args.exclude_noisy)

    bt.summarize_research(trades)
    df = bt.export_trades(trades)
    bt.analyze(df)


if __name__ == "__main__":
    main()
