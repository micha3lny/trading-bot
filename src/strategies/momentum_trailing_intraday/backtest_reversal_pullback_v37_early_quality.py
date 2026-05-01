"""v37: early reclaim trigger with quality filters.

Built from v36 entry audit:
- v36 restored entry frequency but admitted too many fake reclaims
- high 1m close_strength was mostly late/fake
- good timing had stronger pullback proxy and controlled risk

Changes vs v36:
- require pullback/breakout proxy >= 2.0%
- entry risk 5.0% to 10.0%
- current 1m close_strength must stay below 0.80
- keep early reclaim logic: weak previous 1m + current reclaim
- remove repeated weak v36 symbols: RIOT, RXRX, JOBY, DNA

Run:
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v37_early_quality --preset quality
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v17_1m_entry import summarize_research
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit import (
    NOISY_SYMBOLS,
    build_candidate_session_map,
    simulate_v22_exit,
)
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v36_early_trigger import (
    V36Candidate,
    close_strength,
    distance_below_or_high_pct,
    entry_risk_pct,
    find_opening_context,
    get_daily_value_before_session,
)
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryPreset,
    load_all_data,
)

V37_MIN_DAILY_TREND_PCT = -7.0
V37_MAX_DAILY_TREND_PCT = -3.0
V37_MIN_PULLBACK_PROXY_PCT = 2.00
V37_MAX_PULLBACK_PROXY_PCT = 4.00
V37_MIN_ENTRY_RISK_PCT = 5.0
V37_MAX_ENTRY_RISK_PCT = 10.0
V37_MAX_5M_CLOSE_STRENGTH = 0.75
V37_MIN_DISTANCE_BELOW_OR_HIGH_PCT = 0.25
V37_MAX_DISTANCE_BELOW_OR_HIGH_PCT = 3.50

V37_MAX_PREV_1M_CLOSE_STRENGTH = 0.65
V37_MIN_CURRENT_1M_CLOSE_STRENGTH = 0.45
V37_MAX_CURRENT_1M_CLOSE_STRENGTH = 0.80
V37_TRIGGER_LOOKAHEAD_1M_BARS = 8

V37_EXCLUDED_SYMBOLS = set(NOISY_SYMBOLS) | {"UUUU", "RIOT", "RXRX", "JOBY", "DNA"}


@dataclass(frozen=True)
class V37Candidate(V36Candidate):
    trigger_type: str = "v37_early_quality"


def is_v37_reclaim(prev_1m, row_1m) -> bool:
    prev_cs = close_strength(prev_1m)
    current_cs = close_strength(row_1m)
    current_open = float(row_1m["open"])
    current_high = float(row_1m["high"])
    current_low = float(row_1m["low"])
    current_close = float(row_1m["close"])
    prev_close = float(prev_1m["close"])

    if prev_cs > V37_MAX_PREV_1M_CLOSE_STRENGTH:
        return False
    if current_cs < V37_MIN_CURRENT_1M_CLOSE_STRENGTH:
        return False
    if current_cs > V37_MAX_CURRENT_1M_CLOSE_STRENGTH:
        return False
    if current_close <= prev_close:
        return False
    if current_close < current_open:
        return False
    midpoint = current_low + (current_high - current_low) * 0.50
    if current_close < midpoint:
        return False
    return True


def find_v37_entries_for_session(
    symbol: str,
    session_date,
    session_15m,
    session_5m,
    session_1m,
    daily_trend_pct: float,
    avg_daily_range_pct: float,
    preset: SimpleEntryPreset,
    counters: Counter,
):
    context = find_opening_context(session_15m)
    if context is None:
        counters["too_few_15m_bars"] += 1
        return []

    or_high = context["or_high"]
    or_low = context["or_low"]

    if session_5m.empty or session_1m.empty:
        counters["missing_intraday_session"] += 1
        return []

    session_5m = session_5m.sort_values("date").reset_index(drop=True)
    session_1m = session_1m.sort_values("date").reset_index(drop=True)

    entries: list[V37Candidate] = []
    lookback_5m_bars = getattr(preset, "lookback_5m_bars", 6)

    for idx in range(lookback_5m_bars, len(session_5m)):
        row_5m = session_5m.iloc[idx]
        window_5m = session_5m.iloc[idx - lookback_5m_bars : idx + 1]
        recent_high = float(window_5m["high"].max())
        recent_low = float(window_5m["low"].min())
        close_5m = float(row_5m["close"])
        low_5m = float(row_5m["low"])

        if recent_high == 0:
            continue

        pullback_proxy_pct = (recent_high - low_5m) / recent_high * 100.0
        cs_5m = close_strength(row_5m)
        below_or_pct = distance_below_or_high_pct(close_5m, or_high)
        risk_5m = entry_risk_pct(close_5m, min(or_low, recent_low))

        if pullback_proxy_pct < V37_MIN_PULLBACK_PROXY_PCT:
            continue
        if pullback_proxy_pct > V37_MAX_PULLBACK_PROXY_PCT:
            counters["pullback_too_deep"] += 1
            continue
        if cs_5m > V37_MAX_5M_CLOSE_STRENGTH:
            counters["5m_close_strength_too_high"] += 1
            continue
        if below_or_pct < V37_MIN_DISTANCE_BELOW_OR_HIGH_PCT:
            counters["too_close_or_above_opening_high"] += 1
            continue
        if below_or_pct > V37_MAX_DISTANCE_BELOW_OR_HIGH_PCT:
            counters["too_far_below_opening_high"] += 1
            continue
        if risk_5m > V37_MAX_ENTRY_RISK_PCT:
            counters["5m_entry_risk_too_high"] += 1
            continue

        trigger_window = session_1m[session_1m["date"] >= row_5m["date"]].head(V37_TRIGGER_LOOKAHEAD_1M_BARS).reset_index(drop=True)
        if len(trigger_window) < 2:
            counters["no_1m_after_5m"] += 1
            continue

        for j in range(1, len(trigger_window)):
            prev_1m = trigger_window.iloc[j - 1]
            row_1m = trigger_window.iloc[j]
            if not is_v37_reclaim(prev_1m, row_1m):
                continue

            entry_price = float(row_1m["close"])
            risk_1m = entry_risk_pct(entry_price, min(or_low, recent_low))
            if risk_1m < V37_MIN_ENTRY_RISK_PCT:
                counters["1m_entry_risk_too_low"] += 1
                continue
            if risk_1m > V37_MAX_ENTRY_RISK_PCT:
                counters["1m_entry_risk_too_high"] += 1
                continue

            entries.append(
                V37Candidate(
                    symbol=symbol,
                    session_date=str(session_date),
                    entry_time=str(row_1m["date"]),
                    entry_price=entry_price,
                    daily_trend_pct=daily_trend_pct,
                    avg_daily_range_pct=avg_daily_range_pct,
                    distance_below_or_high_pct=below_or_pct,
                    pullback_from_recent_5m_high_pct=pullback_proxy_pct,
                    close_strength_5m=cs_5m,
                    entry_close_strength_1m=close_strength(row_1m),
                    entry_risk_pct_1m=risk_1m,
                )
            )
            counters["passed_v37_early_quality"] += 1
            return entries

    counters["no_v37_entry"] += 1
    return entries


def scan_v37_entries(preset_name: str):
    preset = PRESETS[preset_name]
    data_15m, data_5m, data_1m, daily_data = load_all_data()
    counters = Counter()
    candidates: list[V37Candidate] = []

    for symbol, d15 in data_15m.items():
        if symbol in V37_EXCLUDED_SYMBOLS:
            counters["excluded_symbol"] += 1
            continue

        for session_date, session_15m in d15.groupby("session_date"):
            counters["symbol_days"] += 1
            daily = daily_data[symbol]
            avg_daily_range_pct = get_daily_value_before_session(daily, session_date, "avg_daily_range_pct")
            if avg_daily_range_pct < preset.min_avg_daily_range_pct:
                counters["rejected_adr"] += 1
                continue
            counters["passed_adr"] += 1

            daily_trend_pct = get_daily_value_before_session(daily, session_date, "daily_trend_pct")
            if not (V37_MIN_DAILY_TREND_PCT <= daily_trend_pct <= V37_MAX_DAILY_TREND_PCT):
                counters["daily_trend_rejected"] += 1
                continue
            counters["passed_daily_trend"] += 1

            session_15m = session_15m.sort_values("date").reset_index(drop=True)
            session_5m = data_5m[symbol][data_5m[symbol]["session_date"] == session_date]
            session_1m = data_1m[symbol][data_1m[symbol]["session_date"] == session_date]
            candidates.extend(
                find_v37_entries_for_session(
                    symbol,
                    session_date,
                    session_15m,
                    session_5m,
                    session_1m,
                    daily_trend_pct,
                    avg_daily_range_pct,
                    preset,
                    counters,
                )
            )

    return counters, candidates, data_1m


def run_backtest(preset_name: str):
    counters, candidates, data_1m = scan_v37_entries(preset_name)
    candidates_by_symbol_day = build_candidate_session_map(candidates)

    trades = []
    for (symbol, session_date), day_candidates in candidates_by_symbol_day.items():
        if symbol not in data_1m:
            continue
        session_1m = data_1m[symbol][data_1m[symbol]["session_date"].astype(str) == session_date].sort_values("date").reset_index(drop=True)
        if session_1m.empty:
            continue
        for candidate in day_candidates:
            trade = simulate_v22_exit(symbol, session_1m, candidate, False, False, False)
            if trade is not None:
                trades.append(trade)

    net_trades = apply_costs_to_trades(trades)
    return counters, candidates, net_trades


def summarize_inputs(counters: Counter, candidates: list[V37Candidate]) -> None:
    print("\nFunnel counts:")
    for name, count in counters.most_common():
        print(f"{name:<36} {count:>6}")

    print("\nEntry candidate pool")
    print(f"Candidates after filters: {len(candidates)}")
    active_days = len(set(candidate.session_date for candidate in candidates))
    print(f"Active days: {active_days}")
    if candidates and active_days:
        by_day = Counter(candidate.session_date for candidate in candidates)
        print(f"Max candidates on one day: {max(by_day.values())}")
        print(f"Avg candidates per active day: {len(candidates) / active_days:.2f}")

    print("v37 filters:")
    print(f"- daily_trend: {V37_MIN_DAILY_TREND_PCT:.2f}% to {V37_MAX_DAILY_TREND_PCT:.2f}%")
    print(f"- pullback/breakout proxy: {V37_MIN_PULLBACK_PROXY_PCT:.2f}% to {V37_MAX_PULLBACK_PROXY_PCT:.2f}%")
    print(f"- entry_risk: {V37_MIN_ENTRY_RISK_PCT:.2f}% to {V37_MAX_ENTRY_RISK_PCT:.2f}%")
    print(f"- max 5m close_strength: {V37_MAX_5M_CLOSE_STRENGTH:.2f}")
    print(f"- prev 1m close_strength <= {V37_MAX_PREV_1M_CLOSE_STRENGTH:.2f}")
    print(f"- current 1m close_strength: {V37_MIN_CURRENT_1M_CLOSE_STRENGTH:.2f} to {V37_MAX_CURRENT_1M_CLOSE_STRENGTH:.2f}")
    print(f"- excluded symbols: {sorted(V37_EXCLUDED_SYMBOLS)}")

    by_symbol = Counter(candidate.symbol for candidate in candidates)
    print("\nTop symbols by candidate count")
    for symbol, count in by_symbol.most_common(30):
        print(f"{symbol:<6} {count:>4}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    args = parser.parse_args()

    print("\nExperiment: v37 early quality reclaim + v22 smart exit")
    print(f"Entry preset: {args.preset} - {PRESETS[args.preset].description}")
    print("Exit model:")
    print("- take_profit=3.00%")
    print("- stop_loss=1.00%")
    print("- time_exit_bars=60, min_pnl=0.50%")
    print("- trailing_activation=1.50%, trailing_stop=1.00%")

    counters, candidates, trades = run_backtest(args.preset)
    summarize_inputs(counters, candidates)
    summarize_research(trades)
    df = export_trades(trades)
    analyze(df)


if __name__ == "__main__":
    main()
