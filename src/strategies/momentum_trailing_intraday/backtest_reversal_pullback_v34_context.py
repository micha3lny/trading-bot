"""v34: scaled context entry + symbol hygiene + v22 smart exit.

Built from v33 findings:
- v33 had positive edge but too few trades
- UUUU was the only repeated weak symbol in v33
- v34 removes UUUU and slightly relaxes pullback proxy to increase sample size

Run:
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v34_context --preset quality
python -m src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v34_context --preset balanced
"""

from __future__ import annotations

import argparse
from collections import Counter

from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v17_1m_entry import summarize_research
from src.strategies.momentum_trailing_intraday.backtest_reversal_pullback_v30_simple_entry_exit import (
    NOISY_SYMBOLS,
    build_candidate_session_map,
    simulate_v22_exit,
)
from src.strategies.momentum_trailing_intraday.costs import apply_costs_to_trades
from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    PRESETS,
    SimpleEntryCandidate,
    load_all_data,
    scan_entries,
)

V34_MIN_DAILY_TREND_PCT = -6.0
V34_MAX_DAILY_TREND_PCT = -3.0
V34_MIN_ENTRY_RISK_PCT = 4.0
V34_MIN_1M_CLOSE_STRENGTH = 0.80
V34_MIN_PULLBACK_PROXY_PCT = 1.00

V34_EXCLUDED_SYMBOLS = set(NOISY_SYMBOLS) | {"UUUU"}


def passes_v34_context(candidate: SimpleEntryCandidate) -> bool:
    if candidate.symbol in V34_EXCLUDED_SYMBOLS:
        return False
    if not (V34_MIN_DAILY_TREND_PCT <= candidate.daily_trend_pct <= V34_MAX_DAILY_TREND_PCT):
        return False
    if candidate.entry_risk_pct_1m < V34_MIN_ENTRY_RISK_PCT:
        return False
    if candidate.entry_close_strength_1m < V34_MIN_1M_CLOSE_STRENGTH:
        return False
    if candidate.pullback_from_recent_5m_high_pct < V34_MIN_PULLBACK_PROXY_PCT:
        return False
    return True


def run_backtest(preset_name: str):
    preset = PRESETS[preset_name]
    _counters, candidates, _candidates_by_day = scan_entries(preset)
    candidates = [candidate for candidate in candidates if passes_v34_context(candidate)]

    _data_15m, _data_5m, data_1m, _daily_data = load_all_data()
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
    return candidates, net_trades


def summarize_inputs(candidates: list[SimpleEntryCandidate]) -> None:
    print("\nEntry candidate pool")
    print(f"Candidates after filters: {len(candidates)}")
    active_days = len(set(candidate.session_date for candidate in candidates))
    print(f"Active days: {active_days}")
    if candidates and active_days:
        by_day = Counter(candidate.session_date for candidate in candidates)
        print(f"Max candidates on one day: {max(by_day.values())}")
        print(f"Avg candidates per active day: {len(candidates) / active_days:.2f}")

    print("v34 filters:")
    print(f"- daily_trend: {V34_MIN_DAILY_TREND_PCT:.2f}% to {V34_MAX_DAILY_TREND_PCT:.2f}%")
    print(f"- entry_risk >= {V34_MIN_ENTRY_RISK_PCT:.2f}%")
    print(f"- 1m close_strength >= {V34_MIN_1M_CLOSE_STRENGTH:.2f}")
    print(f"- pullback/breakout proxy >= {V34_MIN_PULLBACK_PROXY_PCT:.2f}%")
    print(f"- excluded symbols: {sorted(V34_EXCLUDED_SYMBOLS)}")

    by_symbol = Counter(candidate.symbol for candidate in candidates)
    print("\nTop symbols by candidate count")
    for symbol, count in by_symbol.most_common(30):
        print(f"{symbol:<6} {count:>4}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    args = parser.parse_args()

    print("\nExperiment: v34 context entry + symbol hygiene + v22 smart exit")
    print(f"Entry preset: {args.preset} - {PRESETS[args.preset].description}")
    print("Exit model:")
    print("- take_profit=3.00%")
    print("- stop_loss=1.00%")
    print("- time_exit_bars=60, min_pnl=0.50%")
    print("- trailing_activation=1.50%, trailing_stop=1.00%")

    candidates, trades = run_backtest(args.preset)
    summarize_inputs(candidates)
    summarize_research(trades)
    df = export_trades(trades)
    analyze(df)


if __name__ == "__main__":
    main()
