"""Reversal pullback v19.1: sweet spot + day-level risk control.

Adds a simple but powerful constraint:
- limit number of trades per day

Motivation:
- v18 experiments showed clustering of losses in single sessions
- goal: reduce drawdown without hurting edge

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

from collections import defaultdict

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v19_sweet_spot as v19

MAX_TRADES_PER_DAY = 2

_original_backtest_symbol = v19.v17.backtest_symbol


def backtest_symbol_with_day_limit(symbol, data_15m, data_5m, data_1m, daily, regimes):
    trades = []
    trades_per_day = defaultdict(int)

    for trade in _original_backtest_symbol(symbol, data_15m, data_5m, data_1m, daily, regimes):
        if trades_per_day[trade.session_date] >= MAX_TRADES_PER_DAY:
            continue
        trades.append(trade)
        trades_per_day[trade.session_date] += 1

    return trades


v19.v17.backtest_symbol = backtest_symbol_with_day_limit


def main():
    print("\nExperiment: reversal pullback v19.1 sweet spot + day filter")
    print(f"- max trades per day: {MAX_TRADES_PER_DAY}")
    v19.main()


if __name__ == "__main__":
    main()
