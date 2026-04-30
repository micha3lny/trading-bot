"""Reversal pullback v24: test earlier funnel expansion with v22 smart exit.

Builds on v23 result.

v23 widened the 15m/5m/1m entry windows but produced only one extra trade and lower
average PnL. That suggests the practical bottleneck is earlier in the funnel:
- market regime filter
- ADR volatility filter

Experiment:
- keep v23 scaled entry
- keep v22 smart exit
- lower ADR from 5.0% to 4.0%
- disable market regime filter

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v23_scaled_entry as v23

# Earlier-funnel expansion.
v23.v22.v20.v17.MIN_AVG_DAILY_RANGE_PCT = 4.0
v23.v22.v20.v17.bt.ENABLE_MARKET_REGIME_FILTER = False


def main():
    print("\nExperiment: reversal pullback v24 regime/ADR expansion + v22 smart exit")
    print("Entry: v23 scaled entry")
    print("Early funnel changes:")
    print(f"- market_regime_filter={v23.v22.v20.v17.bt.ENABLE_MARKET_REGIME_FILTER}")
    print(f"- avg_daily_range >= {v23.v22.v20.v17.MIN_AVG_DAILY_RANGE_PCT:.2f}%")
    print("Exit: v22 smart hybrid exit")
    v23.main()


if __name__ == "__main__":
    main()
