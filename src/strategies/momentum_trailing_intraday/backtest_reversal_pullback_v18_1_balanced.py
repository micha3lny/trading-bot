"""Reversal pullback v18.1: balanced hard filters.

Purpose:
- increase sample size versus v18
- keep the core v18 idea: strong 15m breakout attempt + 1m confirmation
- test whether edge survives with slightly looser filters

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v17_1m_entry as v17

# Balanced filters versus v18:
# - keep breakout >= 1.0%
# - lower 1m close strength from 0.80 to 0.70
# - lower ADR from 6.0% to 5.0%
v17.MIN_BREAKOUT_PCT = 1.00
v17.MAX_BREAKOUT_PCT = 2.50
v17.MIN_1M_CLOSE_STRENGTH = 0.70
v17.MIN_AVG_DAILY_RANGE_PCT = 5.0

v17.REQUIRE_1M_CLOSE_ABOVE_PREV_CLOSE = True
v17.REQUIRE_1M_CLOSE_ABOVE_PREV_HIGH = False


def main():
    print("\nExperiment: reversal pullback v18.1 balanced filters full universe (with costs)")
    print("Changes versus v18:")
    print(f"- 15m breakout_attempt >= {v17.MIN_BREAKOUT_PCT:.2f}%")
    print(f"- 1m close_strength >= {v17.MIN_1M_CLOSE_STRENGTH:.2f}")
    print(f"- avg_daily_range >= {v17.MIN_AVG_DAILY_RANGE_PCT:.2f}%")
    print("- universe: all local symbols with 1D + 15m + 5m + 1m data")
    print("- no portfolio ranking yet; research includes every valid signal")
    v17.main()


if __name__ == "__main__":
    main()
