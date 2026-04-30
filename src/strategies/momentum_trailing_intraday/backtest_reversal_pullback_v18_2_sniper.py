"""Reversal pullback v18.2: sniper filters.

Purpose:
- extreme selectivity
- test if only strongest breakouts drive the edge
- expect very low trade count but high quality

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v17_1m_entry as v17

# Sniper filters:
# - stronger breakout >= 1.5%
# - keep strong 1m confirmation (0.80)
# - keep strict ADR (6.0%)
v17.MIN_BREAKOUT_PCT = 1.50
v17.MAX_BREAKOUT_PCT = 3.00
v17.MIN_1M_CLOSE_STRENGTH = 0.80
v17.MIN_AVG_DAILY_RANGE_PCT = 6.0

v17.REQUIRE_1M_CLOSE_ABOVE_PREV_CLOSE = True
v17.REQUIRE_1M_CLOSE_ABOVE_PREV_HIGH = False


def main():
    print("\nExperiment: reversal pullback v18.2 sniper filters full universe (with costs)")
    print("Changes versus v18:")
    print(f"- 15m breakout_attempt >= {v17.MIN_BREAKOUT_PCT:.2f}%")
    print(f"- 1m close_strength >= {v17.MIN_1M_CLOSE_STRENGTH:.2f}")
    print(f"- avg_daily_range >= {v17.MIN_AVG_DAILY_RANGE_PCT:.2f}%")
    print("- universe: all local symbols with 1D + 15m + 5m + 1m data")
    print("- no portfolio ranking yet; research includes every valid signal")
    v17.main()


if __name__ == "__main__":
    main()
