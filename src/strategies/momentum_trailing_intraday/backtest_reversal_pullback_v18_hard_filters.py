"""Reversal pullback v18: hard-filtered MTF setup with 1m entry trigger.

Builds on v17/v15 research.

Architecture:
- 15m: oversold breakout-attempt setup
- 5m: pullback window after confirmation
- 1m: strong reversal trigger inside/after the 5m pullback

Goal:
- reduce low-quality trades from v17
- focus on stronger breakout attempts
- require stronger 1m confirmation
- tighten volatility with ADR >= 6%

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v17_1m_entry as v17

# v18 hard filters.
# Prior docs/research showed small breakout attempts were noisy, and v17's 1m
# timing did not solve setup quality by itself. This version keeps the same MTF
# pipeline but makes the setup/trigger stricter.
v17.MIN_BREAKOUT_PCT = 1.00
v17.MAX_BREAKOUT_PCT = 2.50
v17.MIN_1M_CLOSE_STRENGTH = 0.80
v17.MIN_AVG_DAILY_RANGE_PCT = 6.0

# Keep these explicit for easier experiment diffs.
v17.REQUIRE_1M_CLOSE_ABOVE_PREV_CLOSE = True
v17.REQUIRE_1M_CLOSE_ABOVE_PREV_HIGH = False


def main():
    print("\nExperiment: reversal pullback v18 hard filters full universe (with costs)")
    print("Changes versus v17:")
    print(f"- 15m breakout_attempt >= {v17.MIN_BREAKOUT_PCT:.2f}%")
    print(f"- 1m close_strength >= {v17.MIN_1M_CLOSE_STRENGTH:.2f}")
    print(f"- avg_daily_range >= {v17.MIN_AVG_DAILY_RANGE_PCT:.2f}%")
    print("- universe: all local symbols with 1D + 15m + 5m + 1m data")
    print("- no portfolio ranking yet; research includes every valid signal")
    v17.main()


if __name__ == "__main__":
    main()
