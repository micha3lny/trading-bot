"""Reversal pullback v23: scaled entry with v22 smart exit.

Builds on v20/v22 plus entry funnel diagnostics.

Diagnostics showed the main bottleneck is the 15m setup:
- many candidates are rejected as breakout_too_small
- very few pass the narrow 1.0% - 1.8% breakout window

Entry changes versus v20:
- widen 15m breakout attempt to 0.8% - 2.2%
- allow slightly stronger 5m pullback close strength
- allow slightly stronger 1m trigger close strength

Exit stays v22:
- TP 3.0%
- SL 1.0%
- trailing starts at +1.5%, stop 1.0%
- time exit after 60 bars if PnL < 0.5%

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v22_smart_exit as v22

# Scaled-entry settings.
v22.v20.v17.MIN_BREAKOUT_PCT = 0.80
v22.v20.v17.MAX_BREAKOUT_PCT = 2.20
v22.v20.v17.MAX_5M_CLOSE_STRENGTH = 0.70
v22.v20.MAX_1M_CLOSE_STRENGTH = 0.92


def main():
    print("\nExperiment: reversal pullback v23 scaled entry + v22 smart exit")
    print("Entry changes versus v20:")
    print(f"- 15m breakout_attempt: {v22.v20.v17.MIN_BREAKOUT_PCT:.2f}% - {v22.v20.v17.MAX_BREAKOUT_PCT:.2f}%")
    print(f"- 5m max close_strength: {v22.v20.v17.MAX_5M_CLOSE_STRENGTH:.2f}")
    print(f"- 1m close_strength: {v22.v20.v17.MIN_1M_CLOSE_STRENGTH:.2f} - {v22.v20.MAX_1M_CLOSE_STRENGTH:.2f}")
    print("Exit: v22 smart hybrid exit")
    v22.main()


if __name__ == "__main__":
    main()
