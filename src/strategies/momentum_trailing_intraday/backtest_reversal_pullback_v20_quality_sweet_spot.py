"""Reversal pullback v20: quality sweet spot.

Builds on v19 results.

v19 showed:
- the 0.75-0.80 1m close-strength bucket was weak
- the 0.80-0.90 bucket carried the edge
- day filter did not change results because v19 already produced max 1 signal/day

Hypothesis:
- keep the breakout sweet spot: 1.0% - 1.8%
- tighten 1m close strength to 0.80 - 0.90
- keep ADR >= 5.0%

Research mode: all valid signals are included, no portfolio ranking.
No orders are placed.
"""

from __future__ import annotations

from src.strategies.momentum_trailing_intraday import backtest_reversal_pullback_v17_1m_entry as v17

# 15m setup sweet spot.
v17.MIN_BREAKOUT_PCT = 1.00
v17.MAX_BREAKOUT_PCT = 1.80

# 1m trigger quality sweet spot.
v17.MIN_1M_CLOSE_STRENGTH = 0.80
v17.REQUIRE_1M_CLOSE_ABOVE_PREV_CLOSE = True
v17.REQUIRE_1M_CLOSE_ABOVE_PREV_HIGH = False

MAX_1M_CLOSE_STRENGTH = 0.90
_original_find_1m_entry_trigger = v17.find_1m_entry_trigger


def find_1m_entry_trigger_with_upper_bound(session_1m, pullback):
    pullback_time = v17.pd.Timestamp(pullback["pullback_time"])
    after_pullback = session_1m[session_1m["date"] >= pullback_time].copy()
    if after_pullback.empty:
        return None

    after_pullback = after_pullback.sort_values("date").reset_index(drop=True)
    window = after_pullback.iloc[: v17.TRIGGER_LOOKAHEAD_1M_BARS].reset_index(drop=True)

    for idx in range(1, len(window)):
        row = window.iloc[idx]
        close_strength = v17.bt.calculate_close_strength(row)
        if close_strength > MAX_1M_CLOSE_STRENGTH:
            continue

        candidate_window = window.iloc[idx - 1 : idx + 1].copy().reset_index(drop=True)
        result = _original_find_1m_entry_trigger(candidate_window, pullback)
        if result is not None:
            return result

    return None


v17.find_1m_entry_trigger = find_1m_entry_trigger_with_upper_bound

# Volatility filter.
v17.MIN_AVG_DAILY_RANGE_PCT = 5.0


def main():
    print("\nExperiment: reversal pullback v20 quality sweet spot full universe (with costs)")
    print("Hypothesis:")
    print(f"- 15m breakout_attempt: {v17.MIN_BREAKOUT_PCT:.2f}% - {v17.MAX_BREAKOUT_PCT:.2f}%")
    print(f"- 1m close_strength: {v17.MIN_1M_CLOSE_STRENGTH:.2f} - {MAX_1M_CLOSE_STRENGTH:.2f}")
    print(f"- avg_daily_range >= {v17.MIN_AVG_DAILY_RANGE_PCT:.2f}%")
    print("- universe: all local symbols with 1D + 15m + 5m + 1m data")
    print("- no portfolio ranking yet; research includes every valid signal")
    v17.main()


if __name__ == "__main__":
    main()
