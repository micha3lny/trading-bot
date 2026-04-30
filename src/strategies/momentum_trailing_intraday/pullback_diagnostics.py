"""Diagnostics for reversal-momentum pullback entry.

This does not simulate a portfolio. It counts how many candidates pass each stage:
1. daily trend filter
2. first breakout filter
3. next-bar confirmation filter
4. confirmation close-strength filter
5. pullback entry filter

Use this before adding more strategy variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt

MIN_DAILY_TREND_PCT = -999.0
MAX_DAILY_TREND_PCT = -10.0
MIN_BREAKOUT_PCT = 0.75
MAX_BREAKOUT_PCT = 1.75
MIN_NEXT_BAR_RETURN_PCT = 0.10
MIN_CONFIRMATION_CLOSE_STRENGTH = 0.60
MAX_CONFIRMATION_CLOSE_STRENGTH = 0.80
MAX_ENTRY_RISK_PCT = 10.0

PULLBACK_LOOKAHEAD_BARS = 3
MIN_PULLBACK_FROM_CONFIRMATION_PCT = 0.15
MAX_PULLBACK_FROM_CONFIRMATION_PCT = 1.20
MIN_PULLBACK_CLOSE_STRENGTH = 0.35
REQUIRE_PULLBACK_CLOSE_ABOVE_OR_HIGH = True


@dataclass
class Counters:
    sessions: int = 0
    trend_pass: int = 0
    has_first_breakout: int = 0
    breakout_range_pass: int = 0
    next_bar_pass: int = 0
    confirmation_cs_pass: int = 0
    pullback_pass: int = 0
    rejected_no_pullback: int = 0
    rejected_pullback_below_or: int = 0
    rejected_pullback_too_small: int = 0
    rejected_pullback_too_large: int = 0
    rejected_pullback_weak_close: int = 0
    rejected_pullback_risk: int = 0


def calculate_next_bar_return(session: pd.DataFrame, position: int) -> float | None:
    if position + 1 >= len(session):
        return None
    current_close = float(session.iloc[position]["close"])
    next_close = float(session.iloc[position + 1]["close"])
    if current_close == 0:
        return None
    return (next_close - current_close) / current_close * 100.0


def diagnose_pullback(session: pd.DataFrame, confirmation_position: int, or_high: float, or_low: float, counters: Counters) -> None:
    confirmation_close = float(session.iloc[confirmation_position]["close"])
    last_position = min(len(session), confirmation_position + 1 + PULLBACK_LOOKAHEAD_BARS)

    saw_pullback_window = False
    for entry_position in range(confirmation_position + 1, last_position):
        saw_pullback_window = True
        row = session.iloc[entry_position]
        close = float(row["close"])
        low = float(row["low"])

        if REQUIRE_PULLBACK_CLOSE_ABOVE_OR_HIGH and close <= or_high:
            counters.rejected_pullback_below_or += 1
            continue

        if confirmation_close == 0:
            continue

        pullback_pct = (confirmation_close - low) / confirmation_close * 100.0
        if pullback_pct < MIN_PULLBACK_FROM_CONFIRMATION_PCT:
            counters.rejected_pullback_too_small += 1
            continue
        if pullback_pct > MAX_PULLBACK_FROM_CONFIRMATION_PCT:
            counters.rejected_pullback_too_large += 1
            continue

        close_strength = bt.calculate_close_strength(row)
        if close_strength < MIN_PULLBACK_CLOSE_STRENGTH:
            counters.rejected_pullback_weak_close += 1
            continue

        entry_risk_pct = bt.calculate_entry_risk_pct(close, or_low)
        if entry_risk_pct > MAX_ENTRY_RISK_PCT:
            counters.rejected_pullback_risk += 1
            continue

        counters.pullback_pass += 1
        return

    if not saw_pullback_window:
        counters.rejected_no_pullback += 1


def main() -> None:
    intraday_data, daily_data = bt.load_all_data()
    regimes = bt.build_market_regimes(intraday_data)
    counters = Counters()

    for symbol, intraday in intraday_data.items():
        daily = daily_data[symbol]
        for session_date, session in intraday.groupby("session_date"):
            regime = regimes.get(str(session_date))
            if bt.ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
                continue

            counters.sessions += 1
            session = session.sort_values("date").reset_index(drop=True)
            if len(session) <= bt.OPENING_RANGE_BARS + 2:
                continue

            daily_trend_pct = bt.get_daily_trend_before_session(daily, session_date)
            if not (MIN_DAILY_TREND_PCT <= daily_trend_pct <= MAX_DAILY_TREND_PCT):
                continue
            counters.trend_pass += 1

            opening = session.iloc[: bt.OPENING_RANGE_BARS]
            or_high = float(opening["high"].max())
            or_low = float(opening["low"].min())

            for breakout_position in range(bt.OPENING_RANGE_BARS, len(session) - 2):
                breakout_row = session.iloc[breakout_position]
                breakout_close = float(breakout_row["close"])
                if breakout_close <= or_high:
                    continue

                counters.has_first_breakout += 1
                breakout_pct = bt.calculate_breakout_pct(breakout_close, or_high)
                if not (MIN_BREAKOUT_PCT <= breakout_pct <= MAX_BREAKOUT_PCT):
                    break
                counters.breakout_range_pass += 1

                next_bar_return_pct = calculate_next_bar_return(session, breakout_position)
                if next_bar_return_pct is None or next_bar_return_pct < MIN_NEXT_BAR_RETURN_PCT:
                    break
                counters.next_bar_pass += 1

                confirmation_position = breakout_position + 1
                confirmation_cs = bt.calculate_close_strength(session.iloc[confirmation_position])
                if not (MIN_CONFIRMATION_CLOSE_STRENGTH <= confirmation_cs <= MAX_CONFIRMATION_CLOSE_STRENGTH):
                    break
                counters.confirmation_cs_pass += 1

                diagnose_pullback(session, confirmation_position, or_high, or_low, counters)
                break

    print("\nPullback diagnostics")
    print("--------------------")
    for key, value in counters.__dict__.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
