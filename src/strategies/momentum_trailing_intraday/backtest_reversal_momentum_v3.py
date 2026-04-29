"""Reversal-momentum v3.

v2 was too restrictive and produced no trades. This version keeps the core edge:
- deeply negative daily trend
- meaningful first breakout
- positive next-bar confirmation

But relaxes close-strength and risk so the strategy can actually trade.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades

MIN_DAILY_TREND_PCT = -999.0
MAX_DAILY_TREND_PCT = -10.0
MIN_BREAKOUT_PCT = 0.75
MAX_BREAKOUT_PCT = 1.75
MIN_NEXT_BAR_RETURN_PCT = 0.10
MIN_CLOSE_STRENGTH = 0.50
MAX_CLOSE_STRENGTH = 1.00
MAX_ENTRY_RISK_PCT = 10.0
MAX_POSITIONS_PER_DAY = 3


def calculate_next_bar_return(session: pd.DataFrame, position: int):
    if position + 1 >= len(session):
        return None
    current_close = float(session.iloc[position]["close"])
    next_close = float(session.iloc[position + 1]["close"])
    if current_close == 0:
        return None
    return (next_close - current_close) / current_close * 100.0


def find_candidate(session: pd.DataFrame, daily_trend_pct: float):
    if len(session) <= bt.OPENING_RANGE_BARS + 1:
        return None
    if not (MIN_DAILY_TREND_PCT <= daily_trend_pct <= MAX_DAILY_TREND_PCT):
        return None

    opening = session.iloc[: bt.OPENING_RANGE_BARS]
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())

    for position in range(bt.OPENING_RANGE_BARS, len(session) - 1):
        row = session.iloc[position]
        close = float(row["close"])
        if close <= or_high:
            continue

        breakout_pct = bt.calculate_breakout_pct(close, or_high)
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, or_low)
        next_bar_return_pct = calculate_next_bar_return(session, position)
        if next_bar_return_pct is None:
            return None

        if (
            MIN_BREAKOUT_PCT <= breakout_pct <= MAX_BREAKOUT_PCT
            and MIN_CLOSE_STRENGTH <= close_strength <= MAX_CLOSE_STRENGTH
            and entry_risk_pct <= MAX_ENTRY_RISK_PCT
            and next_bar_return_pct >= MIN_NEXT_BAR_RETURN_PCT
        ):
            # Enter on the confirmation bar.
            confirmation_position = position + 1
            confirmation_close = float(session.iloc[confirmation_position]["close"])
            return (
                confirmation_position,
                bt.calculate_breakout_pct(confirmation_close, or_high),
                bt.calculate_close_strength(session.iloc[confirmation_position]),
                bt.calculate_entry_risk_pct(confirmation_close, or_low),
            )

        # First breakout only.
        return None

    return None


def backtest_symbol(symbol, intraday, daily, regimes):
    trades = []
    for session_date, session in intraday.groupby("session_date"):
        regime = regimes.get(str(session_date))
        if bt.ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
            continue

        session = session.sort_values("date").reset_index(drop=True)
        daily_trend_pct = bt.get_daily_trend_before_session(daily, session_date)
        candidate = find_candidate(session, daily_trend_pct)
        if not candidate:
            continue

        entry_position, breakout_pct, close_strength, entry_risk_pct = candidate
        trades.append(
            bt.simulate_exit(
                symbol,
                session,
                entry_position,
                breakout_pct,
                close_strength,
                entry_risk_pct,
                daily_trend_pct,
                "reversal_momentum_v3",
            )
        )

    return trades


def main():
    intraday_data, daily_data = bt.load_all_data()
    regimes = bt.build_market_regimes(intraday_data)

    all_trades = []
    for symbol, intraday in intraday_data.items():
        all_trades.extend(backtest_symbol(symbol, intraday, daily_data[symbol], regimes))

    trades = bt.apply_position_sizing(all_trades)

    print("\nExperiment: reversal momentum v3")
    print(
        f"Filters: trend<= {MAX_DAILY_TREND_PCT}, breakout=[{MIN_BREAKOUT_PCT},{MAX_BREAKOUT_PCT}], "
        f"cs=[{MIN_CLOSE_STRENGTH},{MAX_CLOSE_STRENGTH}], risk<={MAX_ENTRY_RISK_PCT}, nret>={MIN_NEXT_BAR_RETURN_PCT}"
    )

    bt.summarize(trades, regimes)
    df = export_trades(trades)
    analyze(df)


if __name__ == "__main__":
    main()
