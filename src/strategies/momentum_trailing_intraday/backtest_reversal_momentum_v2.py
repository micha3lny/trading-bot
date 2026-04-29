"""Reversal-momentum v2.

Tighter filters based on analysis:
- daily_trend <= -10%
- breakout in [1.0%, 1.75%]
- next_bar_return >= 0.1%
- close_strength in [0.6, 0.8]
- entry_risk <= 6%
"""

from __future__ import annotations

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades

MIN_DAILY_TREND_PCT = -999.0
MAX_DAILY_TREND_PCT = -10.0
MIN_BREAKOUT_PCT = 1.0
MAX_BREAKOUT_PCT = 1.75
MIN_NEXT_BAR_RETURN_PCT = 0.10
MIN_CLOSE_STRENGTH = 0.6
MAX_CLOSE_STRENGTH = 0.8
MAX_ENTRY_RISK_PCT = 6.0
MAX_POSITIONS_PER_DAY = 3


def calculate_next_bar_return(session: pd.DataFrame, position: int):
    if position + 1 >= len(session):
        return None
    c = float(session.iloc[position]["close"])
    n = float(session.iloc[position + 1]["close"])
    if c == 0:
        return None
    return (n - c) / c * 100.0


def find_candidate(session: pd.DataFrame, daily_trend_pct: float):
    if len(session) <= bt.OPENING_RANGE_BARS + 1:
        return None
    if not (MIN_DAILY_TREND_PCT <= daily_trend_pct <= MAX_DAILY_TREND_PCT):
        return None

    opening = session.iloc[: bt.OPENING_RANGE_BARS]
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())

    for pos in range(bt.OPENING_RANGE_BARS, len(session) - 1):
        row = session.iloc[pos]
        close = float(row["close"])
        if close <= or_high:
            continue

        breakout = bt.calculate_breakout_pct(close, or_high)
        cs = bt.calculate_close_strength(row)
        risk = bt.calculate_entry_risk_pct(close, or_low)
        nret = calculate_next_bar_return(session, pos)
        if nret is None:
            return None

        if (
            MIN_BREAKOUT_PCT <= breakout <= MAX_BREAKOUT_PCT
            and MIN_CLOSE_STRENGTH <= cs <= MAX_CLOSE_STRENGTH
            and risk <= MAX_ENTRY_RISK_PCT
            and nret >= MIN_NEXT_BAR_RETURN_PCT
        ):
            return pos + 1, breakout, cs, risk
        return None
    return None


def backtest_symbol(symbol, intraday, daily, regimes):
    trades = []
    for d, sess in intraday.groupby("session_date"):
        reg = regimes.get(str(d))
        if bt.ENABLE_MARKET_REGIME_FILTER and (reg is None or not reg.tradable):
            continue
        sess = sess.sort_values("date").reset_index(drop=True)
        trend = bt.get_daily_trend_before_session(daily, d)
        cand = find_candidate(sess, trend)
        if not cand:
            continue
        pos, breakout, cs, risk = cand
        trades.append(
            bt.simulate_exit(symbol, sess, pos, breakout, cs, risk, trend, "reversal_momentum_v2")
        )
    return trades


def main():
    intraday_data, daily_data = bt.load_all_data()
    regimes = bt.build_market_regimes(intraday_data)
    all_trades = []
    for s, intr in intraday_data.items():
        all_trades.extend(backtest_symbol(s, intr, daily_data[s], regimes))

    trades = bt.apply_position_sizing(all_trades)

    print("\nExperiment: reversal momentum v2")
    print(
        f"Filters: trend<= {MAX_DAILY_TREND_PCT}, breakout=[{MIN_BREAKOUT_PCT},{MAX_BREAKOUT_PCT}], "
        f"cs=[{MIN_CLOSE_STRENGTH},{MAX_CLOSE_STRENGTH}], risk<={MAX_ENTRY_RISK_PCT}, nret>={MIN_NEXT_BAR_RETURN_PCT}"
    )

    bt.summarize(trades, regimes)
    df = export_trades(trades)
    analyze(df)


if __name__ == "__main__":
    main()
