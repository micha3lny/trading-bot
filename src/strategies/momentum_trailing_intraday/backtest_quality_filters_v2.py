"""Relaxed quality-filter experiment v2 for Momentum Trailing Intraday.

Goal: keep signal alive while removing worst setups.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades


MIN_BREAKOUT_PCT = 0.50
MAX_ENTRY_RISK_PCT = 1.75
MIN_DAILY_TREND_PCT = 0.0
MAX_DAILY_TREND_PCT = 5.0
MAX_CLOSE_STRENGTH = 0.90


def is_valid_breakout_candidate(close, opening_range_high, opening_range_low, row):
    breakout_pct = bt.calculate_breakout_pct(close, opening_range_high)
    close_strength = bt.calculate_close_strength(row)
    entry_risk_pct = bt.calculate_entry_risk_pct(close, opening_range_low)

    is_valid = (
        breakout_pct >= MIN_BREAKOUT_PCT
        and bt.MIN_CLOSE_STRENGTH <= close_strength <= MAX_CLOSE_STRENGTH
        and entry_risk_pct <= MAX_ENTRY_RISK_PCT
    )

    return is_valid, breakout_pct, close_strength, entry_risk_pct


def backtest_symbol(symbol, intraday, daily, market_regimes):
    trades = []

    for session_date, session in intraday.groupby("session_date"):
        regime = market_regimes.get(str(session_date))
        if bt.ENABLE_MARKET_REGIME_FILTER and (regime is None or not regime.tradable):
            continue

        session = session.sort_values("date").reset_index(drop=True)
        entry = bt.find_entry_bar(session)
        if entry is None:
            continue

        entry_position, breakout_pct, close_strength, entry_risk_pct, setup_type = entry
        daily_trend_pct = bt.get_daily_trend_before_session(daily, session_date)

        if daily_trend_pct < MIN_DAILY_TREND_PCT or daily_trend_pct > MAX_DAILY_TREND_PCT:
            continue

        trades.append(
            bt.simulate_exit(
                symbol,
                session,
                entry_position,
                breakout_pct,
                close_strength,
                entry_risk_pct,
                daily_trend_pct,
                setup_type,
            )
        )

    return trades


def main():
    bt.MIN_BREAKOUT_PCT = MIN_BREAKOUT_PCT
    bt.MAX_ENTRY_RISK_PCT = MAX_ENTRY_RISK_PCT
    bt.MIN_DAILY_TREND_PCT = MIN_DAILY_TREND_PCT
    bt.is_valid_breakout_candidate = is_valid_breakout_candidate

    intraday_data, daily_data = bt.load_all_data()
    market_regimes = bt.build_market_regimes(intraday_data)

    all_trades = []
    for symbol, intraday in intraday_data.items():
        all_trades.extend(backtest_symbol(symbol, intraday, daily_data[symbol], market_regimes))

    trades = bt.apply_position_sizing(all_trades)

    print("\nExperiment v2: relaxed quality filters")
    print(
        f"Filters: breakout>={MIN_BREAKOUT_PCT}, risk<={MAX_ENTRY_RISK_PCT}, trend<= {MAX_DAILY_TREND_PCT}, close<= {MAX_CLOSE_STRENGTH}"
    )

    bt.summarize(trades, market_regimes)

    df = export_trades(trades)
    analyze(df)


if __name__ == "__main__":
    main()
