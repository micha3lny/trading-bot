"""Quality-filtered experiment for Momentum Trailing Intraday backtest.

This runner keeps the current backtest engine intact and applies the first
trade-level-analysis driven filters:
- stronger confirmed breakout
- lower entry risk
- avoid overheated daily trend
- avoid euphoric candle closes
"""

from __future__ import annotations

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades


# Trade-level-analysis driven filters.
MIN_BREAKOUT_PCT = 0.50
MAX_ENTRY_RISK_PCT = 1.50
MIN_DAILY_TREND_PCT = 0.0
MAX_DAILY_TREND_PCT = 2.0
MAX_CLOSE_STRENGTH = 0.85


def is_valid_breakout_candidate(
    close: float,
    opening_range_high: float,
    opening_range_low: float,
    row: pd.Series,
) -> tuple[bool, float, float, float]:
    breakout_pct = bt.calculate_breakout_pct(close, opening_range_high)
    close_strength = bt.calculate_close_strength(row)
    entry_risk_pct = bt.calculate_entry_risk_pct(close, opening_range_low)
    is_valid = (
        breakout_pct >= MIN_BREAKOUT_PCT
        and bt.MIN_CLOSE_STRENGTH <= close_strength <= MAX_CLOSE_STRENGTH
        and entry_risk_pct <= MAX_ENTRY_RISK_PCT
    )
    return is_valid, breakout_pct, close_strength, entry_risk_pct


def backtest_symbol(
    symbol: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    market_regimes: dict[str, bt.MarketRegime],
) -> list[bt.BacktestTrade]:
    trades: list[bt.BacktestTrade] = []
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


def main() -> None:
    # Patch the shared backtest engine for this experiment only.
    bt.MIN_BREAKOUT_PCT = MIN_BREAKOUT_PCT
    bt.MAX_ENTRY_RISK_PCT = MAX_ENTRY_RISK_PCT
    bt.MIN_DAILY_TREND_PCT = MIN_DAILY_TREND_PCT
    bt.is_valid_breakout_candidate = is_valid_breakout_candidate

    intraday_data, daily_data = bt.load_all_data()
    market_regimes = bt.build_market_regimes(intraday_data)
    all_trades: list[bt.BacktestTrade] = []
    for symbol, intraday in intraday_data.items():
        all_trades.extend(backtest_symbol(symbol, intraday, daily_data[symbol], market_regimes))

    trades = bt.apply_position_sizing(all_trades)
    print("\nExperiment: quality-filtered follow-through breakout")
    print(
        f"Quality filters: min_breakout={MIN_BREAKOUT_PCT:.2f}%, "
        f"max_entry_risk={MAX_ENTRY_RISK_PCT:.2f}%, "
        f"daily_trend=[{MIN_DAILY_TREND_PCT:.2f}%, {MAX_DAILY_TREND_PCT:.2f}%], "
        f"close_strength=[{bt.MIN_CLOSE_STRENGTH:.2f}, {MAX_CLOSE_STRENGTH:.2f}]"
    )
    bt.summarize(trades, market_regimes)

    trades_df = export_trades(trades)
    analyze(trades_df)


if __name__ == "__main__":
    main()
