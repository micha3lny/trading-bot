"""Reversal-momentum first-breakout backtest.

Hypothesis from first_breakout_analysis:
- Generic first breakout has no edge.
- The strongest edge appears after deeply negative daily trend.
- Best breakout bucket was roughly 1.0% to 1.5%.
- Positive next-bar return is strongly predictive, so entry waits for confirmation.

No orders are placed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategies.momentum_trailing_intraday import backtest as bt
from src.strategies.momentum_trailing_intraday.analysis import analyze, export_trades


MIN_DAILY_TREND_PCT = -999.0
MAX_DAILY_TREND_PCT = -10.0
MIN_BREAKOUT_PCT = 0.75
MAX_BREAKOUT_PCT = 1.75
MIN_NEXT_BAR_RETURN_PCT = 0.10
MAX_ENTRY_RISK_PCT = 10.0
MAX_POSITIONS_PER_DAY = 3


@dataclass(frozen=True)
class Candidate:
    position: int
    breakout_pct: float
    close_strength: float
    entry_risk_pct: float
    daily_trend_pct: float
    next_bar_return_pct: float


def calculate_next_bar_return(session: pd.DataFrame, position: int) -> float | None:
    if position + 1 >= len(session):
        return None
    current_close = float(session.iloc[position]["close"])
    next_close = float(session.iloc[position + 1]["close"])
    if current_close == 0:
        return None
    return (next_close - current_close) / current_close * 100.0


def find_first_reversal_breakout(session: pd.DataFrame, daily_trend_pct: float) -> Candidate | None:
    if len(session) <= bt.OPENING_RANGE_BARS + 1:
        return None

    if daily_trend_pct < MIN_DAILY_TREND_PCT or daily_trend_pct > MAX_DAILY_TREND_PCT:
        return None

    opening_range = session.iloc[: bt.OPENING_RANGE_BARS]
    opening_range_high = float(opening_range["high"].max())
    opening_range_low = float(opening_range["low"].min())

    for position in range(bt.OPENING_RANGE_BARS, len(session) - 1):
        row = session.iloc[position]
        close = float(row["close"])
        if close <= opening_range_high:
            continue

        breakout_pct = bt.calculate_breakout_pct(close, opening_range_high)
        close_strength = bt.calculate_close_strength(row)
        entry_risk_pct = bt.calculate_entry_risk_pct(close, opening_range_low)
        next_bar_return_pct = calculate_next_bar_return(session, position)

        if next_bar_return_pct is None:
            return None

        if (
            MIN_BREAKOUT_PCT <= breakout_pct <= MAX_BREAKOUT_PCT
            and entry_risk_pct <= MAX_ENTRY_RISK_PCT
            and next_bar_return_pct >= MIN_NEXT_BAR_RETURN_PCT
        ):
            return Candidate(
                position=position + 1,
                breakout_pct=bt.calculate_breakout_pct(float(session.iloc[position + 1]["close"]), opening_range_high),
                close_strength=bt.calculate_close_strength(session.iloc[position + 1]),
                entry_risk_pct=bt.calculate_entry_risk_pct(float(session.iloc[position + 1]["close"]), opening_range_low),
                daily_trend_pct=daily_trend_pct,
                next_bar_return_pct=next_bar_return_pct,
            )

        # first breakout only; do not keep scanning repeated bars above OR high
        return None

    return None


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
        daily_trend_pct = bt.get_daily_trend_before_session(daily, session_date)
        candidate = find_first_reversal_breakout(session, daily_trend_pct)
        if candidate is None:
            continue

        trades.append(
            bt.simulate_exit(
                symbol=symbol,
                session=session,
                entry_position=candidate.position,
                breakout_pct=candidate.breakout_pct,
                close_strength=candidate.close_strength,
                entry_risk_pct=candidate.entry_risk_pct,
                daily_trend_pct=candidate.daily_trend_pct,
                setup_type="reversal_momentum_first_breakout",
            )
        )

    return trades


def rank_trades(day_trades: list[bt.BacktestTrade]) -> list[bt.BacktestTrade]:
    return sorted(
        day_trades,
        key=lambda trade: (
            -abs(trade.daily_trend_pct),
            trade.breakout_pct,
            -trade.entry_risk_pct,
        ),
        reverse=True,
    )[:MAX_POSITIONS_PER_DAY]


def apply_position_sizing(trades: list[bt.BacktestTrade]) -> list[bt.BacktestTrade]:
    original_rank_trades = bt.rank_trades
    original_max_positions = bt.MAX_POSITIONS_PER_DAY
    bt.rank_trades = rank_trades
    bt.MAX_POSITIONS_PER_DAY = MAX_POSITIONS_PER_DAY
    try:
        return bt.apply_position_sizing(trades)
    finally:
        bt.rank_trades = original_rank_trades
        bt.MAX_POSITIONS_PER_DAY = original_max_positions


def main() -> None:
    intraday_data, daily_data = bt.load_all_data()
    market_regimes = bt.build_market_regimes(intraday_data)

    all_trades: list[bt.BacktestTrade] = []
    for symbol, intraday in intraday_data.items():
        all_trades.extend(backtest_symbol(symbol, intraday, daily_data[symbol], market_regimes))

    trades = apply_position_sizing(all_trades)

    print("\nExperiment: reversal momentum first breakout")
    print(
        f"Filters: daily_trend <= {MAX_DAILY_TREND_PCT:.2f}%, "
        f"breakout=[{MIN_BREAKOUT_PCT:.2f}%, {MAX_BREAKOUT_PCT:.2f}%], "
        f"next_bar_return >= {MIN_NEXT_BAR_RETURN_PCT:.2f}%, "
        f"max_entry_risk={MAX_ENTRY_RISK_PCT:.2f}%"
    )

    bt.summarize(trades, market_regimes)
    trades_df = export_trades(trades)
    analyze(trades_df)


if __name__ == "__main__":
    main()
