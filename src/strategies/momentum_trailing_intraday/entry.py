"""Entry logic for Momentum Trailing Intraday strategy.

This module does not place orders.
It only checks whether ranked symbols have a breakout entry signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.load_market_data import load_market_data_bundle
from src.strategies.momentum_trailing_intraday.ranking import (
    RankingRow,
    get_last_intraday_session,
    rank_symbols,
)


OPENING_RANGE_BARS = 4  # 4 x 15m = first hour
MIN_INTRADAY_MOMENTUM_PCT = 0.5
MIN_VOLUME_RATIO = 0.8
TOP_N = 10


@dataclass(frozen=True)
class EntrySignal:
    symbol: str
    has_signal: bool
    reason: str
    last_price: float
    opening_range_high: float
    opening_range_low: float
    intraday_momentum_pct: float
    volume_ratio: float
    ranking_score: float


def check_opening_range_breakout(row: RankingRow, df_intraday: pd.DataFrame) -> EntrySignal:
    """Check whether the latest session broke above the opening range high."""
    latest_session = get_last_intraday_session(df_intraday)

    if len(latest_session) <= OPENING_RANGE_BARS:
        return EntrySignal(
            symbol=row.symbol,
            has_signal=False,
            reason="not enough intraday bars after opening range",
            last_price=0.0,
            opening_range_high=0.0,
            opening_range_low=0.0,
            intraday_momentum_pct=row.intraday_momentum_pct,
            volume_ratio=row.volume_ratio,
            ranking_score=row.score,
        )

    if row.intraday_momentum_pct < MIN_INTRADAY_MOMENTUM_PCT:
        return EntrySignal(
            symbol=row.symbol,
            has_signal=False,
            reason=f"intraday momentum below {MIN_INTRADAY_MOMENTUM_PCT:.2f}%",
            last_price=float(latest_session.iloc[-1]["close"]),
            opening_range_high=float(latest_session.iloc[:OPENING_RANGE_BARS]["high"].max()),
            opening_range_low=float(latest_session.iloc[:OPENING_RANGE_BARS]["low"].min()),
            intraday_momentum_pct=row.intraday_momentum_pct,
            volume_ratio=row.volume_ratio,
            ranking_score=row.score,
        )

    if row.volume_ratio < MIN_VOLUME_RATIO:
        return EntrySignal(
            symbol=row.symbol,
            has_signal=False,
            reason=f"volume ratio below {MIN_VOLUME_RATIO:.2f}",
            last_price=float(latest_session.iloc[-1]["close"]),
            opening_range_high=float(latest_session.iloc[:OPENING_RANGE_BARS]["high"].max()),
            opening_range_low=float(latest_session.iloc[:OPENING_RANGE_BARS]["low"].min()),
            intraday_momentum_pct=row.intraday_momentum_pct,
            volume_ratio=row.volume_ratio,
            ranking_score=row.score,
        )

    opening_range = latest_session.iloc[:OPENING_RANGE_BARS]
    opening_range_high = float(opening_range["high"].max())
    opening_range_low = float(opening_range["low"].min())
    last_price = float(latest_session.iloc[-1]["close"])

    has_breakout = last_price > opening_range_high

    return EntrySignal(
        symbol=row.symbol,
        has_signal=has_breakout,
        reason="breakout above opening range" if has_breakout else "no breakout above opening range",
        last_price=last_price,
        opening_range_high=opening_range_high,
        opening_range_low=opening_range_low,
        intraday_momentum_pct=row.intraday_momentum_pct,
        volume_ratio=row.volume_ratio,
        ranking_score=row.score,
    )


def find_entry_signals(top_n: int = TOP_N) -> list[EntrySignal]:
    signals: list[EntrySignal] = []
    ranking = rank_symbols()[:top_n]

    for row in ranking:
        bundle = load_market_data_bundle(row.symbol)
        signal = check_opening_range_breakout(row, bundle.intraday)
        signals.append(signal)

    return signals


def main() -> None:
    signals = find_entry_signals()

    print("\nEntry signals: Momentum Trailing Intraday\n")
    print("Rank | Symbol | Signal | Last | OR High | OR Low | Mom % | Vol | Score | Reason")
    print("--------------------------------------------------------------------------------")

    for i, signal in enumerate(signals, start=1):
        signal_text = "YES" if signal.has_signal else "NO"
        print(
            f"{i:>4} | "
            f"{signal.symbol:<6} | "
            f"{signal_text:<6} | "
            f"{signal.last_price:>8.2f} | "
            f"{signal.opening_range_high:>7.2f} | "
            f"{signal.opening_range_low:>6.2f} | "
            f"{signal.intraday_momentum_pct:>5.2f} | "
            f"{signal.volume_ratio:>4.2f} | "
            f"{signal.ranking_score:>5.2f} | "
            f"{signal.reason}"
        )


if __name__ == "__main__":
    main()
