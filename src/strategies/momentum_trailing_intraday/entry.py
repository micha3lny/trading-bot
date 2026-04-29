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


# Strategy-specific selection settings.
# Ranking is NOT global. These limits apply only to Momentum Trailing Intraday.
RANKING_CANDIDATES_LIMIT = 10
MAX_POSITIONS = 3

# Entry setup settings.
OPENING_RANGE_BARS = 4  # 4 x 15m = first hour
MIN_INTRADAY_MOMENTUM_PCT = 0.5
MIN_VOLUME_RATIO = 0.8
MAX_ENTRY_RISK_PCT = 4.0


@dataclass(frozen=True)
class EntrySignal:
    symbol: str
    has_signal: bool
    is_final_pick: bool
    reason: str
    last_price: float
    opening_range_high: float
    opening_range_low: float
    breakout_pct: float
    entry_risk_pct: float
    intraday_momentum_pct: float
    volume_ratio: float
    ranking_score: float


def calculate_breakout_pct(last_price: float, opening_range_high: float) -> float:
    if opening_range_high == 0:
        return 0.0
    return (last_price - opening_range_high) / opening_range_high * 100.0


def calculate_entry_risk_pct(last_price: float, opening_range_low: float) -> float:
    """Approximate risk if initial stop is placed below opening range low."""
    if last_price == 0:
        return 0.0
    return (last_price - opening_range_low) / last_price * 100.0


def build_signal(
    row: RankingRow,
    has_signal: bool,
    reason: str,
    last_price: float,
    opening_range_high: float,
    opening_range_low: float,
) -> EntrySignal:
    breakout_pct = calculate_breakout_pct(last_price, opening_range_high)
    entry_risk_pct = calculate_entry_risk_pct(last_price, opening_range_low)

    return EntrySignal(
        symbol=row.symbol,
        has_signal=has_signal,
        is_final_pick=False,
        reason=reason,
        last_price=last_price,
        opening_range_high=opening_range_high,
        opening_range_low=opening_range_low,
        breakout_pct=breakout_pct,
        entry_risk_pct=entry_risk_pct,
        intraday_momentum_pct=row.intraday_momentum_pct,
        volume_ratio=row.volume_ratio,
        ranking_score=row.score,
    )


def get_opening_range_values(latest_session: pd.DataFrame) -> tuple[float, float, float]:
    opening_range = latest_session.iloc[:OPENING_RANGE_BARS]
    opening_range_high = float(opening_range["high"].max())
    opening_range_low = float(opening_range["low"].min())
    last_price = float(latest_session.iloc[-1]["close"])

    return last_price, opening_range_high, opening_range_low


def check_opening_range_breakout(row: RankingRow, df_intraday: pd.DataFrame) -> EntrySignal:
    """Check whether the latest session broke above the opening range high."""
    latest_session = get_last_intraday_session(df_intraday)

    if len(latest_session) <= OPENING_RANGE_BARS:
        return build_signal(
            row=row,
            has_signal=False,
            reason="not enough intraday bars after opening range",
            last_price=0.0,
            opening_range_high=0.0,
            opening_range_low=0.0,
        )

    last_price, opening_range_high, opening_range_low = get_opening_range_values(latest_session)

    if row.intraday_momentum_pct < MIN_INTRADAY_MOMENTUM_PCT:
        return build_signal(
            row=row,
            has_signal=False,
            reason=f"intraday momentum below {MIN_INTRADAY_MOMENTUM_PCT:.2f}%",
            last_price=last_price,
            opening_range_high=opening_range_high,
            opening_range_low=opening_range_low,
        )

    if row.volume_ratio < MIN_VOLUME_RATIO:
        return build_signal(
            row=row,
            has_signal=False,
            reason=f"volume ratio below {MIN_VOLUME_RATIO:.2f}",
            last_price=last_price,
            opening_range_high=opening_range_high,
            opening_range_low=opening_range_low,
        )

    entry_risk_pct = calculate_entry_risk_pct(last_price, opening_range_low)
    if entry_risk_pct > MAX_ENTRY_RISK_PCT:
        return build_signal(
            row=row,
            has_signal=False,
            reason=f"entry risk above {MAX_ENTRY_RISK_PCT:.2f}%",
            last_price=last_price,
            opening_range_high=opening_range_high,
            opening_range_low=opening_range_low,
        )

    has_breakout = last_price > opening_range_high

    return build_signal(
        row=row,
        has_signal=has_breakout,
        reason="breakout above opening range" if has_breakout else "no breakout above opening range",
        last_price=last_price,
        opening_range_high=opening_range_high,
        opening_range_low=opening_range_low,
    )


def mark_final_picks(signals: list[EntrySignal], max_positions: int = MAX_POSITIONS) -> list[EntrySignal]:
    """Mark best trade candidates from valid entry signals.

    Final picks are selected only from YES signals and remain sorted by
    strategy-specific ranking score.
    """
    selected_symbols = {
        signal.symbol
        for signal in sorted(
            [signal for signal in signals if signal.has_signal],
            key=lambda signal: signal.ranking_score,
            reverse=True,
        )[:max_positions]
    }

    return [
        EntrySignal(
            symbol=signal.symbol,
            has_signal=signal.has_signal,
            is_final_pick=signal.symbol in selected_symbols,
            reason=signal.reason,
            last_price=signal.last_price,
            opening_range_high=signal.opening_range_high,
            opening_range_low=signal.opening_range_low,
            breakout_pct=signal.breakout_pct,
            entry_risk_pct=signal.entry_risk_pct,
            intraday_momentum_pct=signal.intraday_momentum_pct,
            volume_ratio=signal.volume_ratio,
            ranking_score=signal.ranking_score,
        )
        for signal in signals
    ]


def find_entry_signals(
    ranking_candidates_limit: int = RANKING_CANDIDATES_LIMIT,
    max_positions: int = MAX_POSITIONS,
) -> list[EntrySignal]:
    signals: list[EntrySignal] = []
    ranking = rank_symbols()[:ranking_candidates_limit]

    for row in ranking:
        bundle = load_market_data_bundle(row.symbol)
        signal = check_opening_range_breakout(row, bundle.intraday)
        signals.append(signal)

    return mark_final_picks(signals, max_positions=max_positions)


def print_final_picks(signals: list[EntrySignal]) -> None:
    final_picks = [signal for signal in signals if signal.is_final_pick]

    print(f"\nFinal picks: top {MAX_POSITIONS} valid YES signals\n")

    if not final_picks:
        print("No final picks. No valid YES signals passed the filters.")
        return

    print("Pick | Symbol | Score | Last | Breakout % | Risk % | Mom % | Vol")
    print("---------------------------------------------------------------")

    for i, signal in enumerate(final_picks, start=1):
        print(
            f"{i:>4} | "
            f"{signal.symbol:<6} | "
            f"{signal.ranking_score:>5.2f} | "
            f"{signal.last_price:>8.2f} | "
            f"{signal.breakout_pct:>10.2f} | "
            f"{signal.entry_risk_pct:>6.2f} | "
            f"{signal.intraday_momentum_pct:>5.2f} | "
            f"{signal.volume_ratio:>4.2f}"
        )


def main() -> None:
    signals = find_entry_signals()

    print("\nEntry signals: Momentum Trailing Intraday\n")
    print(
        "Rank | Symbol | Signal | Pick | Last | OR High | OR Low | "
        "Breakout % | Risk % | Mom % | Vol | Score | Reason"
    )
    print(
        "---------------------------------------------------------------------------------------------"
    )

    for i, signal in enumerate(signals, start=1):
        signal_text = "YES" if signal.has_signal else "NO"
        pick_text = "YES" if signal.is_final_pick else ""
        print(
            f"{i:>4} | "
            f"{signal.symbol:<6} | "
            f"{signal_text:<6} | "
            f"{pick_text:<4} | "
            f"{signal.last_price:>8.2f} | "
            f"{signal.opening_range_high:>7.2f} | "
            f"{signal.opening_range_low:>6.2f} | "
            f"{signal.breakout_pct:>10.2f} | "
            f"{signal.entry_risk_pct:>6.2f} | "
            f"{signal.intraday_momentum_pct:>5.2f} | "
            f"{signal.volume_ratio:>4.2f} | "
            f"{signal.ranking_score:>5.2f} | "
            f"{signal.reason}"
        )

    print_final_picks(signals)


if __name__ == "__main__":
    main()
