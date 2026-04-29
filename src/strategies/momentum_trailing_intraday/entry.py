"""Entry logic for Momentum Trailing Intraday strategy.

This module does not place orders.
It checks whether ranked symbols have a confirmed breakout entry signal.
Pullback/retest logic stays available as an experiment, but it is disabled by default
because the first backtest showed weaker performance than pure breakout entries.
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

# Entry setup settings from the first positive optimization result.
OPENING_RANGE_BARS = 4  # 4 x 15m = first hour
MIN_INTRADAY_MOMENTUM_PCT = 0.5
MIN_VOLUME_RATIO = 0.8
MIN_BREAKOUT_PCT = 0.25
MAX_ENTRY_RISK_PCT = 2.0
MIN_CLOSE_STRENGTH = 0.60

# Pullback/retest settings.
# Disabled by default after initial test: it reduced Avg PnL from optimized breakout-only logic.
ENABLE_PULLBACK_RETEST_ENTRY = False
PULLBACK_RETEST_TOLERANCE_PCT = 0.35


@dataclass(frozen=True)
class EntrySignal:
    symbol: str
    has_signal: bool
    is_final_pick: bool
    reason: str
    setup_type: str
    last_price: float
    opening_range_high: float
    opening_range_low: float
    breakout_pct: float
    close_strength: float
    entry_risk_pct: float
    intraday_momentum_pct: float
    volume_ratio: float
    ranking_score: float


def calculate_breakout_pct(last_price: float, opening_range_high: float) -> float:
    if opening_range_high == 0:
        return 0.0
    return (last_price - opening_range_high) / opening_range_high * 100.0


def calculate_close_strength_from_row(row: pd.Series) -> float:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    if high == low:
        return 0.0

    return (close - low) / (high - low)


def calculate_close_strength(latest_session: pd.DataFrame) -> float:
    if latest_session.empty:
        return 0.0
    return calculate_close_strength_from_row(latest_session.iloc[-1])


def calculate_entry_risk_pct(last_price: float, opening_range_low: float) -> float:
    if last_price == 0:
        return 0.0
    return (last_price - opening_range_low) / last_price * 100.0


def get_opening_range_values(latest_session: pd.DataFrame) -> tuple[float, float, float]:
    opening_range = latest_session.iloc[:OPENING_RANGE_BARS]
    opening_range_high = float(opening_range["high"].max())
    opening_range_low = float(opening_range["low"].min())
    last_price = float(latest_session.iloc[-1]["close"])

    return last_price, opening_range_high, opening_range_low


def build_signal(
    row: RankingRow,
    has_signal: bool,
    reason: str,
    setup_type: str,
    last_price: float,
    opening_range_high: float,
    opening_range_low: float,
    close_strength: float,
) -> EntrySignal:
    breakout_pct = calculate_breakout_pct(last_price, opening_range_high)
    entry_risk_pct = calculate_entry_risk_pct(last_price, opening_range_low)

    return EntrySignal(
        symbol=row.symbol,
        has_signal=has_signal,
        is_final_pick=False,
        reason=reason,
        setup_type=setup_type,
        last_price=last_price,
        opening_range_high=opening_range_high,
        opening_range_low=opening_range_low,
        breakout_pct=breakout_pct,
        close_strength=close_strength,
        entry_risk_pct=entry_risk_pct,
        intraday_momentum_pct=row.intraday_momentum_pct,
        volume_ratio=row.volume_ratio,
        ranking_score=row.score,
    )


def has_pullback_retest(latest_session: pd.DataFrame, opening_range_high: float) -> bool:
    if not ENABLE_PULLBACK_RETEST_ENTRY or len(latest_session) <= OPENING_RANGE_BARS + 1:
        return False

    after_opening_range = latest_session.iloc[OPENING_RANGE_BARS:]
    retest_low_threshold = opening_range_high * (1.0 - PULLBACK_RETEST_TOLERANCE_PCT / 100.0)

    broke_out = False
    retested = False

    for _, bar in after_opening_range.iterrows():
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        if not broke_out and bar_close > opening_range_high:
            broke_out = True
            continue

        if broke_out and bar_low <= opening_range_high and bar_low >= retest_low_threshold:
            retested = True

        if retested and bar_close > opening_range_high:
            return True

    return False


def check_opening_range_breakout(row: RankingRow, df_intraday: pd.DataFrame) -> EntrySignal:
    latest_session = get_last_intraday_session(df_intraday)

    if len(latest_session) <= OPENING_RANGE_BARS:
        return build_signal(row, False, "not enough intraday bars after opening range", "none", 0.0, 0.0, 0.0, 0.0)

    last_price, opening_range_high, opening_range_low = get_opening_range_values(latest_session)
    breakout_pct = calculate_breakout_pct(last_price, opening_range_high)
    close_strength = calculate_close_strength(latest_session)
    entry_risk_pct = calculate_entry_risk_pct(last_price, opening_range_low)
    has_retest = has_pullback_retest(latest_session, opening_range_high)

    if row.intraday_momentum_pct < MIN_INTRADAY_MOMENTUM_PCT:
        return build_signal(row, False, f"intraday momentum below {MIN_INTRADAY_MOMENTUM_PCT:.2f}%", "none", last_price, opening_range_high, opening_range_low, close_strength)

    if row.volume_ratio < MIN_VOLUME_RATIO:
        return build_signal(row, False, f"volume ratio below {MIN_VOLUME_RATIO:.2f}", "none", last_price, opening_range_high, opening_range_low, close_strength)

    if breakout_pct < MIN_BREAKOUT_PCT:
        return build_signal(row, False, f"breakout below {MIN_BREAKOUT_PCT:.2f}%", "none", last_price, opening_range_high, opening_range_low, close_strength)

    if entry_risk_pct > MAX_ENTRY_RISK_PCT:
        return build_signal(row, False, f"entry risk above {MAX_ENTRY_RISK_PCT:.2f}%", "none", last_price, opening_range_high, opening_range_low, close_strength)

    if close_strength >= MIN_CLOSE_STRENGTH:
        return build_signal(row, True, "confirmed breakout above opening range", "breakout", last_price, opening_range_high, opening_range_low, close_strength)

    if has_retest:
        return build_signal(row, True, "pullback/retest after breakout", "pullback_retest", last_price, opening_range_high, opening_range_low, close_strength)

    return build_signal(row, False, f"weak candle close below {MIN_CLOSE_STRENGTH:.2f}", "none", last_price, opening_range_high, opening_range_low, close_strength)


def mark_final_picks(signals: list[EntrySignal], max_positions: int = MAX_POSITIONS) -> list[EntrySignal]:
    selected_symbols = {
        signal.symbol
        for signal in sorted(
            [signal for signal in signals if signal.has_signal],
            key=lambda signal: (signal.ranking_score, signal.breakout_pct, -signal.entry_risk_pct),
            reverse=True,
        )[:max_positions]
    }

    return [
        EntrySignal(
            symbol=signal.symbol,
            has_signal=signal.has_signal,
            is_final_pick=signal.symbol in selected_symbols,
            reason=signal.reason,
            setup_type=signal.setup_type,
            last_price=signal.last_price,
            opening_range_high=signal.opening_range_high,
            opening_range_low=signal.opening_range_low,
            breakout_pct=signal.breakout_pct,
            close_strength=signal.close_strength,
            entry_risk_pct=signal.entry_risk_pct,
            intraday_momentum_pct=signal.intraday_momentum_pct,
            volume_ratio=signal.volume_ratio,
            ranking_score=signal.ranking_score,
        )
        for signal in signals
    ]


def find_entry_signals(ranking_candidates_limit: int = RANKING_CANDIDATES_LIMIT, max_positions: int = MAX_POSITIONS) -> list[EntrySignal]:
    signals: list[EntrySignal] = []
    ranking = rank_symbols()[:ranking_candidates_limit]

    for row in ranking:
        bundle = load_market_data_bundle(row.symbol)
        signals.append(check_opening_range_breakout(row, bundle.intraday))

    return mark_final_picks(signals, max_positions=max_positions)


def print_final_picks(signals: list[EntrySignal]) -> None:
    final_picks = [signal for signal in signals if signal.is_final_pick]

    print(f"\nFinal picks: top {MAX_POSITIONS} valid YES signals\n")

    if not final_picks:
        print("No final picks. No valid YES signals passed the filters.")
        return

    print("Pick | Symbol | Setup | Score | Last | Breakout % | CloseStr | Risk % | Mom % | Vol")
    print("--------------------------------------------------------------------------------")
    for i, signal in enumerate(final_picks, start=1):
        print(f"{i:>4} | {signal.symbol:<6} | {signal.setup_type:<14} | {signal.ranking_score:>5.2f} | {signal.last_price:>8.2f} | {signal.breakout_pct:>10.2f} | {signal.close_strength:>8.2f} | {signal.entry_risk_pct:>6.2f} | {signal.intraday_momentum_pct:>5.2f} | {signal.volume_ratio:>4.2f}")


def main() -> None:
    signals = find_entry_signals()

    print("\nEntry signals: Momentum Trailing Intraday\n")
    print("Rank | Symbol | Signal | Pick | Setup | Last | OR High | OR Low | Breakout % | CloseStr | Risk % | Mom % | Vol | Score | Reason")
    print("----------------------------------------------------------------------------------------------------------------")

    for i, signal in enumerate(signals, start=1):
        signal_text = "YES" if signal.has_signal else "NO"
        pick_text = "YES" if signal.is_final_pick else ""
        print(f"{i:>4} | {signal.symbol:<6} | {signal_text:<6} | {pick_text:<4} | {signal.setup_type:<14} | {signal.last_price:>8.2f} | {signal.opening_range_high:>7.2f} | {signal.opening_range_low:>6.2f} | {signal.breakout_pct:>10.2f} | {signal.close_strength:>8.2f} | {signal.entry_risk_pct:>6.2f} | {signal.intraday_momentum_pct:>5.2f} | {signal.volume_ratio:>4.2f} | {signal.ranking_score:>5.2f} | {signal.reason}")

    print_final_picks(signals)


if __name__ == "__main__":
    main()
