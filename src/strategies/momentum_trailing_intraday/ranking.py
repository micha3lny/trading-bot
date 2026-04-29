"""Very first simple ranking for Momentum Trailing Intraday strategy.

This is intentionally simple and will evolve later.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from src.data.fetch_top30 import UNIVERSE
from src.data.load_market_data import load_market_data_bundle


def compute_intraday_momentum(df_intraday: pd.DataFrame) -> float:
    """Simple momentum: % change from first bar to last bar."""
    if df_intraday.empty:
        return 0.0

    first_price = df_intraday.iloc[0]["close"]
    last_price = df_intraday.iloc[-1]["close"]

    if first_price == 0:
        return 0.0

    return (last_price - first_price) / first_price * 100.0


def compute_daily_trend(df_daily: pd.DataFrame) -> float:
    """Simple trend: last close vs 20-day moving average."""
    if len(df_daily) < 20:
        return 0.0

    df = df_daily.copy()
    df["ma20"] = df["close"].rolling(20).mean()

    last_row = df.iloc[-1]
    if last_row["ma20"] == 0:
        return 0.0

    return (last_row["close"] - last_row["ma20"]) / last_row["ma20"] * 100.0


def rank_symbols() -> List[Tuple[str, float]]:
    scores: list[tuple[str, float]] = []

    for spec in UNIVERSE:
        try:
            bundle = load_market_data_bundle(spec.symbol)

            intraday_momentum = compute_intraday_momentum(bundle.intraday)
            daily_trend = compute_daily_trend(bundle.daily)

            # Simple combined score (can be improved later)
            score = intraday_momentum * 0.7 + daily_trend * 0.3

            scores.append((spec.symbol, score))

        except Exception as exc:
            print(f"Skipping {spec.symbol}: {exc}")

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def main() -> None:
    ranking = rank_symbols()

    print("\nTop 10 (Momentum Trailing Intraday)\n")
    print("Rank | Symbol | Score")
    print("------------------------")

    for i, (symbol, score) in enumerate(ranking[:10], start=1):
        print(f"{i:>4} | {symbol:<6} | {score:>8.2f}")


if __name__ == "__main__":
    main()
