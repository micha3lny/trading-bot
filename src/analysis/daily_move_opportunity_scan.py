"""Daily move opportunity scan.

Goal:
- Check if the dataset actually contains strong daily moves (e.g. >5%).
- Identify whether the strategy is missing large opportunities.

Run:
python -m src.analysis.daily_move_opportunity_scan
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.strategies.momentum_trailing_intraday.reversal_pullback_entry_scan_v29_simple import (
    load_all_data,
)

THRESHOLDS = [3.0, 5.0, 7.0, 10.0]


def main() -> None:
    print("\nDaily move opportunity scan")

    _data_15m, _data_5m, _data_1m, daily_data = load_all_data()

    rows = []
    for symbol, df in daily_data.items():
        df = df.sort_values("date").copy()

        # Try common column names
        open_col = "open"
        close_col = "close"
        high_col = "high"

        if open_col not in df or close_col not in df or high_col not in df:
            continue

        df["daily_return_pct"] = (df[close_col] - df[open_col]) / df[open_col] * 100.0
        df["intraday_high_pct"] = (df[high_col] - df[open_col]) / df[open_col] * 100.0

        for _, row in df.iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "date": row["date"],
                    "daily_return_pct": row["daily_return_pct"],
                    "intraday_high_pct": row["intraday_high_pct"],
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        print("No data.")
        return

    print("\n=== Overall stats ===")
    print(f"Total days: {len(df)}")
    print(f"Avg daily return: {df['daily_return_pct'].mean():.2f}%")
    print(f"Max daily return: {df['daily_return_pct'].max():.2f}%")
    print(f"Max intraday high: {df['intraday_high_pct'].max():.2f}%")

    print("\n=== Threshold analysis ===")
    for threshold in THRESHOLDS:
        close_count = (df["daily_return_pct"] >= threshold).sum()
        high_count = (df["intraday_high_pct"] >= threshold).sum()
        print(f">= {threshold:.1f}% close-to-close: {close_count}")
        print(f">= {threshold:.1f}% intraday high: {high_count}")

    print("\n=== Top daily moves (close-to-close) ===")
    print(
        df.sort_values("daily_return_pct", ascending=False)
        .head(20)[["date", "symbol", "daily_return_pct"]]
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\n=== Top intraday spikes (high vs open) ===")
    print(
        df.sort_values("intraday_high_pct", ascending=False)
        .head(20)[["date", "symbol", "intraday_high_pct"]]
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\n=== Symbols with most >=5% intraday moves ===")
    top_symbols = (
        df[df["intraday_high_pct"] >= 5.0]
        .groupby("symbol")
        .size()
        .sort_values(ascending=False)
        .head(20)
    )
    for symbol, count in top_symbols.items():
        print(f"{symbol:<6} {count}")


if __name__ == "__main__":
    main()
