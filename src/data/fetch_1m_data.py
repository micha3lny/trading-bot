"""
Simple 1m data fetcher (placeholder version).

NOTE:
- This version generates synthetic/random data if no provider is connected.
- Replace `fetch_from_provider` with real API (IBKR, Polygon, Alpaca, etc.) later.

Usage:
python -m src.data.fetch_1m_data --symbols AAPL TSLA --days 5
python -m src.data.fetch_1m_data --symbols-file symbols.txt --days 90
"""

import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

OUTPUT_DIR = "data/1m"


def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_fake_1m(symbol: str, days: int) -> pd.DataFrame:
    rows = []
    now = datetime.utcnow()
    price = 100.0 + np.random.rand() * 20

    for d in range(days):
        day = now - timedelta(days=d)
        for m in range(390):  # approx US session minutes
            ts = datetime(day.year, day.month, day.day, 9, 30) + timedelta(minutes=m)
            change = np.random.randn() * 0.1
            open_p = price
            close_p = price * (1 + change / 100)
            high = max(open_p, close_p) * (1 + abs(np.random.randn()) * 0.001)
            low = min(open_p, close_p) * (1 - abs(np.random.randn()) * 0.001)
            volume = int(abs(np.random.randn()) * 1000)

            rows.append([
                symbol,
                ts.isoformat(),
                open_p,
                high,
                low,
                close_p,
                volume,
            ])

            price = close_p

    df = pd.DataFrame(rows, columns=["symbol", "datetime", "open", "high", "low", "close", "volume"])
    return df


def save(symbol: str, df: pd.DataFrame):
    path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {symbol} -> {path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", help="List of symbols")
    parser.add_argument("--symbols-file", help="File with symbols (one per line)")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    symbols = []

    if args.symbols:
        symbols.extend(args.symbols)

    if args.symbols_file:
        with open(args.symbols_file) as f:
            symbols.extend([line.strip() for line in f if line.strip()])

    if not symbols:
        print("No symbols provided")
        return

    ensure_dir()

    for sym in symbols:
        df = generate_fake_1m(sym, args.days)
        save(sym, df)


if __name__ == "__main__":
    main()
