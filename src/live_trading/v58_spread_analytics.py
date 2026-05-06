from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "data/live/market_data_snapshots.csv"
DEFAULT_OUTPUT = "data/live/spread_analytics_summary.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="v58 spread analytics")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print("=== v58 spread analytics ===")
    print(f"Input: {args.input}")

    path = Path(args.input)
    if not path.exists():
        print("Snapshot CSV not found")
        return 1

    df = pd.read_csv(path)

    if df.empty:
        print("Snapshot CSV is empty")
        return 1

    df["spread_bps"] = pd.to_numeric(df["spread_bps"], errors="coerce")
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    clean = df.dropna(subset=["spread_bps"]).copy()

    if clean.empty:
        print("No valid spread rows")
        return 1

    summary = (
        clean.groupby("symbol")
        .agg(
            samples=("spread_bps", "count"),
            mean_spread_bps=("spread_bps", "mean"),
            median_spread_bps=("spread_bps", "median"),
            p90_spread_bps=("spread_bps", lambda s: s.quantile(0.90)),
            max_spread_bps=("spread_bps", "max"),
            mean_abs_spread=("spread", "mean"),
            median_volume=("volume", "median"),
        )
        .sort_values("mean_spread_bps")
        .reset_index()
    )

    summary["execution_bucket"] = pd.cut(
        summary["mean_spread_bps"],
        bins=[-1, 1, 3, 8, 20, 1000],
        labels=[
            "elite_liquidity",
            "very_liquid",
            "liquid",
            "medium_liquidity",
            "wide_spread",
        ],
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    print("\n=== Spread summary ===")
    print(
        summary[
            [
                "symbol",
                "samples",
                "mean_spread_bps",
                "median_spread_bps",
                "p90_spread_bps",
                "max_spread_bps",
                "execution_bucket",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\n=== Liquidity interpretation ===")
    for _, row in summary.iterrows():
        print(
            f"{row['symbol']}: {row['execution_bucket']} | "
            f"mean={row['mean_spread_bps']:.2f}bps "
            f"p90={row['p90_spread_bps']:.2f}bps"
        )

    print(f"\nSaved summary: {out}")

    print("\nInterpretation hints:")
    print("- Symbols with consistently low spreads are better for scaling and intraday execution.")
    print("- Wide spread names likely require smaller sizing or stronger expected momentum.")
    print("- These statistics can calibrate future backtest execution-cost assumptions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
