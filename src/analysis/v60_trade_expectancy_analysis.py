from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = "data/backtests/v58_wide_universe_costed.csv"
DEFAULT_OUTPUT_DIR = "data/analysis"


PRICE_BUCKETS = [0, 5, 10, 25, 50, 100, 250, 1000, np.inf]
LIQUIDITY_BUCKETS = [0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, 25_000_000, np.inf]


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def summarize_subset(name: str, subset: pd.DataFrame) -> dict:
    if subset.empty:
        return {"segment": name, "trades": 0}

    return {
        "segment": name,
        "trades": len(subset),
        "symbols": subset["symbol"].nunique() if "symbol" in subset.columns else 0,
        "net_profit_usd": subset["net_profit_usd"].sum(),
        "gross_profit_usd": subset["gross_profit_usd"].sum() if "gross_profit_usd" in subset.columns else np.nan,
        "execution_cost_usd": subset["execution_cost_usd"].sum() if "execution_cost_usd" in subset.columns else np.nan,
        "avg_net_trade_usd": subset["net_profit_usd"].mean(),
        "median_net_trade_usd": subset["net_profit_usd"].median(),
        "win_rate_pct": (subset["net_profit_usd"] > 0).mean() * 100.0,
    }


def bucket_analysis(df: pd.DataFrame, value_col: str, bucket_edges: list[float], bucket_name: str) -> pd.DataFrame:
    work = df.copy()
    work = work[work[value_col].notna()].copy()

    labels = []
    for i in range(len(bucket_edges) - 1):
        left = bucket_edges[i]
        right = bucket_edges[i + 1]
        if np.isinf(right):
            labels.append(f">={left:,.0f}")
        else:
            labels.append(f"{left:,.0f}-{right:,.0f}")

    work[bucket_name] = pd.cut(
        work[value_col],
        bins=bucket_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    rows = []
    for bucket, group in work.groupby(bucket_name, observed=False):
        if len(group) == 0:
            continue

        rows.append({
            bucket_name: str(bucket),
            "trades": len(group),
            "symbols": group["symbol"].nunique() if "symbol" in group.columns else 0,
            "net_profit_usd": group["net_profit_usd"].sum(),
            "avg_net_trade_usd": group["net_profit_usd"].mean(),
            "median_net_trade_usd": group["net_profit_usd"].median(),
            "win_rate_pct": (group["net_profit_usd"] > 0).mean() * 100.0,
            "avg_execution_cost_usd": group["execution_cost_usd"].mean() if "execution_cost_usd" in group.columns else np.nan,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("avg_net_trade_usd", ascending=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="v60 expectancy diagnostics")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-bottom-quantile", type=float, default=0.2)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)

    safe_numeric(df, [
        "net_profit_usd",
        "gross_profit_usd",
        "execution_cost_usd",
        "entry_price",
        "price",
        "last_price",
        "median_dollar_volume",
        "avg_dollar_volume",
        "relative_volume",
        "rvol",
    ])

    if "net_profit_usd" not in df.columns:
        raise ValueError("input csv must contain net_profit_usd")

    price_col = pick_column(df, ["entry_price", "price", "last_price"])
    liquidity_col = pick_column(df, ["median_dollar_volume", "avg_dollar_volume"])

    print("=== v60 expectancy diagnostics ===")
    print(f"Input: {path}")
    print(f"Trades: {len(df)}")

    # Winners vs losers
    sorted_df = df.sort_values("net_profit_usd")
    q = max(1, int(len(sorted_df) * args.top_bottom_quantile))

    losers = sorted_df.head(q).copy()
    winners = sorted_df.tail(q).copy()

    winner_summary = summarize_subset("top_winners", winners)
    loser_summary = summarize_subset("top_losers", losers)

    summary_df = pd.DataFrame([winner_summary, loser_summary])

    compare_cols = [
        c for c in [
            price_col,
            liquidity_col,
            "execution_cost_usd",
            "relative_volume",
            "rvol",
            "shares",
            "holding_minutes",
            "gross_profit_usd",
            "net_profit_usd",
        ]
        if c and c in df.columns
    ]

    compare_rows = []
    for label, subset in [("winners", winners), ("losers", losers)]:
        row = {"segment": label}
        for col in compare_cols:
            row[f"avg_{col}"] = subset[col].mean()
            row[f"median_{col}"] = subset[col].median()
        compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows)

    # Price bucket analysis
    if price_col:
        price_df = bucket_analysis(df, price_col, PRICE_BUCKETS, "price_bucket")
    else:
        price_df = pd.DataFrame()

    # Liquidity bucket analysis
    if liquidity_col:
        liquidity_df = bucket_analysis(df, liquidity_col, LIQUIDITY_BUCKETS, "liquidity_bucket")
    else:
        liquidity_df = pd.DataFrame()

    # Save outputs
    summary_df.to_csv(outdir / "v60_winner_loser_summary.csv", index=False)
    compare_df.to_csv(outdir / "v60_winner_loser_feature_compare.csv", index=False)
    price_df.to_csv(outdir / "v60_price_bucket_analysis.csv", index=False)
    liquidity_df.to_csv(outdir / "v60_liquidity_bucket_analysis.csv", index=False)

    print("\n=== Winners vs losers ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Feature comparison ===")
    print(compare_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not price_df.empty:
        print("\n=== Price bucket analysis ===")
        print(price_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not liquidity_df.empty:
        print("\n=== Liquidity bucket analysis ===")
        print(liquidity_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nSaved:")
    print(outdir / "v60_winner_loser_summary.csv")
    print(outdir / "v60_winner_loser_feature_compare.csv")
    print(outdir / "v60_price_bucket_analysis.csv")
    print(outdir / "v60_liquidity_bucket_analysis.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
