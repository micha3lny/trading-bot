from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BACKTEST_DIR = Path("data/backtests")


def pct_fmt(x: float) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x:.2f}"


def load_trades(path: Path, variant: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "variant" in df.columns:
        df = df[df["variant"] == variant].copy()
    required = ["pnl_pct", "symbol", "session_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df.reset_index(drop=True)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "pnl_pct",
        "max_pnl_pct",
        "min_pnl_pct",
        "intraday_high_pct",
        "gap_pct",
        "first_5m_high_pct",
        "first_15m_high_pct",
        "time_to_high_minutes",
        "open_to_close_pct",
        "entry_minutes_from_open",
        "entry_price",
        "or_high_pct",
        "or_low_pct",
        "or_close_pct",
        "or_range_pct",
        "or_volume",
        "or_dollar_volume",
        "session_volume",
        "session_dollar_volume",
        "median_1m_volume",
        "momentum_score",
        "or_close_strength",
        "score_or_high_pct",
        "score_or_range_pct",
        "score_or_dollar_volume",
        "score_gap_pct",
        "score_open_price",
    ]
    return [c for c in preferred if c in df.columns]


def add_labels(df: pd.DataFrame, winner_threshold: float, loser_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["is_winner"] = out["pnl_pct"] >= winner_threshold
    out["is_loser"] = out["pnl_pct"] <= loser_threshold
    out["result_bucket"] = "neutral"
    out.loc[out["is_winner"], "result_bucket"] = "winner"
    out.loc[out["is_loser"], "result_bucket"] = "loser"
    return out


def summarize_groups(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for bucket, g in df.groupby("result_bucket"):
        row: dict[str, object] = {"bucket": bucket, "count": len(g), "avg_pnl": g["pnl_pct"].mean(), "median_pnl": g["pnl_pct"].median()}
        for col in cols:
            if col == "pnl_pct":
                continue
            row[f"{col}_mean"] = pd.to_numeric(g[col], errors="coerce").mean()
            row[f"{col}_median"] = pd.to_numeric(g[col], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bucket")


def feature_separation(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    winners = df[df["is_winner"]]
    losers = df[df["is_loser"]]
    rows = []
    for col in cols:
        if col == "pnl_pct":
            continue
        w = pd.to_numeric(winners[col], errors="coerce")
        l = pd.to_numeric(losers[col], errors="coerce")
        if w.dropna().empty or l.dropna().empty:
            continue
        rows.append(
            {
                "feature": col,
                "winner_mean": w.mean(),
                "loser_mean": l.mean(),
                "diff_mean": w.mean() - l.mean(),
                "winner_median": w.median(),
                "loser_median": l.median(),
                "diff_median": w.median() - l.median(),
                "winner_p25": w.quantile(0.25),
                "winner_p75": w.quantile(0.75),
                "loser_p25": l.quantile(0.25),
                "loser_p75": l.quantile(0.75),
            }
        )
    sep = pd.DataFrame(rows)
    if not sep.empty:
        sep["abs_diff_median"] = sep["diff_median"].abs()
        sep = sep.sort_values("abs_diff_median", ascending=False)
    return sep


def bucket_analysis(df: pd.DataFrame, feature: str, bins: list[float]) -> pd.DataFrame:
    if feature not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp[f"{feature}_bin"] = pd.cut(tmp[feature], bins=bins, include_lowest=True)
    out = tmp.groupby(f"{feature}_bin", observed=True).agg(
        count=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        median_pnl=("pnl_pct", "median"),
        win_rate=("is_winner", "mean"),
        loss_rate=("is_loser", "mean"),
        total_pnl=("pnl_pct", "sum"),
    )
    out["win_rate"] *= 100.0
    out["loss_rate"] *= 100.0
    return out.reset_index()


def symbol_repeatability(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    out = df.groupby("symbol").agg(
        count=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        median_pnl=("pnl_pct", "median"),
        total_pnl=("pnl_pct", "sum"),
        win_rate=("is_winner", "mean"),
        loss_rate=("is_loser", "mean"),
        avg_max_pnl=("max_pnl_pct", "mean") if "max_pnl_pct" in df.columns else ("pnl_pct", "mean"),
        avg_min_pnl=("min_pnl_pct", "mean") if "min_pnl_pct" in df.columns else ("pnl_pct", "mean"),
    ).reset_index()
    out["win_rate"] *= 100.0
    out["loss_rate"] *= 100.0
    return out[out["count"] >= min_trades].sort_values("total_pnl", ascending=False)


def print_table(title: str, df: pd.DataFrame, max_rows: int = 30) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("No rows")
        return
    print(df.head(max_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description="v43 diagnostics for momentum OR trades: winners vs losers.")
    parser.add_argument("--trades-csv", required=True, help="Path to v41/v42 trades CSV")
    parser.add_argument("--variant", default="close_exit", help="Variant to analyze, e.g. close_exit or trail_only")
    parser.add_argument("--winner-threshold", type=float, default=5.0)
    parser.add_argument("--loser-threshold", type=float, default=-5.0)
    parser.add_argument("--min-symbol-trades", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_BACKTEST_DIR))
    args = parser.parse_args()

    path = Path(args.trades_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_trades(path, args.variant)
    df = add_labels(df, args.winner_threshold, args.loser_threshold)
    cols = numeric_columns(df)

    print("v43 momentum trade diagnostics")
    print(f"Trades CSV: {path}")
    print(f"Variant: {args.variant}")
    print(f"Rows: {len(df)}")
    if df.empty:
        return 0

    print("\n=== Overall ===")
    print(f"count: {len(df)}")
    print(f"avg pnl: {df['pnl_pct'].mean():.2f}%")
    print(f"median pnl: {df['pnl_pct'].median():.2f}%")
    print(f"winner >= {args.winner_threshold:.2f}%: {df['is_winner'].sum()} ({df['is_winner'].mean()*100:.2f}%)")
    print(f"loser <= {args.loser_threshold:.2f}%: {df['is_loser'].sum()} ({df['is_loser'].mean()*100:.2f}%)")

    group_summary = summarize_groups(df, cols)
    sep = feature_separation(df, cols)
    sym = symbol_repeatability(df, args.min_symbol_trades)

    print_table("Winner / loser feature summary", group_summary, 20)
    print_table("Feature separation: winners minus losers", sep, 30)

    bucket_specs = {
        "momentum_score": [-999, 6, 7, 8, 9, 10, 999],
        "or_close_strength": [-999, 0.4, 0.55, 0.7, 0.85, 0.95, 1.01],
        "or_high_pct": [-999, 2, 4, 6, 8, 12, 18, 35, 999],
        "or_range_pct": [-999, 2, 4, 6, 10, 15, 25, 999],
        "or_dollar_volume": [0, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 20_000_000, float("inf")],
        "entry_minutes_from_open": [-1, 5, 10, 15, 30, 60, 120, 999],
        "gap_pct": [-999, -20, -10, -5, -2, 0, 2, 5, 10, 20, 999],
        "entry_price": [0, 2, 3, 5, 10, 20, 50, 100, 300, float("inf")],
    }

    bucket_outputs: list[pd.DataFrame] = []
    for feature, bins in bucket_specs.items():
        b = bucket_analysis(df, feature, bins)
        if not b.empty:
            b.insert(0, "feature", feature)
            bucket_outputs.append(b)
            print_table(f"Bucket analysis: {feature}", b, 50)

    print_table("Best repeatable symbols", sym.sort_values("total_pnl", ascending=False), 30)
    print_table("Worst repeatable symbols", sym.sort_values("total_pnl", ascending=True), 30)

    base = path.stem.replace("trades_", "")
    out_prefix = output_dir / f"v43_diagnostics_{base}_{args.variant}"
    group_summary.to_csv(f"{out_prefix}_group_summary.csv", index=False)
    sep.to_csv(f"{out_prefix}_feature_separation.csv", index=False)
    sym.to_csv(f"{out_prefix}_symbol_repeatability.csv", index=False)
    if bucket_outputs:
        pd.concat(bucket_outputs, ignore_index=True).to_csv(f"{out_prefix}_bucket_analysis.csv", index=False)

    print("\nSaved diagnostics:")
    print(f"- {out_prefix}_group_summary.csv")
    print(f"- {out_prefix}_feature_separation.csv")
    print(f"- {out_prefix}_symbol_repeatability.csv")
    print(f"- {out_prefix}_bucket_analysis.csv")

    print("\nInterpretation hints:")
    print("- We need features where winners and losers differ strongly by median, not only mean.")
    print("- If no single bucket has edge, v44 should combine filters or add premarket/catalyst data.")
    print("- If repeatability is symbol-specific, v44 should include a rolling symbol-quality score, not a static blacklist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
