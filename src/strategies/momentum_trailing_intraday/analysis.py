from dataclasses import asdict
from pathlib import Path

import pandas as pd


def export_trades(trades):
    path = Path("data/backtests")
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(t) for t in trades])
    df.to_csv(path / "momentum_trades.csv", index=False)
    return df


def _print_grouped(df: pd.DataFrame, title: str, group_col: str) -> None:
    grouped = df.groupby(group_col, observed=False)["pnl_pct"].agg(
        count="count",
        mean="mean",
        median="median",
        total="sum",
        min="min",
        max="max",
    )
    grouped["win_rate"] = df.groupby(group_col, observed=False)["pnl_pct"].apply(lambda values: (values > 0).mean() * 100.0)
    print(f"\n--- {title} ---")
    print(grouped)


def _add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["breakout_bin"] = pd.cut(
        df["breakout_pct"],
        bins=[-999, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 999],
    )
    df["close_strength_bin"] = pd.cut(
        df["close_strength"],
        bins=[-999, 0.70, 0.80, 0.85, 0.90, 0.95, 999],
    )
    df["daily_trend_bin"] = pd.cut(
        df["daily_trend_pct"],
        bins=[-999, -30, -20, -10, -5, -3, 0, 999],
    )
    df["entry_risk_bin"] = pd.cut(
        df["entry_risk_pct"],
        bins=[-999, 4, 5, 6, 7, 8, 10, 999],
    )
    df["pnl_positive"] = df["pnl_pct"] > 0
    return df


def segment_analysis(df: pd.DataFrame) -> None:
    if df.empty:
        return

    df = _add_bins(df)

    print("\n=== SEGMENTATION ANALYSIS ===")
    print("Goal: identify which pre-entry feature ranges carry edge and which ranges add noise.")

    for title, group_col in [
        ("breakout_pct", "breakout_bin"),
        ("1m close_strength", "close_strength_bin"),
        ("daily_trend_pct", "daily_trend_bin"),
        ("entry_risk_pct", "entry_risk_bin"),
        ("exit_reason", "exit_reason"),
        ("symbol", "symbol"),
    ]:
        _print_grouped(df, title, group_col)

    print("\n--- breakout x close_strength mean pnl ---")
    print(
        df.pivot_table(
            index="breakout_bin",
            columns="close_strength_bin",
            values="pnl_pct",
            aggfunc="mean",
            observed=False,
        )
    )

    print("\n--- breakout x close_strength count ---")
    print(
        df.pivot_table(
            index="breakout_bin",
            columns="close_strength_bin",
            values="pnl_pct",
            aggfunc="count",
            observed=False,
        )
    )

    print("\n--- daily_trend x close_strength mean pnl ---")
    print(
        df.pivot_table(
            index="daily_trend_bin",
            columns="close_strength_bin",
            values="pnl_pct",
            aggfunc="mean",
            observed=False,
        )
    )

    print("\n--- worst symbols by average pnl, min 2 trades ---")
    by_symbol = df.groupby("symbol")["pnl_pct"].agg(count="count", mean="mean", total="sum")
    print(by_symbol[by_symbol["count"] >= 2].sort_values("mean").head(20))

    print("\n--- best symbols by average pnl, min 2 trades ---")
    print(by_symbol[by_symbol["count"] >= 2].sort_values("mean", ascending=False).head(20))


def analyze(df):
    if df.empty:
        print("No trades to analyze")
        return

    print("\n=== ANALYSIS ===")

    for col, bins in {
        "breakout_pct": [0.25, 0.5, 1.0],
        "daily_trend_pct": [0, 2, 5, 10],
        "entry_risk_pct": [1.0, 1.5, 2.0],
        "close_strength": [0.6, 0.8, 0.9],
    }.items():
        df["bin"] = pd.cut(df[col], bins=[-999] + bins + [999])
        grouped = df.groupby("bin", observed=False)["pnl_pct"].agg(["count", "mean"])
        print(f"\n{col}:")
        print(grouped)

    segment_analysis(df)
