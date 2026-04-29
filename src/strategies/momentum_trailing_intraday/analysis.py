import pandas as pd
from dataclasses import asdict
from pathlib import Path


def export_trades(trades):
    path = Path("data/backtests")
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(t) for t in trades])
    df.to_csv(path / "momentum_trades.csv", index=False)
    return df


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
        grouped = df.groupby("bin")["pnl_pct"].agg(["count", "mean"])
        print(f"\n{col}:")
        print(grouped)
