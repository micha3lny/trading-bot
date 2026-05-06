from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR


def load_trades(path: str, variant: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "variant" not in df.columns:
        raise ValueError("trades CSV must contain a 'variant' column")
    out = df[df["variant"] == variant].copy()
    if out.empty:
        raise ValueError(f"No trades found for variant={variant!r}")
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce").dt.date.astype(str)
    out["entry_time_dt"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out = out.sort_values(["session_date", "entry_time_dt", "symbol"]).reset_index(drop=True)
    return out


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create live-like proxy regime features using only trade-time fields.

    This is NOT true QQQ/SPY regime yet. It is a first proxy layer using:
    - number of accepted setups per day
    - average setup score per day
    - median OR strength per day
    - median early gap of accepted candidates

    These proxies are available intraday only after setups appear. They are not final-day outcomes.
    Real v49 should replace/augment this with QQQ/SPY/VIX/premarket breadth data.
    """
    df = df.copy()
    day = df.groupby("session_date")
    regime = day.agg(
        day_trade_count=("symbol", "count"),
        day_avg_v45_score=("v45_score", "mean"),
        day_median_v45_score=("v45_score", "median"),
        day_avg_or_high_pct=("or_high_pct", "mean"),
        day_median_or_high_pct=("or_high_pct", "median"),
        day_avg_or_close_strength=("or_close_strength", "mean"),
        day_avg_gap_pct=("gap_pct", "mean"),
        day_median_gap_pct=("gap_pct", "median"),
    ).reset_index()

    df = df.merge(regime, on="session_date", how="left")
    df["regime_score"] = 0.0
    df.loc[df["day_trade_count"] >= 5, "regime_score"] += 1.0
    df.loc[df["day_trade_count"] >= 10, "regime_score"] += 1.0
    df.loc[df["day_avg_v45_score"] >= 9.0, "regime_score"] += 1.0
    df.loc[df["day_avg_v45_score"] >= 10.0, "regime_score"] += 1.0
    df.loc[df["day_avg_or_close_strength"] >= 0.70, "regime_score"] += 1.0
    df.loc[df["day_median_or_high_pct"].between(2.0, 8.0, inclusive="both"), "regime_score"] += 0.5
    df.loc[df["day_median_gap_pct"].between(-5.0, 5.0, inclusive="both"), "regime_score"] += 0.5
    return df


def setup_quality_multiplier(row: pd.Series, args: argparse.Namespace) -> float:
    """Dynamic sizing based only on setup fields available at entry."""
    score = float(row.get("v45_score", 0.0) or 0.0)
    or_strength = float(row.get("or_close_strength", 0.0) or 0.0)
    entry_minutes = float(row.get("entry_minutes_from_open", 999.0) or 999.0)
    pre_break = float(row.get("pre_entry_break_below_or_pct", -99.0) or -99.0)
    regime_score = float(row.get("regime_score", 0.0) or 0.0)

    mult = 1.0

    # Setup quality from v45 score.
    if score >= 12:
        mult += 0.75
    elif score >= 10:
        mult += 0.50
    elif score >= 9:
        mult += 0.25
    elif score < 8:
        mult -= 0.35

    # Cleaner OR close / no rejection.
    if or_strength >= 0.85:
        mult += 0.25
    elif or_strength < 0.55:
        mult -= 0.25

    # Timing preference from diagnostics.
    if 5 <= entry_minutes <= 10:
        mult += 0.20
    elif 10 < entry_minutes < 15:
        mult -= 0.35
    elif entry_minutes > 30:
        mult -= 0.30

    # Did not deeply lose OR before entry.
    if pre_break >= -0.5:
        mult += 0.20
    elif pre_break < -1.25:
        mult -= 0.25

    # Day-level momentum proxy.
    if regime_score >= 4:
        mult += 0.50
    elif regime_score >= 3:
        mult += 0.25
    elif regime_score < 2:
        mult -= 0.25

    return max(args.min_position_mult, min(args.max_position_mult, mult))


def apply_filters_and_sizing(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_regime_features(df)
    rejected_reasons = []

    for _, row in df.iterrows():
        reasons: list[str] = []
        if args.min_regime_score is not None and float(row["regime_score"]) < args.min_regime_score:
            reasons.append("regime_score_too_low")
        if args.min_day_trade_count is not None and int(row["day_trade_count"]) < args.min_day_trade_count:
            reasons.append("day_trade_count_too_low")
        if args.min_day_avg_v45_score is not None and float(row["day_avg_v45_score"]) < args.min_day_avg_v45_score:
            reasons.append("day_avg_v45_score_too_low")
        if args.avoid_entry_10_15 and 10 < float(row.get("entry_minutes_from_open", 999)) < 15:
            reasons.append("bad_entry_timing_10_15")
        rejected_reasons.append(";".join(reasons))

    df["v48_reject_reason"] = rejected_reasons
    accepted = df[df["v48_reject_reason"] == ""].copy()
    rejected = df[df["v48_reject_reason"] != ""].copy()

    if not accepted.empty:
        accepted["position_mult"] = accepted.apply(lambda r: setup_quality_multiplier(r, args), axis=1)
        accepted["position_usd"] = args.base_position_size * accepted["position_mult"]
        accepted["profit_usd"] = accepted["position_usd"] * accepted["pnl_pct"] / 100.0
        accepted["baseline_profit_usd"] = args.base_position_size * accepted["pnl_pct"] / 100.0
    return accepted, rejected


def summarize_portfolio(label: str, df: pd.DataFrame, base_position: float) -> dict[str, float | str | int]:
    if df.empty:
        return {
            "strategy": label,
            "count": 0,
            "active_days": 0,
            "symbols": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "total_pnl": 0.0,
            "fixed_1000_profit_usd": 0.0,
            "dynamic_profit_usd": 0.0,
            "avg_position_usd": 0.0,
            "max_position_usd": 0.0,
            "avg_profit_per_trade_usd": 0.0,
            "max_drawdown_usd": 0.0,
        }

    pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
    profit = pd.to_numeric(df["profit_usd"], errors="coerce").fillna(0.0)
    equity = profit.cumsum()
    dd = equity - equity.cummax()
    return {
        "strategy": label,
        "count": int(len(df)),
        "active_days": int(df["session_date"].nunique()),
        "symbols": int(df["symbol"].nunique()),
        "win_rate": float((pnl > 0).mean() * 100.0),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "total_pnl": float(pnl.sum()),
        "fixed_1000_profit_usd": float(pnl.sum() / 100.0 * base_position),
        "dynamic_profit_usd": float(profit.sum()),
        "avg_position_usd": float(df["position_usd"].mean()),
        "max_position_usd": float(df["position_usd"].max()),
        "avg_profit_per_trade_usd": float(profit.mean()),
        "max_drawdown_usd": float(dd.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v48 regime proxy + dynamic position sizing on v46 wide_trail trades.")
    parser.add_argument("--trades-csv", default="data/backtests/v46_trend_exit_momentum_or_breakout_trades_recent90_or5_score7_trendexit.csv")
    parser.add_argument("--source-variant", default="wide_trail")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--opening-range-minutes", type=int, default=5)

    parser.add_argument("--base-position-size", type=float, default=1000.0)
    parser.add_argument("--min-position-mult", type=float, default=0.50)
    parser.add_argument("--max-position-mult", type=float, default=2.00)

    parser.add_argument("--min-regime-score", type=float, default=None)
    parser.add_argument("--min-day-trade-count", type=int, default=None)
    parser.add_argument("--min-day-avg-v45-score", type=float, default=None)
    parser.add_argument("--avoid-entry-10-15", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    print("Experiment: v48 regime proxy + dynamic sizing")
    print("Base: v46 wide_trail trade CSV")
    print("This is a portfolio/risk-layer simulation, not a rescan of 1m files.")
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Source variant: {args.source_variant}")
    print(f"Base position size: ${args.base_position_size:.2f}")
    print(f"Position multiplier range: {args.min_position_mult:.2f}x - {args.max_position_mult:.2f}x")
    print(f"Avoid entry 10-15 min: {args.avoid_entry_10_15}")

    trades = load_trades(args.trades_csv, args.source_variant)
    baseline = trades.copy()
    baseline["position_mult"] = 1.0
    baseline["position_usd"] = args.base_position_size
    baseline["profit_usd"] = baseline["position_usd"] * baseline["pnl_pct"] / 100.0
    baseline["baseline_profit_usd"] = baseline["profit_usd"]

    accepted, rejected = apply_filters_and_sizing(trades, args)

    summary = pd.DataFrame([
        summarize_portfolio(f"baseline_{args.source_variant}_fixed", baseline, args.base_position_size),
        summarize_portfolio("v48_regime_dynamic_sizing", accepted, args.base_position_size),
    ])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"recent{args.recent_days}_or{args.opening_range_minutes}_{args.source_variant}_dynsize"
    out_trades = output_dir / f"v48_regime_sizing_trades_{suffix}.csv"
    out_rejected = output_dir / f"v48_regime_sizing_rejected_{suffix}.csv"
    out_summary = output_dir / f"v48_regime_sizing_summary_{suffix}.csv"
    accepted.to_csv(out_trades, index=False)
    rejected.to_csv(out_rejected, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\nSaved accepted trades CSV: {out_trades}")
    print(f"Saved rejected trades CSV: {out_rejected}")
    print(f"Saved summary CSV: {out_summary}")

    print("\n=== Portfolio comparison ===")
    cols = [
        "strategy", "count", "active_days", "symbols", "win_rate", "avg_pnl", "median_pnl", "total_pnl",
        "fixed_1000_profit_usd", "dynamic_profit_usd", "avg_position_usd", "max_position_usd", "avg_profit_per_trade_usd", "max_drawdown_usd",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== v48 scan stats ===")
    print(f"baseline trades: {len(baseline)}")
    print(f"accepted trades: {len(accepted)}")
    print(f"rejected trades: {len(rejected)}")
    if not rejected.empty:
        print("\n=== Rejection reasons ===")
        print(rejected["v48_reject_reason"].value_counts().to_string())

    if not accepted.empty:
        print("\n=== Position multiplier buckets ===")
        tmp = accepted.copy()
        tmp["position_mult_bin"] = pd.cut(tmp["position_mult"], bins=[0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 99])
        print(tmp.groupby("position_mult_bin", observed=True)["profit_usd"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Regime score buckets ===")
        tmp["regime_score_bin"] = pd.cut(tmp["regime_score"], bins=[-999, 1, 2, 3, 4, 5, 999])
        print(tmp.groupby("regime_score_bin", observed=True)["profit_usd"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Top days by dynamic profit ===")
        print(tmp.groupby("session_date")["profit_usd"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False).head(20).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Worst days by dynamic profit ===")
        print(tmp.groupby("session_date")["profit_usd"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=True).head(20).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- Good outcome: dynamic_profit_usd > fixed baseline without much worse drawdown.")
    print("- This uses a proxy regime from accepted setups, not QQQ/SPY/VIX yet.")
    print("- If this helps, v49 should add real market regime data: QQQ/SPY/VIX/breadth/premarket.")
    print("- If this hurts, keep fixed sizing until we have stronger regime features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
