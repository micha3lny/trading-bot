from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v49_market_regime import (
    add_market_features,
    load_market_data,
    load_trades,
)


def market_regime_label(row: pd.Series) -> str:
    ret = float(row.get("market_return_from_open_pct", 0.0) or 0.0)
    above_vwap = bool(row.get("market_above_vwap", False))
    first15_low = float(row.get("market_first15_low_pct", 0.0) or 0.0)

    if pd.isna(row.get("market_return_from_open_pct")):
        return "unknown"
    if ret >= 0.50 and above_vwap and first15_low > -0.75:
        return "strong"
    if ret >= 0.20 and above_vwap:
        return "good"
    if ret >= 0.00 and above_vwap:
        return "neutral"
    if ret >= -0.20:
        return "weak"
    return "bad"


def setup_quality_label(row: pd.Series) -> str:
    score = float(row.get("v45_score", 0.0) or 0.0)
    or_strength = float(row.get("or_close_strength", 0.0) or 0.0)
    entry_minutes = float(row.get("entry_minutes_from_open", 999.0) or 999.0)
    pre_break = float(row.get("pre_entry_break_below_or_pct", -99.0) or -99.0)

    q = 0.0
    if score >= 12:
        q += 3.0
    elif score >= 10:
        q += 2.0
    elif score >= 9:
        q += 1.0
    elif score < 8:
        q -= 1.0

    if or_strength >= 0.85:
        q += 1.0
    elif or_strength < 0.55:
        q -= 1.0

    if 5 <= entry_minutes <= 10:
        q += 0.5
    elif 10 < entry_minutes < 15:
        q -= 1.0
    elif entry_minutes > 30:
        q -= 0.5

    if pre_break >= -0.50:
        q += 0.5
    elif pre_break < -1.25:
        q -= 0.5

    if q >= 4:
        return "A+"
    if q >= 2.5:
        return "A"
    if q >= 1.0:
        return "B"
    return "C"


def exposure_multiplier(regime: str, quality: str, args: argparse.Namespace) -> float:
    table = {
        ("strong", "A+"): args.size_strong_aplus,
        ("strong", "A"): args.size_strong_a,
        ("strong", "B"): args.size_strong_b,
        ("strong", "C"): args.size_strong_c,
        ("good", "A+"): args.size_good_aplus,
        ("good", "A"): args.size_good_a,
        ("good", "B"): args.size_good_b,
        ("good", "C"): args.size_good_c,
        ("neutral", "A+"): args.size_neutral_aplus,
        ("neutral", "A"): args.size_neutral_a,
        ("neutral", "B"): args.size_neutral_b,
        ("neutral", "C"): args.size_neutral_c,
        ("weak", "A+"): args.size_weak_aplus,
        ("weak", "A"): args.size_weak_a,
        ("weak", "B"): args.size_weak_b,
        ("weak", "C"): args.size_weak_c,
        ("bad", "A+"): args.size_bad_aplus,
        ("bad", "A"): args.size_bad_a,
        ("bad", "B"): args.size_bad_b,
        ("bad", "C"): args.size_bad_c,
    }
    return float(table.get((regime, quality), args.size_unknown))


def apply_adaptive_exposure(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = trades.copy()
    out["market_regime"] = out.apply(market_regime_label, axis=1)
    out["setup_quality"] = out.apply(setup_quality_label, axis=1)
    out["position_mult"] = out.apply(lambda r: exposure_multiplier(str(r["market_regime"]), str(r["setup_quality"]), args), axis=1)

    if args.skip_zero_size:
        out = out[out["position_mult"] > 0].copy()

    out["position_usd"] = args.base_position_size * out["position_mult"]
    out["profit_usd"] = out["position_usd"] * out["pnl_pct"] / 100.0
    out["baseline_profit_usd"] = args.base_position_size * out["pnl_pct"] / 100.0
    return out


def summarize_portfolio(label: str, df: pd.DataFrame, base_position: float) -> dict[str, object]:
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
            "fixed_profit_usd": 0.0,
            "adaptive_profit_usd": 0.0,
            "avg_position_usd": 0.0,
            "max_position_usd": 0.0,
            "avg_profit_per_trade_usd": 0.0,
            "max_drawdown_usd": 0.0,
        }
    pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
    adaptive_profit = pd.to_numeric(df["profit_usd"], errors="coerce").fillna(0.0)
    fixed_profit = pnl / 100.0 * base_position
    equity = adaptive_profit.cumsum()
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
        "fixed_profit_usd": float(fixed_profit.sum()),
        "adaptive_profit_usd": float(adaptive_profit.sum()),
        "avg_position_usd": float(df["position_usd"].mean()),
        "max_position_usd": float(df["position_usd"].max()),
        "avg_profit_per_trade_usd": float(adaptive_profit.mean()),
        "max_drawdown_usd": float(dd.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v50 adaptive exposure by QQQ/SPY regime and setup quality.")
    parser.add_argument("--trades-csv", default="data/backtests/v46_trend_exit_momentum_or_breakout_trades_recent90_or5_score7_trendexit.csv")
    parser.add_argument("--source-variant", default="wide_trail")
    parser.add_argument("--data-dir", default="data/1m")
    parser.add_argument("--market-symbols", nargs="*", default=["QQQ", "SPY", "IWM"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--opening-range-minutes", type=int, default=5)
    parser.add_argument("--base-position-size", type=float, default=1000.0)
    parser.add_argument("--skip-zero-size", action=argparse.BooleanOptionalAction, default=True)

    # Strong market sizing.
    parser.add_argument("--size-strong-aplus", type=float, default=3.0)
    parser.add_argument("--size-strong-a", type=float, default=2.0)
    parser.add_argument("--size-strong-b", type=float, default=1.5)
    parser.add_argument("--size-strong-c", type=float, default=1.0)
    # Good market sizing.
    parser.add_argument("--size-good-aplus", type=float, default=2.0)
    parser.add_argument("--size-good-a", type=float, default=1.5)
    parser.add_argument("--size-good-b", type=float, default=1.0)
    parser.add_argument("--size-good-c", type=float, default=0.75)
    # Neutral market sizing.
    parser.add_argument("--size-neutral-aplus", type=float, default=1.5)
    parser.add_argument("--size-neutral-a", type=float, default=1.0)
    parser.add_argument("--size-neutral-b", type=float, default=0.75)
    parser.add_argument("--size-neutral-c", type=float, default=0.50)
    # Weak market sizing.
    parser.add_argument("--size-weak-aplus", type=float, default=1.0)
    parser.add_argument("--size-weak-a", type=float, default=0.75)
    parser.add_argument("--size-weak-b", type=float, default=0.50)
    parser.add_argument("--size-weak-c", type=float, default=0.0)
    # Bad market sizing.
    parser.add_argument("--size-bad-aplus", type=float, default=0.50)
    parser.add_argument("--size-bad-a", type=float, default=0.0)
    parser.add_argument("--size-bad-b", type=float, default=0.0)
    parser.add_argument("--size-bad-c", type=float, default=0.0)
    parser.add_argument("--size-unknown", type=float, default=0.50)
    args = parser.parse_args()

    print("Experiment: v50 adaptive exposure")
    print("Base: v46 wide_trail trades + real market ETF regime + dynamic position size")
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Source variant: {args.source_variant}")
    print(f"Base position: ${args.base_position_size:.2f}")

    trades = load_trades(args.trades_csv, args.source_variant)
    market_symbol, market = load_market_data(args.data_dir, args.market_symbols)
    joined = add_market_features(trades, market)

    baseline = joined.copy()
    baseline["position_mult"] = 1.0
    baseline["position_usd"] = args.base_position_size
    baseline["profit_usd"] = baseline["position_usd"] * baseline["pnl_pct"] / 100.0
    baseline["baseline_profit_usd"] = baseline["profit_usd"]

    adaptive = apply_adaptive_exposure(joined, args)

    summary = pd.DataFrame([
        summarize_portfolio(f"baseline_{args.source_variant}_fixed", baseline, args.base_position_size),
        summarize_portfolio(f"v50_adaptive_{market_symbol}", adaptive, args.base_position_size),
    ])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"recent{args.recent_days}_or{args.opening_range_minutes}_{args.source_variant}_{market_symbol}_adaptive"
    out_trades = output_dir / f"v50_adaptive_exposure_trades_{suffix}.csv"
    out_summary = output_dir / f"v50_adaptive_exposure_summary_{suffix}.csv"
    adaptive.to_csv(out_trades, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\nMarket regime symbol used: {market_symbol}")
    print(f"Saved adaptive trades CSV: {out_trades}")
    print(f"Saved summary CSV: {out_summary}")

    print("\n=== Portfolio comparison ===")
    cols = [
        "strategy", "count", "active_days", "symbols", "win_rate", "avg_pnl", "median_pnl", "total_pnl",
        "fixed_profit_usd", "adaptive_profit_usd", "avg_position_usd", "max_position_usd", "avg_profit_per_trade_usd", "max_drawdown_usd",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not adaptive.empty:
        print("\n=== Exposure matrix: count / profit_usd ===")
        matrix = adaptive.pivot_table(
            index="market_regime",
            columns="setup_quality",
            values="profit_usd",
            aggfunc=["count", "sum", "mean"],
            fill_value=0,
            observed=True,
        )
        print(matrix.to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Market regime summary ===")
        print(adaptive.groupby("market_regime")["profit_usd"].agg(["count", "mean", "median", "sum"]).sort_values("sum", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Setup quality summary ===")
        print(adaptive.groupby("setup_quality")["profit_usd"].agg(["count", "mean", "median", "sum"]).sort_values("sum", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Top days by adaptive profit ===")
        print(adaptive.groupby("session_date")["profit_usd"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False).head(20).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Worst days by adaptive profit ===")
        print(adaptive.groupby("session_date")["profit_usd"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=True).head(20).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- Good outcome: adaptive_profit_usd > fixed_profit_usd with acceptable drawdown.")
    print("- If profit rises but drawdown rises too much, cap max size or reduce strong/A+ multipliers.")
    print("- If adaptive underperforms, market/setup quality labels need recalibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
