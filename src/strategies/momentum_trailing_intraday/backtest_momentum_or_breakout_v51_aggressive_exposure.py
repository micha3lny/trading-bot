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
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v50_adaptive_exposure import (
    market_regime_label,
    setup_quality_label,
    summarize_portfolio,
)


def exposure_multiplier_v51(regime: str, quality: str, args: argparse.Namespace) -> float:
    """More aggressive v50 calibration.

    Hypothesis from v50:
    - C setups barely contribute and should receive minimal / zero exposure.
    - Strong market + A/A+ setups carry the system and can justify larger exposure.
    - Bad regime should generally be avoided except very small A+ exposure.
    """
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


def apply_v51_exposure(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = trades.copy()
    out["market_regime"] = out.apply(market_regime_label, axis=1)
    out["setup_quality"] = out.apply(setup_quality_label, axis=1)
    out["position_mult"] = out.apply(
        lambda r: exposure_multiplier_v51(str(r["market_regime"]), str(r["setup_quality"]), args), axis=1
    )
    if args.skip_zero_size:
        out = out[out["position_mult"] > 0].copy()
    out["position_usd"] = args.base_position_size * out["position_mult"]
    out["profit_usd"] = out["position_usd"] * out["pnl_pct"] / 100.0
    out["baseline_profit_usd"] = args.base_position_size * out["pnl_pct"] / 100.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="v51 aggressive adaptive exposure calibration.")
    parser.add_argument("--trades-csv", default="data/backtests/v46_trend_exit_momentum_or_breakout_trades_recent90_or5_score7_trendexit.csv")
    parser.add_argument("--source-variant", default="wide_trail")
    parser.add_argument("--data-dir", default="data/1m")
    parser.add_argument("--market-symbols", nargs="*", default=["QQQ", "SPY", "IWM"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--opening-range-minutes", type=int, default=5)
    parser.add_argument("--base-position-size", type=float, default=1000.0)
    parser.add_argument("--skip-zero-size", action=argparse.BooleanOptionalAction, default=True)

    # More aggressive than v50 by default.
    parser.add_argument("--size-strong-aplus", type=float, default=4.0)
    parser.add_argument("--size-strong-a", type=float, default=3.0)
    parser.add_argument("--size-strong-b", type=float, default=1.75)
    parser.add_argument("--size-strong-c", type=float, default=0.25)

    parser.add_argument("--size-good-aplus", type=float, default=2.25)
    parser.add_argument("--size-good-a", type=float, default=1.75)
    parser.add_argument("--size-good-b", type=float, default=1.0)
    parser.add_argument("--size-good-c", type=float, default=0.25)

    parser.add_argument("--size-neutral-aplus", type=float, default=1.75)
    parser.add_argument("--size-neutral-a", type=float, default=1.25)
    parser.add_argument("--size-neutral-b", type=float, default=0.75)
    parser.add_argument("--size-neutral-c", type=float, default=0.0)

    parser.add_argument("--size-weak-aplus", type=float, default=1.0)
    parser.add_argument("--size-weak-a", type=float, default=0.75)
    parser.add_argument("--size-weak-b", type=float, default=0.50)
    parser.add_argument("--size-weak-c", type=float, default=0.0)

    parser.add_argument("--size-bad-aplus", type=float, default=0.25)
    parser.add_argument("--size-bad-a", type=float, default=0.0)
    parser.add_argument("--size-bad-b", type=float, default=0.0)
    parser.add_argument("--size-bad-c", type=float, default=0.0)
    parser.add_argument("--size-unknown", type=float, default=0.25)
    args = parser.parse_args()

    print("Experiment: v51 aggressive adaptive exposure")
    print("Base: v46 wide_trail trades + real market ETF regime + aggressive v50 sizing calibration")
    print("Hypothesis: cut C exposure, increase strong-market A/A+ exposure.")
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

    adaptive = apply_v51_exposure(joined, args)

    summary = pd.DataFrame([
        summarize_portfolio(f"baseline_{args.source_variant}_fixed", baseline, args.base_position_size),
        summarize_portfolio(f"v51_aggressive_{market_symbol}", adaptive, args.base_position_size),
    ])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"recent{args.recent_days}_or{args.opening_range_minutes}_{args.source_variant}_{market_symbol}_aggressive"
    out_trades = output_dir / f"v51_aggressive_exposure_trades_{suffix}.csv"
    out_summary = output_dir / f"v51_aggressive_exposure_summary_{suffix}.csv"
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
    print("- Good outcome: higher adaptive profit than v50 without unacceptable drawdown increase.")
    print("- If drawdown increases too much, reduce --size-strong-aplus and --size-strong-a.")
    print("- If profit falls, C setups were contributing more diversification than expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
