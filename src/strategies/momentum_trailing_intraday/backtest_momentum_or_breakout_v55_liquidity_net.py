from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v54_execution_costs import (
    apply_costs,
    load_trades,
    summarize,
)


def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_price_num"] = pd.to_numeric(out.get("entry_price"), errors="coerce")
    out["position_usd_num"] = pd.to_numeric(out.get("position_usd"), errors="coerce")
    out["shares_est"] = out["position_usd_num"] / out["entry_price_num"].replace(0, pd.NA)

    # These columns exist in v51/v53 trade CSVs when inherited from v45/v46.
    for col in [
        "or_dollar_volume",
        "or_volume",
        "first_15m_dollar_volume",
        "first_15m_volume",
        "market_return_from_open_pct",
        "v45_score",
        "or_close_strength",
        "entry_minutes_from_open",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Position as % of opening-range dollar volume. Lower is easier to execute.
    if "or_dollar_volume" in out.columns:
        out["position_vs_or_dv_pct"] = out["position_usd_num"] / out["or_dollar_volume"].replace(0, pd.NA) * 100.0
    else:
        out["position_vs_or_dv_pct"] = pd.NA

    if "first_15m_dollar_volume" in out.columns:
        out["position_vs_first15_dv_pct"] = out["position_usd_num"] / out["first_15m_dollar_volume"].replace(0, pd.NA) * 100.0
    else:
        out["position_vs_first15_dv_pct"] = pd.NA

    return out


def liquidity_filter(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = add_liquidity_features(df)
    reasons = []
    for _, row in out.iterrows():
        r: list[str] = []
        price = float(row.get("entry_price_num", 0.0) or 0.0)
        shares = float(row.get("shares_est", 0.0) or 0.0)
        or_dv = row.get("or_dollar_volume", pd.NA)
        first15_dv = row.get("first_15m_dollar_volume", pd.NA)
        pos_or_pct = row.get("position_vs_or_dv_pct", pd.NA)
        pos_15_pct = row.get("position_vs_first15_dv_pct", pd.NA)
        quality = str(row.get("setup_quality", ""))
        regime = str(row.get("market_regime", ""))

        if price < args.min_entry_price:
            r.append("entry_price_too_low")
        if args.max_shares is not None and shares > args.max_shares:
            r.append("share_count_too_high")
        if pd.notna(or_dv) and float(or_dv) < args.min_or_dollar_volume:
            r.append("or_dollar_volume_too_low")
        if pd.notna(first15_dv) and float(first15_dv) < args.min_first15_dollar_volume:
            r.append("first15_dollar_volume_too_low")
        if pd.notna(pos_or_pct) and float(pos_or_pct) > args.max_position_vs_or_dv_pct:
            r.append("position_too_large_vs_or_dv")
        if pd.notna(pos_15_pct) and float(pos_15_pct) > args.max_position_vs_first15_dv_pct:
            r.append("position_too_large_vs_first15_dv")
        if args.drop_c_quality and quality == "C":
            r.append("drop_c_quality")
        if args.drop_bad_regime and regime == "bad":
            r.append("drop_bad_regime")

        reasons.append(";".join(r))

    out["v55_reject_reason"] = reasons
    accepted = out[out["v55_reject_reason"] == ""].copy()
    rejected = out[out["v55_reject_reason"] != ""].copy()
    return accepted, rejected


def summarize_pair(label: str, raw: pd.DataFrame, costed: pd.DataFrame, starting_cash: float) -> dict[str, object]:
    gross_profit = float(raw["profit_usd"].sum()) if not raw.empty else 0.0
    if costed.empty:
        return {
            "strategy": label,
            "trades": 0,
            "gross_profit_usd": 0.0,
            "execution_cost_usd": 0.0,
            "net_profit_usd": 0.0,
            "net_return_on_starting_cash_pct": 0.0,
            "net_win_rate": 0.0,
            "avg_net_profit_per_trade_usd": 0.0,
            "net_median_trade_usd": 0.0,
        }
    s = summarize(label, costed, starting_cash)
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="v55 liquidity filter + net execution-cost summary.")
    parser.add_argument("--trades-csv", default="data/backtests/v53_portfolio_accepted_cash20000_exposure20000_pos8.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--starting-cash", type=float, default=20000.0)

    # Liquidity / execution quality filters.
    parser.add_argument("--min-entry-price", type=float, default=5.0)
    parser.add_argument("--max-shares", type=float, default=750.0)
    parser.add_argument("--min-or-dollar-volume", type=float, default=500_000.0)
    parser.add_argument("--min-first15-dollar-volume", type=float, default=1_000_000.0)
    parser.add_argument("--max-position-vs-or-dv-pct", type=float, default=2.0)
    parser.add_argument("--max-position-vs-first15-dv-pct", type=float, default=1.0)
    parser.add_argument("--drop-c-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-bad-regime", action=argparse.BooleanOptionalAction, default=True)

    # Cost assumptions, same shape as v54.
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--min-commission-per-order", type=float, default=1.0)
    parser.add_argument("--sec-fee-rate", type=float, default=0.0000278)
    parser.add_argument("--finra-taf-per-share", type=float, default=0.000166)
    parser.add_argument("--finra-taf-cap", type=float, default=8.30)
    parser.add_argument("--base-slippage-bps", type=float, default=8.0)
    parser.add_argument("--spread-bps-per-side", type=float, default=5.0)
    parser.add_argument("--low-price-extra-slippage-bps", type=float, default=10.0)
    parser.add_argument("--mid-low-price-extra-slippage-bps", type=float, default=5.0)
    parser.add_argument("--medium-position-extra-slippage-bps", type=float, default=2.0)
    parser.add_argument("--large-position-extra-slippage-bps", type=float, default=4.0)
    parser.add_argument("--low-quality-extra-slippage-bps", type=float, default=4.0)
    parser.add_argument("--weak-regime-extra-slippage-bps", type=float, default=3.0)
    args = parser.parse_args()

    print("Experiment: v55 liquidity filter + net costs")
    print(f"Trades CSV: {args.trades_csv}")
    print("Important: spread/slippage are configurable assumptions, not real order-book measurements yet.")
    print(f"Min entry price: ${args.min_entry_price:.2f}")
    print(f"Min OR dollar volume: ${args.min_or_dollar_volume:.0f}")
    print(f"Min first15 dollar volume: ${args.min_first15_dollar_volume:.0f}")

    trades = load_trades(args.trades_csv)
    baseline_costed = apply_costs(trades, args)
    accepted_raw, rejected = liquidity_filter(trades, args)
    accepted_costed = apply_costs(accepted_raw, args) if not accepted_raw.empty else accepted_raw.copy()

    summary = pd.DataFrame([
        summarize_pair("baseline_before_v55_after_costs", trades, baseline_costed, args.starting_cash),
        summarize_pair("v55_liquidity_after_costs", accepted_raw, accepted_costed, args.starting_cash),
    ])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.trades_csv).stem
    out_accepted = output_dir / f"v55_liquidity_accepted_{stem}.csv"
    out_rejected = output_dir / f"v55_liquidity_rejected_{stem}.csv"
    out_costed = output_dir / f"v55_liquidity_costed_{stem}.csv"
    out_summary = output_dir / f"v55_liquidity_net_summary_{stem}.csv"
    accepted_raw.to_csv(out_accepted, index=False)
    rejected.to_csv(out_rejected, index=False)
    accepted_costed.to_csv(out_costed, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\nSaved accepted CSV: {out_accepted}")
    print(f"Saved rejected CSV: {out_rejected}")
    print(f"Saved costed CSV: {out_costed}")
    print(f"Saved summary CSV: {out_summary}")

    print("\n=== Net summary after execution costs ===")
    cols = [
        "strategy", "trades", "active_days", "symbols", "gross_profit_usd", "execution_cost_usd", "net_profit_usd",
        "net_return_on_starting_cash_pct", "avg_cost_per_trade_usd", "avg_net_profit_per_trade_usd",
        "gross_win_rate", "net_win_rate", "net_median_trade_usd", "max_drawdown_net_usd",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== v55 rejection reasons ===")
    if rejected.empty:
        print("none")
    else:
        print(rejected["v55_reject_reason"].value_counts().head(30).to_string())

    if not accepted_costed.empty:
        print("\n=== Net by market regime ===")
        if "market_regime" in accepted_costed.columns:
            print(accepted_costed.groupby("market_regime")["profit_after_costs_usd"].agg(["count", "mean", "median", "sum"]).sort_values("sum", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Net by setup quality ===")
        if "setup_quality" in accepted_costed.columns:
            print(accepted_costed.groupby("setup_quality")["profit_after_costs_usd"].agg(["count", "mean", "median", "sum"]).sort_values("sum", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Worst net trades after v55 ===")
        cols2 = ["session_date", "symbol", "entry_time", "entry_price", "position_usd", "profit_usd", "execution_cost_usd", "profit_after_costs_usd", "pnl_after_costs_pct"]
        print(accepted_costed.sort_values("profit_after_costs_usd").head(20)[cols2].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        print("\n=== Best net trades after v55 ===")
        print(accepted_costed.sort_values("profit_after_costs_usd", ascending=False).head(20)[cols2].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- v54 showed costs nearly killed gross edge; v55 checks if liquidity filters recover net edge.")
    print("- If net profit improves but trades collapse, loosen liquidity thresholds carefully.")
    print("- If net is still weak, paper trading should run with tiny size until real fills/slippage are measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
