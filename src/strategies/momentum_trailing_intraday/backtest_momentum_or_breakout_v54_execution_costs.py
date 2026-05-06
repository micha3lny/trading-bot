from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR


def load_trades(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    required = {"symbol", "session_date", "entry_time", "exit_time", "entry_price", "position_usd", "profit_usd", "pnl_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {p}: {missing}")
    df = df.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date.astype(str)
    return df.dropna(subset=["entry_dt", "exit_dt"]).sort_values(["entry_dt", "symbol"]).reset_index(drop=True)


def estimate_shares(position_usd: float, entry_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return position_usd / entry_price


def slippage_bps(row: pd.Series, args: argparse.Namespace) -> float:
    """Simple conservative momentum slippage model in basis points per side.

    This is intentionally configurable because real slippage depends on order type,
    spread, liquidity, route, and market conditions.
    """
    bps = args.base_slippage_bps
    price = float(row.get("entry_price", 0.0) or 0.0)
    position = float(row.get("position_usd", 0.0) or 0.0)
    quality = str(row.get("setup_quality", ""))
    regime = str(row.get("market_regime", ""))

    if price < 2:
        bps += args.low_price_extra_slippage_bps
    elif price < 5:
        bps += args.mid_low_price_extra_slippage_bps

    if position >= 3000:
        bps += args.large_position_extra_slippage_bps
    elif position >= 2000:
        bps += args.medium_position_extra_slippage_bps

    if quality == "C":
        bps += args.low_quality_extra_slippage_bps
    if regime in {"weak", "bad"}:
        bps += args.weak_regime_extra_slippage_bps

    return max(0.0, bps)


def apply_costs(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["shares_est"] = out.apply(lambda r: estimate_shares(float(r["position_usd"]), float(r["entry_price"])), axis=1)
    out["commission_entry_usd"] = (out["shares_est"] * args.commission_per_share).clip(lower=args.min_commission_per_order)
    out["commission_exit_usd"] = (out["shares_est"] * args.commission_per_share).clip(lower=args.min_commission_per_order)
    out["commission_total_usd"] = out["commission_entry_usd"] + out["commission_exit_usd"]

    # SEC/FINRA mostly on sell side, approximate on notional exit.
    out["sell_notional_est_usd"] = out["position_usd"] * (1.0 + out["pnl_pct"] / 100.0)
    out["sec_fee_usd"] = out["sell_notional_est_usd"].clip(lower=0) * args.sec_fee_rate
    out["finra_taf_usd"] = (out["shares_est"] * args.finra_taf_per_share).clip(upper=args.finra_taf_cap)

    out["slippage_bps_per_side"] = out.apply(lambda r: slippage_bps(r, args), axis=1)
    out["slippage_total_usd"] = out["position_usd"] * (out["slippage_bps_per_side"] * 2.0) / 10_000.0

    out["spread_bps_per_side"] = args.spread_bps_per_side
    out["spread_total_usd"] = out["position_usd"] * (out["spread_bps_per_side"] * 2.0) / 10_000.0

    out["execution_cost_usd"] = (
        out["commission_total_usd"]
        + out["sec_fee_usd"]
        + out["finra_taf_usd"]
        + out["slippage_total_usd"]
        + out["spread_total_usd"]
    )
    out["profit_after_costs_usd"] = out["profit_usd"] - out["execution_cost_usd"]
    out["pnl_after_costs_pct"] = out["profit_after_costs_usd"] / out["position_usd"] * 100.0
    return out


def summarize(label: str, df: pd.DataFrame, starting_cash: float | None = None) -> dict[str, object]:
    if df.empty:
        return {}
    gross = pd.to_numeric(df["profit_usd"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(df["profit_after_costs_usd"], errors="coerce").fillna(0.0)
    equity = net.cumsum()
    dd = equity - equity.cummax()
    total_cost = pd.to_numeric(df["execution_cost_usd"], errors="coerce").fillna(0.0)
    ret = None if starting_cash is None or starting_cash <= 0 else float(net.sum() / starting_cash * 100.0)
    return {
        "strategy": label,
        "trades": int(len(df)),
        "active_days": int(df["session_date"].nunique()),
        "symbols": int(df["symbol"].nunique()),
        "gross_profit_usd": float(gross.sum()),
        "execution_cost_usd": float(total_cost.sum()),
        "net_profit_usd": float(net.sum()),
        "net_return_on_starting_cash_pct": ret,
        "avg_cost_per_trade_usd": float(total_cost.mean()),
        "avg_net_profit_per_trade_usd": float(net.mean()),
        "gross_win_rate": float((gross > 0).mean() * 100.0),
        "net_win_rate": float((net > 0).mean() * 100.0),
        "gross_avg_trade_usd": float(gross.mean()),
        "net_avg_trade_usd": float(net.mean()),
        "net_median_trade_usd": float(net.median()),
        "max_drawdown_net_usd": float(dd.min()),
        "worst_net_trade_usd": float(net.min()),
        "best_net_trade_usd": float(net.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v54 execution costs simulator for v53/v51 trade CSVs.")
    parser.add_argument("--trades-csv", default="data/backtests/v53_portfolio_accepted_cash20000_exposure20000_pos8.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--starting-cash", type=float, default=20000.0)

    # IBKR-like configurable assumptions. Defaults are deliberately conservative placeholders.
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

    print("Experiment: v54 execution costs")
    print(f"Trades CSV: {args.trades_csv}")
    print("Costs included: commission, SEC fee, FINRA TAF, spread, slippage")
    print(f"Base slippage: {args.base_slippage_bps:.2f} bps/side")
    print(f"Spread: {args.spread_bps_per_side:.2f} bps/side")

    trades = load_trades(args.trades_csv)
    costed = apply_costs(trades, args)
    summary = pd.DataFrame([summarize("v54_after_execution_costs", costed, args.starting_cash)])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.trades_csv).stem
    out_trades = output_dir / f"v54_execution_costs_{stem}.csv"
    out_summary = output_dir / f"v54_execution_costs_summary_{stem}.csv"
    costed.to_csv(out_trades, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\nSaved costed trades CSV: {out_trades}")
    print(f"Saved summary CSV: {out_summary}")

    print("\n=== Execution-cost summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Cost components ===")
    components = {
        "commission_total_usd": float(costed["commission_total_usd"].sum()),
        "sec_fee_usd": float(costed["sec_fee_usd"].sum()),
        "finra_taf_usd": float(costed["finra_taf_usd"].sum()),
        "slippage_total_usd": float(costed["slippage_total_usd"].sum()),
        "spread_total_usd": float(costed["spread_total_usd"].sum()),
    }
    for k, v in components.items():
        print(f"{k}: ${v:.2f}")

    print("\n=== Worst net trades ===")
    cols = ["session_date", "symbol", "entry_time", "position_usd", "profit_usd", "execution_cost_usd", "profit_after_costs_usd", "pnl_after_costs_pct"]
    print(costed.sort_values("profit_after_costs_usd").head(20)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Best net trades ===")
    print(costed.sort_values("profit_after_costs_usd", ascending=False).head(20)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- If net profit remains positive after conservative costs, paper trading is justified.")
    print("- If costs eat most profit, reduce low-quality/low-price names and require better liquidity.")
    print("- Defaults are configurable; compare conservative vs optimistic assumptions before going live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
