from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_EVENTS = "data/live/v59_simulated_trade_events.csv"
DEFAULT_DAILY = "data/live/v59_daily_report.csv"
DEFAULT_SUMMARY = "data/live/v59_simulated_summary.csv"
DEFAULT_OUTPUT = "data/live/v59_daily_report_analysis.csv"


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze v59 daily paper/simulated trading reports")
    parser.add_argument("--events", default=DEFAULT_EVENTS)
    parser.add_argument("--daily", default=DEFAULT_DAILY)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--starting-cash", type=float, default=25_000.0)
    args = parser.parse_args()

    print("=== v59 daily report analyzer ===")
    print(f"Events: {args.events}")
    print(f"Daily: {args.daily}")

    events = read_csv(args.events)
    daily = read_csv(args.daily)
    summary = read_csv(args.summary)

    if events.empty:
        print("No trade events found")
        return 1

    exits = events[events["event"] == "EXIT"].copy()
    entries = events[events["event"] == "ENTRY"].copy()
    if exits.empty:
        print("No exits found")
        return 1

    for df in [exits, entries]:
        for col in [
            "quantity",
            "entry_price",
            "exit_price",
            "gross_pnl_usd",
            "commission_usd",
            "reg_fees_usd",
            "total_costs_usd",
            "net_pnl_usd",
            "score",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    exits["entry_notional_usd"] = exits["quantity"].abs() * exits["entry_price"]
    exits["exit_notional_usd"] = exits["quantity"].abs() * exits["exit_price"]
    exits["round_trip_notional_usd"] = exits["entry_notional_usd"] + exits["exit_notional_usd"]
    exits["cost_bps_of_round_trip_notional"] = exits["total_costs_usd"] / exits["round_trip_notional_usd"] * 10_000.0
    exits["net_bps_of_entry_notional"] = exits["net_pnl_usd"] / exits["entry_notional_usd"] * 10_000.0
    exits["gross_bps_of_entry_notional"] = exits["gross_pnl_usd"] / exits["entry_notional_usd"] * 10_000.0

    total_gross = float(exits["gross_pnl_usd"].sum())
    total_costs = float(exits["total_costs_usd"].sum())
    total_net = float(exits["net_pnl_usd"].sum())
    total_notional = float(exits["round_trip_notional_usd"].sum())
    trades = len(exits)

    gross_to_cost_ratio = None
    if total_costs > 0:
        gross_to_cost_ratio = total_gross / total_costs

    analysis_rows = [{
        "metric": "trades",
        "value": trades,
        "interpretation": "Number of completed round-trip trades.",
    }, {
        "metric": "gross_pnl_usd",
        "value": total_gross,
        "interpretation": "PnL before commissions/regulatory fees.",
    }, {
        "metric": "total_costs_usd",
        "value": total_costs,
        "interpretation": "Commissions + SEC/FINRA fees.",
    }, {
        "metric": "net_pnl_usd",
        "value": total_net,
        "interpretation": "PnL after estimated execution costs.",
    }, {
        "metric": "avg_cost_per_trade_usd",
        "value": total_costs / trades if trades else 0.0,
        "interpretation": "Average explicit broker/regulatory cost per round-trip.",
    }, {
        "metric": "avg_net_per_trade_usd",
        "value": total_net / trades if trades else 0.0,
        "interpretation": "Average net edge per completed trade.",
    }, {
        "metric": "avg_cost_bps_round_trip_notional",
        "value": total_costs / total_notional * 10_000.0 if total_notional else 0.0,
        "interpretation": "Explicit costs relative to total traded notional.",
    }, {
        "metric": "gross_to_cost_ratio",
        "value": gross_to_cost_ratio if gross_to_cost_ratio is not None else 0.0,
        "interpretation": "If below 2.0, costs are too large relative to gross edge.",
    }]

    analysis = pd.DataFrame(analysis_rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(out, index=False)

    print("\n=== Cost / edge analysis ===")
    print(analysis.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Best net trades ===")
    cols = [
        "timestamp_utc", "symbol", "quantity", "entry_price", "exit_price", "gross_pnl_usd",
        "total_costs_usd", "net_pnl_usd", "net_bps_of_entry_notional", "reason", "score",
    ]
    print(exits.sort_values("net_pnl_usd", ascending=False).head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Worst net trades ===")
    print(exits.sort_values("net_pnl_usd", ascending=True).head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== By symbol ===")
    by_symbol = exits.groupby("symbol").agg(
        trades=("symbol", "count"),
        gross_pnl_usd=("gross_pnl_usd", "sum"),
        total_costs_usd=("total_costs_usd", "sum"),
        net_pnl_usd=("net_pnl_usd", "sum"),
        avg_net_trade_usd=("net_pnl_usd", "mean"),
        avg_cost_bps=("cost_bps_of_round_trip_notional", "mean"),
    ).reset_index().sort_values("net_pnl_usd", ascending=False)
    print(by_symbol.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved analysis: {out}")

    print("\nInterpretation hints:")
    if trades and total_costs > abs(total_gross):
        print("- Costs are larger than gross edge. Increase average position size, reduce trade count, or require stronger signals.")
    elif gross_to_cost_ratio is not None and gross_to_cost_ratio < 2.0:
        print("- Gross edge is positive but not comfortably above costs. Execution quality remains critical.")
    else:
        print("- Gross edge is comfortably above explicit costs in this sample.")
    print("- Real IBKR commission reports should replace estimated costs once broker execution is connected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
