from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_EVENTS = "data/live/v59_simulated_trade_events.csv"

POSITION_SIZES = [100, 250, 500, 1000, 2000]
FIXED_COMMISSION_PER_SIDE = 1.00
SEC_FEE_RATE = 0.0000278
FINRA_TAF_PER_SHARE = 0.000166
FINRA_TAF_CAP = 8.30


def estimate_round_trip_cost(quantity: float, exit_notional: float) -> float:
    commission = FIXED_COMMISSION_PER_SIDE * 2.0
    sec_fee = exit_notional * SEC_FEE_RATE
    taf = min(abs(quantity) * FINRA_TAF_PER_SHARE, FINRA_TAF_CAP)
    return commission + sec_fee + taf


def main() -> int:
    parser = argparse.ArgumentParser(description="Position size sensitivity analysis")
    parser.add_argument("--events", default=DEFAULT_EVENTS)
    args = parser.parse_args()

    print("=== v59 position size sensitivity ===")

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"Missing events file: {events_path}")
        return 1

    events = pd.read_csv(events_path)
    exits = events[events["event"] == "EXIT"].copy()

    if exits.empty:
        print("No EXIT events found")
        return 1

    numeric_cols = [
        "quantity",
        "entry_price",
        "exit_price",
        "gross_pnl_usd",
    ]

    for col in numeric_cols:
        exits[col] = pd.to_numeric(exits[col], errors="coerce")

    rows = []

    for target_size in POSITION_SIZES:
        gross_total = 0.0
        net_total = 0.0
        costs_total = 0.0
        trade_count = 0

        for _, trade in exits.iterrows():
            entry_price = trade["entry_price"]
            exit_price = trade["exit_price"]
            base_qty = trade["quantity"]

            if entry_price <= 0:
                continue

            scaled_qty = max(1, round(target_size / entry_price))

            pnl_per_share = exit_price - entry_price
            gross_pnl = pnl_per_share * scaled_qty

            exit_notional = abs(scaled_qty * exit_price)
            cost = estimate_round_trip_cost(scaled_qty, exit_notional)
            net_pnl = gross_pnl - cost

            gross_total += gross_pnl
            net_total += net_pnl
            costs_total += cost
            trade_count += 1

        gross_to_cost = gross_total / costs_total if costs_total else 0.0

        rows.append({
            "position_size_usd": target_size,
            "trades": trade_count,
            "gross_pnl_usd": round(gross_total, 2),
            "costs_usd": round(costs_total, 2),
            "net_pnl_usd": round(net_total, 2),
            "avg_cost_per_trade_usd": round(costs_total / trade_count, 2) if trade_count else 0.0,
            "avg_net_per_trade_usd": round(net_total / trade_count, 2) if trade_count else 0.0,
            "gross_to_cost_ratio": round(gross_to_cost, 2),
        })

    result = pd.DataFrame(rows)

    print("\n=== Sensitivity table ===")
    print(result.to_string(index=False))

    print("\nInterpretation hints:")
    print("- Larger positions dilute fixed commissions.")
    print("- If gross_to_cost_ratio stays below 1-2x, strategy edge is too weak for IBKR Fixed.")
    print("- Tiered commissions may become superior once trading frequency increases.")

    out = Path("data/live/v59_position_size_sensitivity.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
