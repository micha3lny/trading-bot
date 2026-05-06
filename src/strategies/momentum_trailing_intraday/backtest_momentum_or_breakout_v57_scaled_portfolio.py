from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v54_execution_costs import apply_costs, summarize
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v53_portfolio_constraints import quality_rank


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce")
    return df.dropna(subset=["entry_dt", "exit_dt"]).copy()


def dynamic_position_size(row: pd.Series, args: argparse.Namespace) -> float:
    score = quality_rank(row)

    regime = str(row.get("market_regime", "neutral"))
    quality = str(row.get("setup_quality", "B"))

    size = args.min_position_usd

    if quality == "B":
        size = 750
    if quality == "A":
        size = 1250
    if quality == "A+":
        size = 1750

    if regime == "strong":
        size += 250
    elif regime == "good":
        size += 100
    elif regime == "bad":
        size -= 250

    if score >= 10:
        size += 250

    return float(max(args.min_position_usd, min(args.max_position_usd, size)))


def main() -> int:
    parser = argparse.ArgumentParser(description="v57 scaled portfolio simulation")
    parser.add_argument("--trades-csv", default="data/backtests/v53_portfolio_accepted_cash20000_exposure20000_pos8.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--starting-cash", type=float, default=25000.0)
    parser.add_argument("--max-gross-exposure", type=float, default=25000.0)
    parser.add_argument("--max-positions", type=int, default=10)

    parser.add_argument("--min-position-usd", type=float, default=500.0)
    parser.add_argument("--max-position-usd", type=float, default=2000.0)

    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--min-commission-per-order", type=float, default=1.0)
    parser.add_argument("--sec-fee-rate", type=float, default=0.0000278)
    parser.add_argument("--finra-taf-per-share", type=float, default=0.000166)
    parser.add_argument("--finra-taf-cap", type=float, default=8.30)
    parser.add_argument("--base-slippage-bps", type=float, default=4.0)
    parser.add_argument("--spread-bps-per-side", type=float, default=3.0)
    parser.add_argument("--low-price-extra-slippage_bps", type=float, default=10.0)
    parser.add_argument("--mid-low-price-extra_slippage_bps", type=float, default=5.0)
    parser.add_argument("--medium-position-extra_slippage_bps", type=float, default=2.0)
    parser.add_argument("--large-position-extra_slippage_bps", type=float, default=4.0)
    parser.add_argument("--low-quality-extra_slippage_bps", type=float, default=4.0)
    parser.add_argument("--weak-regime-extra_slippage_bps", type=float, default=3.0)

    args = parser.parse_args()

    print("Experiment: v57 scaled portfolio")
    print("Goal: 25k account with realistic dynamic sizing 500-2000 USD")

    trades = load_trades(args.trades_csv)
    trades = trades.sort_values("entry_dt").reset_index(drop=True)

    accepted = []
    rejected = []

    cash = args.starting_cash
    open_positions = []

    for _, row in trades.iterrows():
        now = row["entry_dt"]

        still_open = []
        for pos in open_positions:
            if pos["exit_dt"] <= now:
                cash += pos["position_usd"] + pos["profit_usd"]
            else:
                still_open.append(pos)
        open_positions = still_open

        open_exposure = sum(float(p["position_usd"]) for p in open_positions)

        row = row.copy()
        row["position_usd"] = dynamic_position_size(row, args)

        pnl_pct = float(row.get("pnl_pct", 0.0))
        row["profit_usd"] = row["position_usd"] * pnl_pct / 100.0

        if len(open_positions) >= args.max_positions:
            rejected.append({**row.to_dict(), "reason": "max_positions"})
            continue

        if open_exposure + row["position_usd"] > args.max_gross_exposure:
            rejected.append({**row.to_dict(), "reason": "max_exposure"})
            continue

        if cash < row["position_usd"]:
            rejected.append({**row.to_dict(), "reason": "insufficient_cash"})
            continue

        cash -= row["position_usd"]
        accepted.append(row.to_dict())
        open_positions.append(row.to_dict())

    accepted_df = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)

    if accepted_df.empty:
        print("No accepted trades")
        return 1

    class CostArgs:
        pass

    c = CostArgs()
    for k, v in vars(args).items():
        setattr(c, k.replace('-', '_'), v)

    c.low_price_extra_slippage_bps = 10.0
    c.mid_low_price_extra_slippage_bps = 5.0
    c.medium_position_extra_slippage_bps = 2.0
    c.large_position_extra_slippage_bps = 4.0
    c.low_quality_extra_slippage_bps = 4.0
    c.weak_regime_extra_slippage_bps = 3.0

    costed = apply_costs(accepted_df, c)

    summary = pd.DataFrame([
        summarize("v57_scaled_portfolio", costed, args.starting_cash)
    ])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_costed = output_dir / "v57_scaled_portfolio_costed.csv"
    out_summary = output_dir / "v57_scaled_portfolio_summary.csv"

    costed.to_csv(out_costed, index=False)
    summary.to_csv(out_summary, index=False)

    print("\n=== v57 summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nAccepted trades: {len(costed)}")
    print(f"Rejected trades: {len(rejected_df)}")
    print(f"Saved costed trades: {out_costed}")
    print(f"Saved summary: {out_summary}")

    print("\nInterpretation hints:")
    print("- This simulates a more realistic retail intraday account.")
    print("- Dynamic sizing should reduce commission drag versus too many tiny trades.")
    print("- If moderate-cost assumptions remain profitable, paper trading is justified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
