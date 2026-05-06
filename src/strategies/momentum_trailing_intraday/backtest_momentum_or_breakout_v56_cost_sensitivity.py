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


SCENARIOS = {
    "ultra_optimistic": {
        "base_slippage_bps": 1.0,
        "spread_bps_per_side": 1.0,
    },
    "optimistic": {
        "base_slippage_bps": 2.0,
        "spread_bps_per_side": 2.0,
    },
    "moderate": {
        "base_slippage_bps": 4.0,
        "spread_bps_per_side": 3.0,
    },
    "conservative": {
        "base_slippage_bps": 8.0,
        "spread_bps_per_side": 5.0,
    },
    "extreme": {
        "base_slippage_bps": 12.0,
        "spread_bps_per_side": 8.0,
    },
}


def run_scenario(trades: pd.DataFrame, args: argparse.Namespace, name: str, cfg: dict[str, float]) -> tuple[pd.DataFrame, dict[str, object]]:
    class CostArgs:
        pass

    cost_args = CostArgs()
    cost_args.commission_per_share = args.commission_per_share
    cost_args.min_commission_per_order = args.min_commission_per_order
    cost_args.sec_fee_rate = args.sec_fee_rate
    cost_args.finra_taf_per_share = args.finra_taf_per_share
    cost_args.finra_taf_cap = args.finra_taf_cap

    cost_args.base_slippage_bps = cfg["base_slippage_bps"]
    cost_args.spread_bps_per_side = cfg["spread_bps_per_side"]

    cost_args.low_price_extra_slippage_bps = args.low_price_extra_slippage_bps
    cost_args.mid_low_price_extra_slippage_bps = args.mid_low_price_extra_slippage_bps
    cost_args.medium_position_extra_slippage_bps = args.medium_position_extra_slippage_bps
    cost_args.large_position_extra_slippage_bps = args.large_position_extra_slippage_bps
    cost_args.low_quality_extra_slippage_bps = args.low_quality_extra_slippage_bps
    cost_args.weak_regime_extra_slippage_bps = args.weak_regime_extra_slippage_bps

    costed = apply_costs(trades, cost_args)
    s = summarize(name, costed, args.starting_cash)
    s["scenario"] = name
    s["base_slippage_bps"] = cfg["base_slippage_bps"]
    s["spread_bps_per_side"] = cfg["spread_bps_per_side"]
    return costed, s


def main() -> int:
    parser = argparse.ArgumentParser(description="v56 execution cost sensitivity matrix")
    parser.add_argument("--trades-csv", default="data/backtests/v53_portfolio_accepted_cash20000_exposure20000_pos8.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--starting-cash", type=float, default=20000.0)

    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--min-commission-per-order", type=float, default=1.0)
    parser.add_argument("--sec-fee-rate", type=float, default=0.0000278)
    parser.add_argument("--finra-taf-per-share", type=float, default=0.000166)
    parser.add_argument("--finra-taf-cap", type=float, default=8.30)

    parser.add_argument("--low-price-extra-slippage-bps", type=float, default=10.0)
    parser.add_argument("--mid-low-price-extra-slippage-bps", type=float, default=5.0)
    parser.add_argument("--medium-position-extra-slippage-bps", type=float, default=2.0)
    parser.add_argument("--large-position-extra-slippage-bps", type=float, default=4.0)
    parser.add_argument("--low-quality-extra-slippage-bps", type=float, default=4.0)
    parser.add_argument("--weak-regime-extra-slippage-bps", type=float, default=3.0)
    args = parser.parse_args()

    print("Experiment: v56 execution cost sensitivity")
    print(f"Trades CSV: {args.trades_csv}")

    trades = load_trades(args.trades_csv)

    summaries = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name, cfg in SCENARIOS.items():
        print(f"\n=== Running scenario: {scenario_name} ===")
        print(f"slippage={cfg['base_slippage_bps']}bps/side spread={cfg['spread_bps_per_side']}bps/side")

        costed, summary = run_scenario(trades, args, scenario_name, cfg)
        summaries.append(summary)

        out_costed = output_dir / f"v56_cost_sensitivity_{scenario_name}.csv"
        costed.to_csv(out_costed, index=False)
        print(f"saved: {out_costed}")

    summary_df = pd.DataFrame(summaries)
    out_summary = output_dir / "v56_cost_sensitivity_summary.csv"
    summary_df.to_csv(out_summary, index=False)

    print("\n=== Cost sensitivity summary ===")
    cols = [
        "scenario",
        "base_slippage_bps",
        "spread_bps_per_side",
        "gross_profit_usd",
        "execution_cost_usd",
        "net_profit_usd",
        "net_return_on_starting_cash_pct",
        "avg_cost_per_trade_usd",
        "avg_net_profit_per_trade_usd",
        "net_win_rate",
        "max_drawdown_net_usd",
    ]
    print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nSaved summary: {out_summary}")

    print("\nInterpretation hints:")
    print("- If optimistic/moderate scenarios stay strongly profitable, real fill quality becomes the key unknown.")
    print("- If only ultra_optimistic works, the strategy likely depends on unrealistic execution.")
    print("- Use this before paper trading to decide initial size and acceptable liquidity universe.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
