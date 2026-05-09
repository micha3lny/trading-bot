from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v54_execution_costs import apply_costs, summarize
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v57_scaled_portfolio import (
    dynamic_position_size,
    load_trades,
)


DEFAULT_TRADES = "data/backtests/v53_portfolio_accepted_cash20000_exposure20000_pos8.csv"


def not_bad_regime(out: pd.DataFrame) -> pd.Series:
    if "market_regime" not in out.columns:
        return pd.Series(True, index=out.index)
    return out["market_regime"].astype(str).str.lower().ne("bad")


def not_c_quality(out: pd.DataFrame) -> pd.Series:
    if "setup_quality" not in out.columns:
        return pd.Series(True, index=out.index)
    return out["setup_quality"].astype(str).str.upper().ne("C")


def live_safe_core(out: pd.DataFrame) -> pd.Series:
    return (
        (out["entry_price"] >= 50.0)
        & (out["first_5m_high_pct"] >= 4.0)
        & (out["first_15m_high_pct"] >= 6.5)
        & (out["or_range_pct"] >= 5.0)
        & not_bad_regime(out)
        & not_c_quality(out)
    )


def apply_trade_filters(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = df.copy()

    if scenario == "baseline":
        return out

    # Research-only hindsight archetype filters.
    if scenario == "continuation_only":
        return out[
            (out["first_15m_high_pct"] >= 5.0)
            & (out["time_to_high_minutes"] >= 30.0)
        ].copy()

    if scenario == "price_ge_50":
        return out[out["entry_price"] >= 50.0].copy()

    if scenario == "price_50_250":
        return out[
            (out["entry_price"] >= 50.0)
            & (out["entry_price"] <= 250.0)
        ].copy()

    if scenario == "price_100_250":
        return out[
            (out["entry_price"] >= 100.0)
            & (out["entry_price"] <= 250.0)
        ].copy()

    if scenario == "price_100_250_continuation":
        return out[
            (out["entry_price"] >= 100.0)
            & (out["entry_price"] <= 250.0)
            & (out["first_15m_high_pct"] >= 5.0)
            & (out["time_to_high_minutes"] >= 30.0)
        ].copy()

    if scenario == "price_50_250_continuation":
        return out[
            (out["entry_price"] >= 50.0)
            & (out["entry_price"] <= 250.0)
            & (out["first_15m_high_pct"] >= 5.0)
            & (out["time_to_high_minutes"] >= 30.0)
        ].copy()

    # Live-safe filters: only use information known early in session / at entry.
    if scenario == "live_safe_expansion":
        return out[
            (out["first_5m_high_pct"] >= 4.0)
            & (out["first_15m_high_pct"] >= 6.5)
            & (out["or_range_pct"] >= 5.0)
        ].copy()

    if scenario == "live_safe_expansion_price_ge_50":
        return out[
            (out["entry_price"] >= 50.0)
            & (out["first_5m_high_pct"] >= 4.0)
            & (out["first_15m_high_pct"] >= 6.5)
            & (out["or_range_pct"] >= 5.0)
        ].copy()

    if scenario == "live_safe_full":
        return out[live_safe_core(out)].copy()

    if scenario == "live_safe_full_price_50_250":
        return out[
            live_safe_core(out)
            & (out["entry_price"] <= 250.0)
        ].copy()

    if scenario == "live_safe_full_price_100_250":
        return out[
            live_safe_core(out)
            & (out["entry_price"] >= 100.0)
            & (out["entry_price"] <= 250.0)
        ].copy()

    raise ValueError(f"Unknown scenario: {scenario}")


def simulate_portfolio(trades: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def run_scenario(label: str, trades: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = apply_trade_filters(trades, label)
    accepted, rejected = simulate_portfolio(filtered, args)

    if accepted.empty:
        summary = pd.DataFrame([{
            "strategy": f"v60_{label}",
            "candidate_trades_after_filter": len(filtered),
            "trades": 0,
            "active_days": 0,
            "symbols": 0,
            "gross_profit_usd": 0.0,
            "execution_cost_usd": 0.0,
            "net_profit_usd": 0.0,
            "net_return_on_starting_cash_pct": 0.0,
        }])
        return summary, accepted, rejected

    costed = apply_costs(accepted, args)
    summary = pd.DataFrame([summarize(f"v60_{label}", costed, args.starting_cash)])
    summary.insert(1, "candidate_trades_after_filter", len(filtered))
    summary.insert(2, "rejected_by_portfolio", len(rejected))
    return summary, costed, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description="v60 filter scenario tester")
    parser.add_argument("--trades-csv", default=DEFAULT_TRADES)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--starting-cash", type=float, default=25_000.0)
    parser.add_argument("--max-gross-exposure", type=float, default=25_000.0)
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--min-position-usd", type=float, default=1_000.0)
    parser.add_argument("--max-position-usd", type=float, default=3_000.0)

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

    print("Experiment: v60 filter scenarios")
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Position range: ${args.min_position_usd:.0f}-${args.max_position_usd:.0f}")

    trades = load_trades(args.trades_csv)
    numeric_cols = [
        "entry_price",
        "first_5m_high_pct",
        "first_15m_high_pct",
        "or_range_pct",
        "time_to_high_minutes",
        "pnl_pct",
    ]
    for col in numeric_cols:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    scenarios = [
        "baseline",
        "continuation_only",
        "price_ge_50",
        "price_50_250",
        "price_100_250",
        "price_50_250_continuation",
        "price_100_250_continuation",
        "live_safe_expansion",
        "live_safe_expansion_price_ge_50",
        "live_safe_full",
        "live_safe_full_price_50_250",
        "live_safe_full_price_100_250",
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for scenario in scenarios:
        print(f"\n=== Running {scenario} ===")
        summary, costed, rejected = run_scenario(scenario, trades, args)
        summaries.append(summary)
        costed.to_csv(output_dir / f"v60_{scenario}_costed.csv", index=False)
        rejected.to_csv(output_dir / f"v60_{scenario}_rejected.csv", index=False)
        summary.to_csv(output_dir / f"v60_{scenario}_summary.csv", index=False)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_dir / "v60_filter_scenarios_summary.csv", index=False)

    print("\n=== v60 filter scenario summary ===")
    print(combined.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved: {output_dir / 'v60_filter_scenarios_summary.csv'}")

    print("\nInterpretation hints:")
    print("- price_100_250 tests the profitable price bucket discovered in v60 expectancy diagnostics.")
    print("- continuation_only uses hindsight time_to_high_minutes, so treat it as research/proxy, not a live rule.")
    print("- live_safe_* scenarios use only price, first 5m/15m expansion, OR range, regime, and setup quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
