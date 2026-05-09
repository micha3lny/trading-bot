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
DEFAULT_WIDE = "data/universe/v62_symbols_wide.txt"
DEFAULT_LIQUID = "data/universe/v62_symbols_liquid.txt"
DEFAULT_V64_FOCUS = "data/universe/v64_symbols_focus.txt"
DEFAULT_V64_TRADEABLE = "data/universe/v64_symbols_tradeable.txt"


def read_symbols(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing universe file: {path}")
    return {line.strip().upper() for line in p.read_text().splitlines() if line.strip()}


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


def run_universe(label: str, trades: pd.DataFrame, symbols: set[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = trades[trades["symbol"].astype(str).str.upper().isin(symbols)].copy()

    accepted_df, rejected_df = simulate_portfolio(filtered, args)
    if accepted_df.empty:
        summary = pd.DataFrame([{
            "strategy": label,
            "universe_symbols": len(symbols),
            "candidate_trades_after_universe_filter": len(filtered),
            "rejected_by_portfolio": len(rejected_df),
            "trades": 0,
            "active_days": 0,
            "symbols": 0,
            "gross_profit_usd": 0.0,
            "execution_cost_usd": 0.0,
            "net_profit_usd": 0.0,
            "net_return_on_starting_cash_pct": 0.0,
        }])
        return summary, accepted_df, rejected_df

    costed = apply_costs(accepted_df, args)
    summary = pd.DataFrame([summarize(label, costed, args.starting_cash)])
    summary.insert(1, "universe_symbols", len(symbols))
    summary.insert(2, "candidate_trades_after_universe_filter", len(filtered))
    summary.insert(3, "rejected_by_portfolio", len(rejected_df))
    return summary, costed, rejected_df


def main() -> int:
    parser = argparse.ArgumentParser(description="v58 universe comparison: wide/liquid/v64 alpha")
    parser.add_argument("--trades-csv", default=DEFAULT_TRADES)
    parser.add_argument("--wide-symbols", default=DEFAULT_WIDE)
    parser.add_argument("--liquid-symbols", default=DEFAULT_LIQUID)
    parser.add_argument("--v64-focus-symbols", default=DEFAULT_V64_FOCUS)
    parser.add_argument("--v64-tradeable-symbols", default=DEFAULT_V64_TRADEABLE)
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
    parser.add_argument("--low-price-extra-slippage-bps", type=float, default=10.0)
    parser.add_argument("--mid-low-price-extra-slippage-bps", type=float, default=5.0)
    parser.add_argument("--medium-position-extra-slippage-bps", type=float, default=2.0)
    parser.add_argument("--large-position-extra-slippage-bps", type=float, default=4.0)
    parser.add_argument("--low-quality-extra-slippage-bps", type=float, default=4.0)
    parser.add_argument("--weak-regime-extra-slippage-bps", type=float, default=3.0)
    args = parser.parse_args()

    print("Experiment: v58 universe comparison")
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Position range: ${args.min_position_usd:.0f}-${args.max_position_usd:.0f}")
    print(f"Starting cash: ${args.starting_cash:.0f}")

    trades = load_trades(args.trades_csv)
    trades["symbol"] = trades["symbol"].astype(str).str.upper()

    scenarios = [
        ("v58_wide_universe", read_symbols(args.wide_symbols)),
        ("v58_liquid_universe", read_symbols(args.liquid_symbols)),
        ("v64_focus_universe", read_symbols(args.v64_focus_symbols)),
        ("v64_tradeable_universe", read_symbols(args.v64_tradeable_symbols)),
    ]

    summaries = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, symbols in scenarios:
        print(f"\n=== Running {label} ===")
        print(f"Universe symbols: {len(symbols)}")
        summary, costed, rejected = run_universe(label, trades, symbols, args)
        summaries.append(summary)

        costed_path = output_dir / f"{label}_costed.csv"
        rejected_path = output_dir / f"{label}_rejected.csv"
        summary_path = output_dir / f"{label}_summary.csv"

        costed.to_csv(costed_path, index=False)
        rejected.to_csv(rejected_path, index=False)
        summary.to_csv(summary_path, index=False)

        print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"Saved costed: {costed_path}")
        print(f"Saved rejected: {rejected_path}")

    combined = pd.concat(summaries, ignore_index=True)
    combined_path = output_dir / "v58_universe_comparison_summary.csv"
    combined.to_csv(combined_path, index=False)

    print("\n=== v58 universe comparison summary ===")
    print(combined.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved combined summary: {combined_path}")

    print("\nInterpretation hints:")
    print("- Wide universe should capture more movers but may carry more execution/noise risk.")
    print("- Liquid universe reduces execution risk but may miss smaller momentum names.")
    print("- v64 focus/tradeable test whether alpha-style universe selection improves expectancy.")
    print("- Compare net profit, drawdown, trade count, and avg net profit per trade before deleting data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
