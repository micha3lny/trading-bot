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
DEFAULT_ALPHA_RANK = "data/universe/v64_universe_alpha_ranked.csv"
DEFAULT_WIDE = "data/universe/v62_symbols_wide.txt"


def read_symbols(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing universe file: {path}")
    return {line.strip().upper() for line in p.read_text().splitlines() if line.strip()}


def load_alpha_rank(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing alpha rank file: {path}")
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        raise ValueError("alpha rank file must contain symbol column")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if "alpha_score" not in df.columns:
        df["alpha_score"] = 0.0
    return df


def quality_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = row.get(col, default)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def score_trade_candidate(row: pd.Series, alpha_scores: dict[str, float], args: argparse.Namespace) -> float:
    symbol = str(row.get("symbol", "")).upper()
    score = 0.0
    score += args.alpha_weight * alpha_scores.get(symbol, 0.0)

    quality = str(row.get("setup_quality", "")).upper()
    if quality == "A+":
        score += 35.0
    elif quality == "A":
        score += 25.0
    elif quality == "B":
        score += 12.0

    regime = str(row.get("market_regime", "")).lower()
    if regime == "strong":
        score += 18.0
    elif regime == "good":
        score += 10.0
    elif regime == "bad":
        score -= 20.0

    for col, weight in [
        ("score", 3.0),
        ("quality_score", 2.0),
        ("momentum_score", 2.0),
        ("breakout_score", 2.0),
        ("relative_volume", 4.0),
        ("rvol", 4.0),
        ("or_breakout_pct", 5.0),
        ("pnl_pct", 0.0),
    ]:
        if col in row.index and weight:
            score += weight * quality_float(row, col)

    for spread_col in ["spread_bps", "entry_spread_bps"]:
        if spread_col in row.index:
            spread = quality_float(row, spread_col, default=999.0)
            if spread <= 5:
                score += 8.0
            elif spread <= 10:
                score += 4.0
            elif spread > 25:
                score -= 20.0
            break

    price = None
    for price_col in ["entry_price", "price", "open"]:
        if price_col in row.index:
            price = quality_float(row, price_col, default=0.0)
            break
    if price is not None:
        if price < 5:
            score -= 100.0
        elif price < 10:
            score -= 5.0

    return float(score)


def build_daily_top_universe(trades: pd.DataFrame, alpha_rank: pd.DataFrame, wide_symbols: set[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = trades.copy()
    trades["symbol"] = trades["symbol"].astype(str).str.upper()
    trades = trades[trades["symbol"].isin(wide_symbols)].copy()
    trades["entry_date"] = trades["entry_dt"].dt.date.astype(str)

    alpha_scores = dict(zip(alpha_rank["symbol"], pd.to_numeric(alpha_rank["alpha_score"], errors="coerce").fillna(0.0)))
    trades["daily_rank_score"] = trades.apply(lambda row: score_trade_candidate(row, alpha_scores, args), axis=1)

    selected_rows = []
    daily_universe_rows = []

    for date, day in trades.groupby("entry_date", sort=True):
        symbol_scores = (
            day.groupby("symbol", as_index=False)["daily_rank_score"]
            .max()
            .sort_values("daily_rank_score", ascending=False)
        )
        top_symbols = set(symbol_scores.head(args.top_n)["symbol"].tolist())
        selected = day[day["symbol"].isin(top_symbols)].copy()
        selected_rows.append(selected)
        for rank, (_, r) in enumerate(symbol_scores.head(args.top_n).iterrows(), start=1):
            daily_universe_rows.append({
                "date": date,
                "rank": rank,
                "symbol": r["symbol"],
                "daily_rank_score": r["daily_rank_score"],
            })

    selected_trades = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(columns=trades.columns)
    daily_universe = pd.DataFrame(daily_universe_rows)
    return selected_trades, daily_universe


def apply_live_safe_expansion(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    mask = (
        (out["first_5m_high_pct"] >= args.min_first_5m_high_pct)
        & (out["first_15m_high_pct"] >= args.min_first_15m_high_pct)
        & (out["or_range_pct"] >= args.min_or_range_pct)
    )
    if args.min_entry_price is not None:
        mask &= out["entry_price"] >= args.min_entry_price
    if args.max_entry_price is not None:
        mask &= out["entry_price"] <= args.max_entry_price
    if args.exclude_bad_regime and "market_regime" in out.columns:
        mask &= out["market_regime"].astype(str).str.lower().ne("bad")
    if args.exclude_c_quality and "setup_quality" in out.columns:
        mask &= out["setup_quality"].astype(str).str.upper().ne("C")
    return out[mask].copy()


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


def run_case(label: str, trades: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted_df, rejected_df = simulate_portfolio(trades, args)
    if accepted_df.empty:
        summary = pd.DataFrame([{
            "strategy": label,
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
    return summary, costed, rejected_df


def add_selection_metadata(summary: pd.DataFrame, top_n: int, trades: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.insert(1, "daily_top_n", top_n)
    out.insert(2, "candidate_trades_after_daily_filter", len(trades))
    out.insert(3, "unique_symbols_in_daily_universe", trades["symbol"].nunique() if "symbol" in trades.columns and not trades.empty else 0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="v59 daily top-N universe simulation")
    parser.add_argument("--trades-csv", default=DEFAULT_TRADES)
    parser.add_argument("--alpha-rank-csv", default=DEFAULT_ALPHA_RANK)
    parser.add_argument("--wide-symbols", default=DEFAULT_WIDE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--alpha-weight", type=float, default=0.25)
    parser.add_argument("--apply-live-safe-expansion", action="store_true")
    parser.add_argument("--min-first-5m-high-pct", type=float, default=4.0)
    parser.add_argument("--min-first-15m-high-pct", type=float, default=6.5)
    parser.add_argument("--min-or-range-pct", type=float, default=5.0)
    parser.add_argument("--min-entry-price", type=float, default=None)
    parser.add_argument("--max-entry-price", type=float, default=None)
    parser.add_argument("--exclude-bad-regime", action="store_true")
    parser.add_argument("--exclude-c-quality", action="store_true")

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

    print("Experiment: v59 daily top-N universe")
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Alpha rank: {args.alpha_rank_csv}")
    print(f"Top N/day: {args.top_n}")
    print(f"Live-safe expansion: {args.apply_live_safe_expansion}")
    print(f"Position range: ${args.min_position_usd:.0f}-${args.max_position_usd:.0f}")

    trades = load_trades(args.trades_csv)
    trades["symbol"] = trades["symbol"].astype(str).str.upper()
    for col in ["entry_price", "first_5m_high_pct", "first_15m_high_pct", "or_range_pct", "pnl_pct"]:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")
    alpha = load_alpha_rank(args.alpha_rank_csv)
    wide = read_symbols(args.wide_symbols)

    wide_trades = trades[trades["symbol"].isin(wide)].copy()
    selected_trades, daily_universe = build_daily_top_universe(trades, alpha, wide, args)

    if args.apply_live_safe_expansion:
        wide_trades = apply_live_safe_expansion(wide_trades, args)
        selected_trades = apply_live_safe_expansion(selected_trades, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"top_{args.top_n}"
    if args.apply_live_safe_expansion:
        suffix += "_live_safe_expansion"

    baseline_summary, baseline_costed, baseline_rejected = run_case("v59_baseline_wide_all_candidates" + ("_live_safe_expansion" if args.apply_live_safe_expansion else ""), wide_trades, args)
    top_summary, top_costed, top_rejected = run_case(f"v59_daily_top_{args.top_n}" + ("_live_safe_expansion" if args.apply_live_safe_expansion else ""), selected_trades, args)

    baseline_summary = add_selection_metadata(baseline_summary, 0, wide_trades)
    top_summary = add_selection_metadata(top_summary, args.top_n, selected_trades)
    combined = pd.concat([baseline_summary, top_summary], ignore_index=True)

    daily_universe.to_csv(output_dir / f"v59_daily_{suffix}_universe_by_day.csv", index=False)
    selected_trades.to_csv(output_dir / f"v59_daily_{suffix}_selected_candidate_trades.csv", index=False)
    baseline_costed.to_csv(output_dir / f"v59_baseline_wide_all_candidates_{suffix}_costed.csv", index=False)
    top_costed.to_csv(output_dir / f"v59_daily_{suffix}_costed.csv", index=False)
    baseline_rejected.to_csv(output_dir / f"v59_baseline_wide_all_candidates_{suffix}_rejected.csv", index=False)
    top_rejected.to_csv(output_dir / f"v59_daily_{suffix}_rejected.csv", index=False)
    combined.to_csv(output_dir / f"v59_daily_{suffix}_comparison_summary.csv", index=False)

    print("\n=== v59 daily top-N comparison ===")
    print(combined.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not daily_universe.empty:
        print("\n=== Daily universe stats ===")
        counts = daily_universe.groupby("date")["symbol"].nunique()
        print(counts.describe().to_string(float_format=lambda x: f"{x:.2f}"))

    print(f"\nSaved summary: {output_dir / f'v59_daily_{suffix}_comparison_summary.csv'}")
    print("\nInterpretation hints:")
    print("- This is the live architecture test: broad universe -> daily top N -> optional live-safe expansion -> portfolio simulation.")
    print("- live-safe expansion uses first_5m_high_pct, first_15m_high_pct, or_range_pct, and optional price/regime/quality filters.")
    print("- It does not use time_to_high_minutes, so it avoids the earlier hindsight bias.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
