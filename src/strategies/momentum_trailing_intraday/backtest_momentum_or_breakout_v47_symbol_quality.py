from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v46_trend_exit import (
    fixed_position_profit,
    make_baseline_variants,
    run_backtest_single_pass,
)
from src.analysis.big_momentum_available_1m_research import DEFAULT_OUTPUT_DIR
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v40 import summarize


@dataclass
class SymbolMemory:
    maxlen: int
    pnls: deque[float] = field(default_factory=deque)
    max_pnls: deque[float] = field(default_factory=deque)
    min_pnls: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.pnls = deque(maxlen=self.maxlen)
        self.max_pnls = deque(maxlen=self.maxlen)
        self.min_pnls = deque(maxlen=self.maxlen)

    @property
    def count(self) -> int:
        return len(self.pnls)

    def add(self, pnl: float, max_pnl: float, min_pnl: float) -> None:
        self.pnls.append(float(pnl))
        self.max_pnls.append(float(max_pnl))
        self.min_pnls.append(float(min_pnl))

    def stats(self) -> dict[str, float]:
        if not self.pnls:
            return {
                "symbol_prior_count": 0.0,
                "symbol_prior_avg_pnl": 0.0,
                "symbol_prior_median_pnl": 0.0,
                "symbol_prior_win_rate": 0.0,
                "symbol_prior_loss_rate": 0.0,
                "symbol_prior_stop_rate": 0.0,
                "symbol_prior_avg_max_pnl": 0.0,
                "symbol_prior_avg_min_pnl": 0.0,
                "symbol_quality_score": 0.0,
            }
        s = pd.Series(list(self.pnls), dtype="float64")
        max_s = pd.Series(list(self.max_pnls), dtype="float64")
        min_s = pd.Series(list(self.min_pnls), dtype="float64")
        win_rate = float((s > 0).mean() * 100.0)
        loss_rate = float((s < 0).mean() * 100.0)
        stop_rate = float((s <= -7.9).mean() * 100.0)
        avg = float(s.mean())
        med = float(s.median())
        avg_max = float(max_s.mean())
        avg_min = float(min_s.mean())

        score = 0.0
        if self.count >= 2:
            score += 0.75
        if self.count >= 3:
            score += 0.75
        if avg > 0.25:
            score += 1.0
        if avg > 1.0:
            score += 1.0
        if med > 0.25:
            score += 1.0
        if med > 1.0:
            score += 1.0
        if win_rate >= 55.0:
            score += 0.75
        if win_rate >= 65.0:
            score += 0.75
        if stop_rate >= 25.0:
            score -= 1.0
        if stop_rate >= 40.0:
            score -= 1.0
        if avg_min <= -5.0:
            score -= 0.75
        if avg_max >= 4.0:
            score += 0.5

        return {
            "symbol_prior_count": float(self.count),
            "symbol_prior_avg_pnl": avg,
            "symbol_prior_median_pnl": med,
            "symbol_prior_win_rate": win_rate,
            "symbol_prior_loss_rate": loss_rate,
            "symbol_prior_stop_rate": stop_rate,
            "symbol_prior_avg_max_pnl": avg_max,
            "symbol_prior_avg_min_pnl": avg_min,
            "symbol_quality_score": score,
        }


def load_or_generate_v46_trades(args: argparse.Namespace) -> pd.DataFrame:
    if args.trades_csv:
        path = Path(args.trades_csv)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    print("No --trades-csv provided; generating v46 trades first.")
    baseline_variants = make_baseline_variants(args)
    trades_df, _, _, _ = run_backtest_single_pass(args, baseline_variants)
    return trades_df


def apply_symbol_quality(
    trades: pd.DataFrame,
    source_variant: str,
    min_prior_trades: int,
    min_symbol_quality_score: float,
    window: int,
    allow_new_symbols: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), pd.DataFrame()
    required = {"variant", "symbol", "session_date", "entry_time", "pnl_pct", "max_pnl_pct", "min_pnl_pct"}
    missing = sorted(required - set(trades.columns))
    if missing:
        raise ValueError(f"Missing required trade columns: {missing}")

    src = trades[trades["variant"] == source_variant].copy()
    if src.empty:
        raise ValueError(f"No trades found for source variant: {source_variant}")

    src["_session_dt"] = pd.to_datetime(src["session_date"], errors="coerce")
    src["_entry_dt"] = pd.to_datetime(src["entry_time"], errors="coerce")
    src = src.sort_values(["_session_dt", "_entry_dt", "symbol"]).reset_index(drop=True)

    memories: dict[str, SymbolMemory] = defaultdict(lambda: SymbolMemory(maxlen=window))
    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []

    for _, row in src.iterrows():
        symbol = str(row["symbol"])
        mem = memories[symbol]
        st = mem.stats()
        prior_count = int(st["symbol_prior_count"])
        enough_history = prior_count >= min_prior_trades

        accept = True
        reason = "accepted"
        if not enough_history:
            if allow_new_symbols:
                reason = "accepted_new_symbol"
            else:
                accept = False
                reason = "not_enough_symbol_history"
        elif st["symbol_quality_score"] < min_symbol_quality_score:
            accept = False
            reason = "symbol_quality_too_low"

        enriched = {**row.to_dict(), **st, "symbol_quality_reason": reason}
        if accept:
            accepted_rows.append(enriched)
        else:
            rejected_rows.append(enriched)

        # Critical no-lookahead rule: update memory AFTER decision for current trade.
        mem.add(float(row["pnl_pct"]), float(row["max_pnl_pct"]), float(row["min_pnl_pct"]))

    accepted = pd.DataFrame(accepted_rows)
    rejected = pd.DataFrame(rejected_rows)
    for df in (accepted, rejected):
        if not df.empty:
            df.drop(columns=[c for c in ["_session_dt", "_entry_dt"] if c in df.columns], inplace=True)
    return accepted, rejected


def summarize_symbol_quality(label: str, df: pd.DataFrame, fixed_position_size: float) -> dict[str, object]:
    row = summarize(label, df)
    row["fixed_1000_profit_usd"] = fixed_position_profit(float(row.get("total_pnl", 0.0) or 0.0), fixed_position_size)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="v47 rolling symbol-quality layer on top of v46/v45 trades.")

    # v46 generation args, used when --trades-csv is omitted.
    parser.add_argument("--data-dir", default="data/1m")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--opening-range-minutes", type=int, default=5, choices=[5, 15, 30])
    parser.add_argument("--min-entry-price", type=float, default=2.0)
    parser.add_argument("--max-entry-price", type=float, default=300.0)
    parser.add_argument("--max-abs-gap", type=float, default=30.0)
    parser.add_argument("--max-first-5m-high", type=float, default=50.0)
    parser.add_argument("--min-or-high-pct", type=float, default=2.0)
    parser.add_argument("--max-or-high-pct", type=float, default=35.0)
    parser.add_argument("--max-or-range-pct", type=float, default=25.0)
    parser.add_argument("--min-or-volume", type=float, default=20_000)
    parser.add_argument("--min-or-dollar-volume", type=float, default=250_000)
    parser.add_argument("--exclude-weird-symbols", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-leveraged-etfs", action="store_true")
    parser.add_argument("--min-v45-score", type=float, default=7.0)
    parser.add_argument("--min-or-close-strength", type=float, default=0.45)
    parser.add_argument("--min-entry-candle-strength", type=float, default=0.35)
    parser.add_argument("--max-pre-entry-break-below-or-pct", type=float, default=1.50)
    parser.add_argument("--max-entry-extension-from-open-pct", type=float, default=18.0)
    parser.add_argument("--max-entry-minutes-from-open", type=float, default=90.0)
    parser.add_argument("--mid-open-penalty", type=float, default=0.5)
    parser.add_argument("--stop-loss", type=float, default=8.0)
    parser.add_argument("--take-profit", type=float, default=25.0)
    parser.add_argument("--trailing-activation", type=float, default=2.5)
    parser.add_argument("--trailing-stop", type=float, default=1.5)
    parser.add_argument("--wide-trailing-activation", type=float, default=3.0)
    parser.add_argument("--wide-trailing-stop", type=float, default=3.0)
    parser.add_argument("--breakeven-activation", type=float, default=2.0)
    parser.add_argument("--breakeven-lock-pct", type=float, default=0.1)
    parser.add_argument("--trend-activation", type=float, default=5.0)
    parser.add_argument("--trend-trailing-stop", type=float, default=4.0)
    parser.add_argument("--runner-activation", type=float, default=6.0)
    parser.add_argument("--runner-trailing-stop", type=float, default=5.0)
    parser.add_argument("--stage-profit", type=float, default=4.0)
    parser.add_argument("--stage-fraction", type=float, default=0.5)
    parser.add_argument("--stage-stop-lock-pct", type=float, default=0.5)
    parser.add_argument("--ratchet-trail-6", type=float, default=3.0)
    parser.add_argument("--ratchet-trail-10", type=float, default=4.5)
    parser.add_argument("--max-hold-minutes", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10_000)

    # v47 args.
    parser.add_argument("--trades-csv", default="data/backtests/v46_trend_exit_momentum_or_breakout_trades_recent90_or5_score7_trendexit.csv")
    parser.add_argument("--source-variant", default="wide_trail")
    parser.add_argument("--symbol-window", type=int, default=20)
    parser.add_argument("--min-prior-trades", type=int, default=3)
    parser.add_argument("--min-symbol-quality-score", type=float, default=1.5)
    parser.add_argument("--allow-new-symbols", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-position-size", type=float, default=1000.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment: v47 rolling symbol quality")
    print("Base: v45 entry + v46 selected exit variant")
    print("No lookahead: symbol memory is updated only AFTER each trade decision.")
    print(f"Trades CSV: {args.trades_csv or '[generated from v46]'}")
    print(f"Source variant: {args.source_variant}")
    print(f"Symbol window: {args.symbol_window}")
    print(f"Min prior trades: {args.min_prior_trades}")
    print(f"Min symbol quality score: {args.min_symbol_quality_score:.2f}")
    print(f"Allow new symbols: {args.allow_new_symbols}")

    trades = load_or_generate_v46_trades(args)
    base = trades[trades["variant"] == args.source_variant].copy()
    accepted, rejected = apply_symbol_quality(
        trades=trades,
        source_variant=args.source_variant,
        min_prior_trades=args.min_prior_trades,
        min_symbol_quality_score=args.min_symbol_quality_score,
        window=args.symbol_window,
        allow_new_symbols=args.allow_new_symbols,
    )

    summary_rows = [
        summarize_symbol_quality(f"baseline_{args.source_variant}", base, args.fixed_position_size),
        summarize_symbol_quality(f"v47_symbol_quality_{args.source_variant}", accepted, args.fixed_position_size),
    ]
    summary = pd.DataFrame(summary_rows)

    suffix = (
        f"recent{args.recent_days}_or{args.opening_range_minutes}_{args.source_variant}_"
        f"w{args.symbol_window}_minprior{args.min_prior_trades}_score{args.min_symbol_quality_score:g}"
    )
    out_trades = output_dir / f"v47_symbol_quality_trades_{suffix}.csv"
    out_rejected = output_dir / f"v47_symbol_quality_rejected_{suffix}.csv"
    out_summary = output_dir / f"v47_symbol_quality_summary_{suffix}.csv"
    accepted.to_csv(out_trades, index=False)
    rejected.to_csv(out_rejected, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\nSaved accepted trades CSV: {out_trades}")
    print(f"Saved rejected trades CSV: {out_rejected}")
    print(f"Saved summary CSV: {out_summary}")

    print("\n=== Variant comparison ===")
    cols = ["strategy", "count", "active_days", "symbols", "win_rate", "avg_pnl", "median_pnl", "total_pnl", "fixed_1000_profit_usd", "avg_win", "avg_loss", "avg_max_pnl", "avg_min_pnl", "max_loss", "max_win"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== v47 scan stats ===")
    print(f"baseline trades: {len(base)}")
    print(f"accepted trades: {len(accepted)}")
    print(f"rejected trades: {len(rejected)}")
    if not rejected.empty:
        print("\n=== Rejection reasons ===")
        print(rejected["symbol_quality_reason"].value_counts().to_string())

    if not accepted.empty:
        print("\n=== Symbol quality score buckets ===")
        tmp = accepted.copy()
        tmp["symbol_quality_bin"] = pd.cut(tmp["symbol_quality_score"], bins=[-999, 0, 1, 2, 3, 5, 999])
        print(tmp.groupby("symbol_quality_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Top symbols after v47 ===")
        print(accepted.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).head(30).to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Worst symbols still accepted by v47 ===")
        print(accepted.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=True).head(30).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- Good outcome: v47 keeps enough trades while improving avg/median and fixed-position profit.")
    print("- If v47 rejects too much, lower --min-prior-trades or --min-symbol-quality-score.")
    print("- If v47 underperforms baseline, symbol repeatability needs richer features or longer history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
