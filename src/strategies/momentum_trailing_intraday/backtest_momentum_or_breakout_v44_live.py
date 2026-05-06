from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.analysis.big_momentum_available_1m_clean_research import DEFAULT_EXCLUDED_SYMBOLS, apply_quality_filter
from src.analysis.big_momentum_available_1m_research import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v40 import (
    ExitConfig,
    TradeResult,
    day_features,
    find_or_breakout_entry,
    iter_days,
    simulate_exit,
    summarize,
)
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v41_tradable import (
    looks_like_weird_security,
    opening_range_stats,
    session_liquidity,
)
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v42_score import (
    momentum_score,
    or_close_strength,
)


def make_variants(args: argparse.Namespace) -> dict[str, ExitConfig]:
    return {
        "close_exit": ExitConfig(args.stop_loss, None, None, None, args.max_hold_minutes, True, None, 0.0),
        "trail_only": ExitConfig(args.stop_loss, None, args.trailing_activation, args.trailing_stop, args.max_hold_minutes, True, None, 0.0),
        "tp_trail": ExitConfig(args.stop_loss, args.take_profit, args.trailing_activation, args.trailing_stop, args.max_hold_minutes, True, None, 0.0),
        "staged": ExitConfig(args.stop_loss, args.take_profit, args.trailing_activation, args.trailing_stop, args.max_hold_minutes, True, args.staged_take_profit, args.staged_fraction),
    }


def add_reject(rejected: list[dict[str, object]], counter: Counter[str], payload: dict[str, object], reason: str) -> None:
    counter[reason] += 1
    if len(rejected) < 200_000:
        rejected.append({**payload, "reject_reason": reason})


def minutes_from_open(day: pd.DataFrame, pos: int) -> float:
    start = pd.Timestamp(day.iloc[0]["datetime"])
    t = pd.Timestamp(day.iloc[pos]["datetime"])
    return float((t - start).total_seconds() / 60.0)


def passes_live_filter(symbol: str, day: pd.DataFrame, feats: dict[str, object], args: argparse.Namespace) -> tuple[bool, list[str], dict[str, float]]:
    """Strict no-lookahead filter.

    This function deliberately does NOT use:
    - intraday_high_pct
    - open_to_close_pct
    - time_to_high_minutes
    - final daily return

    It only uses information available before/around the candidate entry:
    symbol/static filters, opening range stats, early volume/liquidity, gap from previous close.
    """
    reasons: list[str] = []
    open_price = float(day.iloc[0]["open"])
    extra = {**opening_range_stats(day, args.opening_range_minutes), **session_liquidity(day)}

    if args.exclude_weird_symbols and looks_like_weird_security(symbol):
        reasons.append("weird_security_suffix")
    if not args.include_leveraged_etfs and symbol.upper() in DEFAULT_EXCLUDED_SYMBOLS:
        reasons.append("leveraged_or_excluded_etf")
    if open_price < args.min_entry_price:
        reasons.append("entry_price_too_low")
    if args.max_entry_price is not None and open_price > args.max_entry_price:
        reasons.append("entry_price_too_high")
    if extra["or_dollar_volume"] < args.min_or_dollar_volume:
        reasons.append("opening_range_dollar_volume_too_low")
    if extra["or_volume"] < args.min_or_volume:
        reasons.append("opening_range_volume_too_low")
    if extra["or_range_pct"] > args.max_or_range_pct:
        reasons.append("opening_range_too_wide")
    if extra["or_high_pct"] < args.min_or_high_pct:
        reasons.append("opening_range_momentum_too_low")
    if extra["or_high_pct"] > args.max_or_high_pct:
        reasons.append("opening_range_momentum_too_extreme")

    gap = feats.get("gap_pct")
    if pd.notna(gap) and abs(float(gap)) > args.max_abs_gap:
        reasons.append("gap_abs_too_large")

    score, score_details = momentum_score(feats, extra, args)
    extra.update(score_details)
    extra["or_close_strength"] = or_close_strength(extra)

    if extra["or_close_strength"] < args.min_or_close_strength:
        reasons.append("or_close_strength_too_low")
    if score < args.min_momentum_score:
        reasons.append("momentum_score_too_low")

    return not reasons, reasons, extra


def find_continuation_entry(
    day: pd.DataFrame,
    opening_range_minutes: int,
    max_entry_minutes_from_open: float,
    hold_minutes: int,
    max_break_below_or_pct: float,
    min_second_push_pct: float,
    min_pullback_pct: float,
    max_pullback_pct: float,
) -> tuple[int, float, dict[str, float]] | None:
    if len(day) <= opening_range_minutes + hold_minutes + 2:
        return None

    opening = day.iloc[:opening_range_minutes]
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())
    open_price = float(day.iloc[0]["open"])

    initial = find_or_breakout_entry(day, opening_range_minutes)
    if initial is None:
        return None
    breakout_pos, breakout_price = initial
    if minutes_from_open(day, breakout_pos) > max_entry_minutes_from_open:
        return None

    hold_start = breakout_pos
    hold_end = min(len(day), breakout_pos + hold_minutes)
    hold_slice = day.iloc[hold_start:hold_end]
    if hold_slice.empty:
        return None

    min_allowed = or_high * (1.0 - max_break_below_or_pct / 100.0)
    if float(hold_slice["low"].min()) < min_allowed:
        return None

    breakout_high = float(hold_slice["high"].max())
    pullback_low = float(hold_slice["low"].min())
    pullback_pct = (breakout_high / pullback_low - 1.0) * 100.0 if pullback_low > 0 else 999.0
    if pullback_pct < min_pullback_pct or pullback_pct > max_pullback_pct:
        return None

    continuation_trigger = breakout_high * (1.0 + min_second_push_pct / 100.0)
    for pos in range(hold_end, len(day)):
        mins = minutes_from_open(day, pos)
        if mins > max_entry_minutes_from_open:
            return None
        row = day.iloc[pos]
        if float(row["low"]) < min_allowed:
            return None
        if float(row["high"]) >= continuation_trigger:
            entry_price = continuation_trigger
            return pos, entry_price, {
                "or_high": or_high,
                "or_low": or_low,
                "initial_breakout_pos": float(breakout_pos),
                "initial_breakout_minutes": minutes_from_open(day, breakout_pos),
                "initial_breakout_price": breakout_price,
                "continuation_entry_minutes": mins,
                "continuation_trigger": continuation_trigger,
                "hold_minutes": float(hold_minutes),
                "hold_low_vs_or_high_pct": (float(hold_slice["low"].min()) / or_high - 1.0) * 100.0,
                "breakout_hold_high_pct_from_open": (breakout_high / open_price - 1.0) * 100.0,
                "controlled_pullback_pct": pullback_pct,
                "second_push_pct": min_second_push_pct,
            }
    return None


def run_backtest_single_pass(args: argparse.Namespace, variants: dict[str, ExitConfig]) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str], Counter[str]]:
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    reject_counter: Counter[str] = Counter()
    stats: Counter[str] = Counter()

    for symbol, day, prev_close in iter_days(Path(args.data_dir), args.recent_days):
        stats["sessions_seen"] += 1
        if stats["sessions_seen"] % args.progress_every == 0:
            print(
                "Progress: "
                f"sessions={stats['sessions_seen']}, "
                f"accepted_entries={stats['accepted_entries']}, "
                f"rejects={sum(reject_counter.values())}, "
                f"trades={len(rows)}",
                flush=True,
            )

        if len(day) < args.min_rows:
            stats["sessions_short_rows"] += 1
            continue

        feats = day_features(symbol, day, prev_close)
        base_payload = {"symbol": symbol, **feats}

        # Base quality filter called in live-safe mode: no intraday high filter, no first_5m future issue beyond OR5.
        raw = pd.DataFrame([feats])
        for col in ["gap_pct", "first_5m_high_pct", "first_15m_high_pct", "rows_1m"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")

        clean, _ = apply_quality_filter(
            raw,
            min_open_price=args.min_entry_price,
            max_open_price=args.max_entry_price,
            max_abs_gap=args.max_abs_gap,
            max_intraday_high=10_000.0,  # intentionally disabled: future-aware filter
            max_first_5m_high=args.max_first_5m_high,
            min_dollar_volume=0.0,
            exclude_weird_symbols=args.exclude_weird_symbols,
            excluded_symbols=set() if args.include_leveraged_etfs else DEFAULT_EXCLUDED_SYMBOLS,
        )
        if clean.empty:
            add_reject(rejected, reject_counter, base_payload, "base_clean_filter")
            continue

        ok, reasons, extra = passes_live_filter(symbol, day, feats, args)
        if not ok:
            add_reject(rejected, reject_counter, {**base_payload, **extra}, ";".join(reasons))
            continue

        continuation = find_continuation_entry(
            day=day,
            opening_range_minutes=args.opening_range_minutes,
            max_entry_minutes_from_open=args.max_entry_minutes_from_open,
            hold_minutes=args.hold_minutes,
            max_break_below_or_pct=args.max_break_below_or_pct,
            min_second_push_pct=args.min_second_push_pct,
            min_pullback_pct=args.min_pullback_pct,
            max_pullback_pct=args.max_pullback_pct,
        )
        if continuation is None:
            add_reject(rejected, reject_counter, {**base_payload, **extra}, "no_continuation_confirmation")
            continue

        entry_pos, entry_price, continuation_info = continuation
        post_entry = day.iloc[entry_pos:]
        if post_entry.empty:
            add_reject(rejected, reject_counter, {**base_payload, **extra}, "empty_post_entry")
            continue

        stats["accepted_entries"] += 1
        common = TradeResult(
            symbol=symbol,
            session_date=str(feats["session_date"]),
            entry_time=str(pd.Timestamp(post_entry.iloc[0]["datetime"])),
            entry_price=entry_price,
            exit_time="",
            exit_price=entry_price,
            pnl_pct=0.0,
            max_pnl_pct=0.0,
            min_pnl_pct=0.0,
            reason="",
            intraday_high_pct=float(feats["intraday_high_pct"]),
            gap_pct=float(feats["gap_pct"]) if pd.notna(feats["gap_pct"]) else None,
            first_5m_high_pct=float(feats["first_5m_high_pct"]),
            first_15m_high_pct=float(feats["first_15m_high_pct"]),
            time_to_high_minutes=float(feats["time_to_high_minutes"]),
            opening_range_minutes=args.opening_range_minutes,
            open_to_close_pct=float(feats["open_to_close_pct"]),
        ).__dict__
        common["entry_minutes_from_open"] = minutes_from_open(day, entry_pos)

        for label, cfg in variants.items():
            pnl, reason, exit_time, max_pnl, min_pnl = simulate_exit(post_entry, entry_price, cfg)
            rows.append({
                **common,
                **extra,
                **continuation_info,
                "variant": label,
                "exit_time": str(exit_time),
                "exit_price": entry_price * (1.0 + pnl / 100.0),
                "pnl_pct": pnl,
                "max_pnl_pct": max_pnl,
                "min_pnl_pct": min_pnl,
                "reason": reason,
            })

    trades = pd.DataFrame(rows)
    rejects = pd.DataFrame(rejected)
    if not trades.empty:
        trades = trades.sort_values(["variant", "session_date", "symbol", "entry_time"]).reset_index(drop=True)
    return trades, rejects, reject_counter, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="v44 live-like OR breakout continuation confirmation backtest.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
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

    parser.add_argument("--min-momentum-score", type=float, default=6.0)
    parser.add_argument("--min-or-close-strength", type=float, default=0.55)
    parser.add_argument("--score-extreme-or-high-penalty-at", type=float, default=18.0)
    parser.add_argument("--score-wide-or-penalty-at", type=float, default=15.0)
    parser.add_argument("--max-entry-minutes-from-open", type=float, default=60.0)

    parser.add_argument("--hold-minutes", type=int, default=5)
    parser.add_argument("--max-break-below-or-pct", type=float, default=0.75)
    parser.add_argument("--min-second-push-pct", type=float, default=0.25)
    parser.add_argument("--min-pullback-pct", type=float, default=0.25)
    parser.add_argument("--max-pullback-pct", type=float, default=8.0)

    parser.add_argument("--stop-loss", type=float, default=8.0)
    parser.add_argument("--take-profit", type=float, default=25.0)
    parser.add_argument("--trailing-activation", type=float, default=2.5)
    parser.add_argument("--trailing-stop", type=float, default=1.5)
    parser.add_argument("--max-hold-minutes", type=int, default=None)
    parser.add_argument("--staged-take-profit", type=float, default=10.0)
    parser.add_argument("--staged-fraction", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = make_variants(args)

    print("Experiment: v44 LIVE-LIKE OR breakout continuation confirmation")
    print("No future filters: intraday_high/open_to_close/time_to_high are diagnostics only, not entry filters.")
    print(f"Data dir: {args.data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Entry: OR{args.opening_range_minutes} breakout + continuation")
    print("Continuation filters:")
    print(f"- hold_minutes={args.hold_minutes}")
    print(f"- max_break_below_or_pct={args.max_break_below_or_pct:.2f}%")
    print(f"- min_second_push_pct={args.min_second_push_pct:.2f}%")
    print(f"- pullback_pct between {args.min_pullback_pct:.2f}% and {args.max_pullback_pct:.2f}%")
    print(f"- max_entry_minutes_from_open={args.max_entry_minutes_from_open}")

    trades_df, rejects_df, reject_counter, stats = run_backtest_single_pass(args, variants)

    summaries = []
    for label in variants:
        subset = trades_df[trades_df["variant"] == label] if not trades_df.empty else pd.DataFrame()
        summaries.append(summarize(label, subset))
    summary = pd.DataFrame(summaries)

    suffix = (
        f"recent{args.recent_days}_or{args.opening_range_minutes}_livelike_"
        f"hold{args.hold_minutes}_push{args.min_second_push_pct:g}_pb{args.min_pullback_pct:g}-{args.max_pullback_pct:g}"
    )
    out_trades = output_dir / f"v44_live_continuation_momentum_or_breakout_trades_{suffix}.csv"
    out_summary = output_dir / f"v44_live_continuation_momentum_or_breakout_summary_{suffix}.csv"
    out_rejects = output_dir / f"v44_live_continuation_momentum_or_breakout_rejected_{suffix}.csv"
    trades_df.to_csv(out_trades, index=False)
    summary.to_csv(out_summary, index=False)
    rejects_df.to_csv(out_rejects, index=False)

    print(f"\nSaved trades CSV: {out_trades}")
    print(f"Saved summary CSV: {out_summary}")
    print(f"Saved rejected CSV: {out_rejects}")

    print("\n=== Scan stats ===")
    print(f"sessions_seen: {stats['sessions_seen']}")
    print(f"sessions_short_rows: {stats['sessions_short_rows']}")
    print(f"accepted_entries: {stats['accepted_entries']}")
    print(f"trade_rows: {len(trades_df)}")
    print(f"rejected_sessions: {sum(reject_counter.values())}")

    print("\n=== Variant comparison ===")
    if not summary.empty:
        cols = ["strategy", "count", "active_days", "symbols", "win_rate", "avg_pnl", "median_pnl", "total_pnl", "avg_win", "avg_loss", "avg_max_pnl", "avg_min_pnl", "max_loss", "max_win"]
        print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not trades_df.empty:
        print("\n=== Exit reasons by variant ===")
        print(pd.crosstab(trades_df["variant"], trades_df["reason"]).to_string())
        best_name = summary.sort_values("avg_pnl", ascending=False).iloc[0]["strategy"]
        best = trades_df[trades_df["variant"] == best_name]
        print("\n=== Continuation entry timing buckets, best variant ===")
        tmp = best.copy()
        tmp["entry_minutes_bin"] = pd.cut(tmp["entry_minutes_from_open"], bins=[-1, 5, 10, 15, 30, 60, 120, 999])
        print(tmp.groupby("entry_minutes_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Controlled pullback buckets, best variant ===")
        tmp["controlled_pullback_bin"] = pd.cut(tmp["controlled_pullback_pct"], bins=[0, 0.5, 1, 2, 4, 8, 999])
        print(tmp.groupby("controlled_pullback_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Top symbols by total pnl, best variant only ===")
        print(f"Best variant: {best_name}")
        print(best.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).head(30).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Recent trades, best variant ===")
        cols = ["session_date", "symbol", "entry_time", "entry_minutes_from_open", "entry_price", "pnl_pct", "max_pnl_pct", "min_pnl_pct", "reason", "controlled_pullback_pct", "hold_low_vs_or_high_pct", "or_high_pct", "intraday_high_pct", "gap_pct"]
        print(best[cols].tail(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if reject_counter:
        print("\n=== Top rejection reasons ===")
        for reason, count in reject_counter.most_common(20):
            print(f"{reason:110s} {count}")

    print("\nInterpretation hints:")
    print("- This is the strict live-like v44; no future-aware filters should appear in rejection reasons.")
    print("- If median is still weak, next edge must come from premarket/catalyst/relative-volume filtering.")
    print("- If count is too low, lower --min-pullback-pct or increase --max-entry-minutes-from-open.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
