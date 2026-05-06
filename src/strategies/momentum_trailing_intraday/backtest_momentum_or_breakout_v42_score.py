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


def or_close_strength(extra: dict[str, float]) -> float:
    high = extra["or_high_pct"]
    low = extra["or_low_pct"]
    close = extra["or_close_pct"]
    rng = high - low
    if rng <= 0:
        return 0.0
    return max(0.0, min(1.0, (close - low) / rng))


def momentum_score(feats: dict[str, object], extra: dict[str, float], args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    """Score uses only information available before/at OR breakout.

    It intentionally does not use final intraday high or close result.
    The goal is to replace the research-only +10% future filter with a live-like
    quality score.
    """
    score = 0.0
    details: dict[str, float] = {}

    or_high = float(extra["or_high_pct"])
    or_range = float(extra["or_range_pct"])
    or_dv = float(extra["or_dollar_volume"])
    strength = or_close_strength(extra)
    gap = float(feats["gap_pct"]) if pd.notna(feats.get("gap_pct")) else 0.0
    price = float(feats.get("open", 0.0)) if "open" in feats else 0.0

    # OR momentum: needs visible demand, but not an absurd vertical one-candle pump.
    if or_high >= 2.0:
        score += 1.0
    if or_high >= 4.0:
        score += 1.0
    if or_high >= 7.0:
        score += 1.0
    if or_high > args.score_extreme_or_high_penalty_at:
        score -= 1.0

    # OR close quality: close near high means buyers held control during the opening range.
    if strength >= 0.55:
        score += 1.0
    if strength >= 0.75:
        score += 1.0
    if strength < args.min_or_close_strength:
        score -= 2.0

    # Dollar volume: proxy for tradability and institutional attention.
    if or_dv >= 500_000:
        score += 1.0
    if or_dv >= 1_000_000:
        score += 1.0
    if or_dv >= 5_000_000:
        score += 1.0

    # Avoid too-chaotic opening ranges. Some range is fine; extreme range often means chase risk.
    if 1.0 <= or_range <= 12.0:
        score += 1.0
    if or_range > args.score_wide_or_penalty_at:
        score -= 1.0

    # Gap context: big enough to attract attention, but not so large that the move is already exhausted.
    abs_gap = abs(gap)
    if 1.0 <= abs_gap <= 12.0:
        score += 1.0
    if abs_gap > 20.0:
        score -= 1.0

    # Price sanity: avoid the most fragile ultra-low-priced names.
    if price >= 5.0:
        score += 1.0
    if price < 3.0:
        score -= 1.0

    details["momentum_score"] = score
    details["or_close_strength"] = strength
    details["score_or_high_pct"] = or_high
    details["score_or_range_pct"] = or_range
    details["score_or_dollar_volume"] = or_dv
    details["score_gap_pct"] = gap
    details["score_open_price"] = price
    return score, details


def passes_v42_filter(symbol: str, day: pd.DataFrame, feats: dict[str, object], args: argparse.Namespace) -> tuple[bool, list[str], dict[str, float]]:
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
    if float(feats["first_5m_high_pct"]) > args.max_first_5m_high:
        reasons.append("first_5m_spike_too_large")
    if args.research_min_intraday_high is not None and float(feats["intraday_high_pct"]) < args.research_min_intraday_high:
        reasons.append("research_intraday_high_too_low")
    if float(feats["intraday_high_pct"]) > args.max_intraday_high:
        reasons.append("intraday_high_too_large")

    score, score_details = momentum_score(feats, extra, args)
    extra.update(score_details)
    if extra["or_close_strength"] < args.min_or_close_strength:
        reasons.append("or_close_strength_too_low")
    if score < args.min_momentum_score:
        reasons.append("momentum_score_too_low")

    return not reasons, reasons, extra


def minutes_from_open(day: pd.DataFrame, pos: int) -> float:
    start = pd.Timestamp(day.iloc[0]["datetime"])
    entry_time = pd.Timestamp(day.iloc[pos]["datetime"])
    return float((entry_time - start).total_seconds() / 60.0)


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

        raw = pd.DataFrame([feats])
        for col in ["gap_pct", "intraday_high_pct", "first_5m_high_pct", "first_15m_high_pct", "rows_1m"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")

        clean, _ = apply_quality_filter(
            raw,
            min_open_price=args.min_entry_price,
            max_open_price=args.max_entry_price,
            max_abs_gap=args.max_abs_gap,
            max_intraday_high=args.max_intraday_high,
            max_first_5m_high=args.max_first_5m_high,
            min_dollar_volume=0.0,
            exclude_weird_symbols=args.exclude_weird_symbols,
            excluded_symbols=set() if args.include_leveraged_etfs else DEFAULT_EXCLUDED_SYMBOLS,
        )
        if clean.empty:
            add_reject(rejected, reject_counter, base_payload, "base_clean_filter")
            continue

        ok, reasons, extra = passes_v42_filter(symbol, day, feats, args)
        if not ok:
            add_reject(rejected, reject_counter, {**base_payload, **extra}, ";".join(reasons))
            continue

        entry = find_or_breakout_entry(day, args.opening_range_minutes)
        if entry is None:
            add_reject(rejected, reject_counter, {**base_payload, **extra}, "no_or_breakout")
            continue

        entry_pos, entry_price = entry
        entry_minutes = minutes_from_open(day, entry_pos)
        if args.max_entry_minutes_from_open is not None and entry_minutes > args.max_entry_minutes_from_open:
            add_reject(rejected, reject_counter, {**base_payload, **extra, "entry_minutes_from_open": entry_minutes}, "entry_too_late")
            continue

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
        common["entry_minutes_from_open"] = entry_minutes

        for label, cfg in variants.items():
            pnl, reason, exit_time, max_pnl, min_pnl = simulate_exit(post_entry, entry_price, cfg)
            rows.append(
                {
                    **common,
                    **extra,
                    "variant": label,
                    "exit_time": str(exit_time),
                    "exit_price": entry_price * (1.0 + pnl / 100.0),
                    "pnl_pct": pnl,
                    "max_pnl_pct": max_pnl,
                    "min_pnl_pct": min_pnl,
                    "reason": reason,
                }
            )

    trades = pd.DataFrame(rows)
    rejects = pd.DataFrame(rejected)
    if not trades.empty:
        trades = trades.sort_values(["variant", "session_date", "symbol", "entry_time"]).reset_index(drop=True)
    return trades, rejects, reject_counter, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="v42 scored momentum watchlist OR breakout backtest.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--opening-range-minutes", type=int, default=5, choices=[5, 15, 30])

    parser.add_argument("--min-entry-price", type=float, default=2.0)
    parser.add_argument("--max-entry-price", type=float, default=300.0)
    parser.add_argument("--max-abs-gap", type=float, default=30.0)
    parser.add_argument("--max-intraday-high", type=float, default=80.0)
    parser.add_argument("--max-first-5m-high", type=float, default=50.0)
    parser.add_argument("--min-or-high-pct", type=float, default=2.0)
    parser.add_argument("--max-or-high-pct", type=float, default=35.0)
    parser.add_argument("--max-or-range-pct", type=float, default=25.0)
    parser.add_argument("--min-or-volume", type=float, default=20_000)
    parser.add_argument("--min-or-dollar-volume", type=float, default=250_000)
    parser.add_argument("--research-min-intraday-high", type=float, default=None)
    parser.add_argument("--exclude-weird-symbols", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-leveraged-etfs", action="store_true")

    parser.add_argument("--min-momentum-score", type=float, default=7.0)
    parser.add_argument("--min-or-close-strength", type=float, default=0.55)
    parser.add_argument("--score-extreme-or-high-penalty-at", type=float, default=18.0)
    parser.add_argument("--score-wide-or-penalty-at", type=float, default=15.0)
    parser.add_argument("--max-entry-minutes-from-open", type=float, default=120.0)

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

    print("Experiment: v42 scored momentum watchlist OR breakout")
    print("Goal: reduce v41 false positives with live-like quality score, without fixed daily trade cap.")
    print(f"Data dir: {args.data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Entry: OR{args.opening_range_minutes} breakout")
    print("Live-like score filters:")
    print(f"- min momentum score >= {args.min_momentum_score:.2f}")
    print(f"- min OR close strength >= {args.min_or_close_strength:.2f}")
    print(f"- max entry minutes from open <= {args.max_entry_minutes_from_open}")
    print(f"- OR dollar volume >= {args.min_or_dollar_volume:.0f}")
    print(f"- OR high pct: {args.min_or_high_pct:.2f}% to {args.max_or_high_pct:.2f}%")
    print(f"- OR range <= {args.max_or_range_pct:.2f}%")
    if args.research_min_intraday_high is not None:
        print(f"Research-only future filter: intraday_high >= {args.research_min_intraday_high:.2f}%")

    trades_df, rejects_df, reject_counter, stats = run_backtest_single_pass(args, variants)

    summaries = []
    for label in variants:
        subset = trades_df[trades_df["variant"] == label] if not trades_df.empty else pd.DataFrame()
        summaries.append(summarize(label, subset))
    summary = pd.DataFrame(summaries)

    research = "research" if args.research_min_intraday_high is not None else "livelike"
    suffix = (
        f"recent{args.recent_days}_or{args.opening_range_minutes}_{research}_"
        f"score{args.min_momentum_score:g}_minordv{int(args.min_or_dollar_volume)}"
    )
    out_trades = output_dir / f"v42_scored_momentum_or_breakout_trades_{suffix}.csv"
    out_summary = output_dir / f"v42_scored_momentum_or_breakout_summary_{suffix}.csv"
    out_rejects = output_dir / f"v42_scored_momentum_or_breakout_rejected_{suffix}.csv"
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
        print("\n=== Score buckets, best variant ===")
        tmp = best.copy()
        tmp["score_bin"] = pd.cut(tmp["momentum_score"], bins=[-999, 6, 7, 8, 9, 10, 999])
        print(tmp.groupby("score_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Liquidity buckets, best variant ===")
        tmp["or_dollar_volume_bin"] = pd.cut(tmp["or_dollar_volume"], bins=[0, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, float("inf")])
        print(tmp.groupby("or_dollar_volume_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Top symbols by total pnl, best variant only ===")
        print(f"Best variant: {best_name}")
        print(best.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).head(30).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Recent trades, best variant ===")
        cols = ["session_date", "symbol", "entry_time", "entry_minutes_from_open", "entry_price", "pnl_pct", "max_pnl_pct", "min_pnl_pct", "reason", "momentum_score", "or_close_strength", "or_high_pct", "or_dollar_volume", "intraday_high_pct", "gap_pct"]
        print(best[cols].tail(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if reject_counter:
        print("\n=== Top rejection reasons ===")
        for reason, count in reject_counter.most_common(20):
            print(f"{reason:110s} {count}")

    print("\nInterpretation hints:")
    print("- v42 is stricter than v41 and tries to keep only higher-quality momentum candidates.")
    print("- It does not impose a fixed daily trade cap; it uses score thresholds instead.")
    print("- If count is too low, lower --min-momentum-score or increase --max-entry-minutes-from-open.")
    print("- If avg/median remains weak, v43 should add premarket relative volume/catalyst data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
