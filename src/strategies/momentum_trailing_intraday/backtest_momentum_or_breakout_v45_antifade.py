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
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v42_score import or_close_strength


def make_variants(args: argparse.Namespace) -> dict[str, ExitConfig]:
    return {
        "trail_only": ExitConfig(args.stop_loss, None, args.trailing_activation, args.trailing_stop, args.max_hold_minutes, True, None, 0.0),
        "close_exit": ExitConfig(args.stop_loss, None, None, None, args.max_hold_minutes, True, None, 0.0),
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


def candle_body_strength(row: pd.Series) -> float:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    if high <= low:
        return 0.5
    return max(0.0, min(1.0, (close - low) / (high - low)))


def first_n_stats(day: pd.DataFrame, n: int) -> dict[str, float]:
    sl = day.iloc[: max(1, min(n, len(day)))]
    if "volume" not in sl.columns:
        return {f"first_{n}m_volume": 0.0, f"first_{n}m_dollar_volume": 0.0}
    return {
        f"first_{n}m_volume": float(sl["volume"].sum()),
        f"first_{n}m_dollar_volume": float((sl["close"] * sl["volume"]).sum()),
    }


def live_antifade_score(day: pd.DataFrame, feats: dict[str, object], extra: dict[str, float], entry_pos: int, args: argparse.Namespace) -> tuple[float, dict[str, float], list[str]]:
    """Live-like anti-fade score.

    Uses only information available at/just before the entry candle. It deliberately
    avoids final intraday high, open-to-close, and time-to-high as filters.
    """
    reasons: list[str] = []
    score = 0.0
    entry_row = day.iloc[entry_pos]
    open_price = float(day.iloc[0]["open"])
    entry_price = float(entry_row["close"])
    entry_minutes = minutes_from_open(day, entry_pos)
    or_strength = or_close_strength(extra)
    or_high_pct = float(extra["or_high_pct"])
    or_range_pct = float(extra["or_range_pct"])
    or_dv = float(extra["or_dollar_volume"])
    gap = float(feats["gap_pct"]) if pd.notna(feats.get("gap_pct")) else 0.0

    # Price/location strength.
    if or_high_pct >= 2.0:
        score += 1.0
    if or_high_pct >= 4.0:
        score += 1.0
    if 8.0 <= or_high_pct <= 12.0:
        score += 1.0  # observed good v41/v43 bucket
    if or_high_pct > 18.0:
        score -= 1.0

    # OR close should not be a heavy rejection candle.
    if or_strength >= 0.55:
        score += 1.0
    if or_strength >= 0.75:
        score += 1.0
    if or_strength < args.min_or_close_strength:
        reasons.append("or_close_strength_too_low")

    # Tradability / attention.
    if or_dv >= 250_000:
        score += 1.0
    if or_dv >= 1_000_000:
        score += 1.0
    if or_dv >= 5_000_000:
        score += 1.0

    # Avoid ultra-dead or ultra-chaotic OR ranges.
    if 1.0 <= or_range_pct <= 25.0:
        score += 1.0
    if 8.0 <= or_range_pct <= 25.0:
        score += 0.5  # v41 trail-only did not hate bigger OR range
    if or_range_pct > args.max_or_range_pct:
        reasons.append("opening_range_too_wide")

    # Gap is context, not a hard edge. Penalize extreme exhaustion only.
    if -10.0 <= gap <= 10.0:
        score += 0.5
    if abs(gap) > args.max_abs_gap:
        reasons.append("gap_abs_too_large")

    # Entry timing: v43 found 10-15m can be weak, but do not hard reject by default.
    if 5.0 <= entry_minutes <= 10.0:
        score += 0.75
    elif 15.0 <= entry_minutes <= 30.0:
        score += 1.0
    elif 10.0 < entry_minutes < 15.0:
        score -= args.mid_open_penalty
    elif entry_minutes > args.max_entry_minutes_from_open:
        reasons.append("entry_too_late")

    # Anti-fade: entry candle should not close near low after breakout.
    entry_body_strength = candle_body_strength(entry_row)
    if entry_body_strength >= 0.55:
        score += 0.75
    if entry_body_strength >= 0.75:
        score += 0.75
    if entry_body_strength < args.min_entry_candle_strength:
        reasons.append("entry_candle_rejection")

    # Soft anti-fade: after opening range, price should not have already lost OR high badly before entry.
    opening = day.iloc[: args.opening_range_minutes]
    or_high = float(opening["high"].max())
    pre_entry = day.iloc[args.opening_range_minutes : entry_pos + 1]
    if not pre_entry.empty:
        max_break_below = (float(pre_entry["low"].min()) / or_high - 1.0) * 100.0
    else:
        max_break_below = 0.0
    if max_break_below >= -0.50:
        score += 1.0
    elif max_break_below < -args.max_pre_entry_break_below_or_pct:
        reasons.append("lost_or_high_before_entry")

    # Avoid immediate vertical chase from open.
    entry_from_open_pct = (entry_price / open_price - 1.0) * 100.0 if open_price > 0 else 0.0
    if entry_from_open_pct > args.max_entry_extension_from_open_pct:
        reasons.append("entry_extension_too_large")
    elif entry_from_open_pct <= 12.0:
        score += 0.5

    details = {
        "v45_score": score,
        "or_close_strength": or_strength,
        "entry_minutes_from_open": entry_minutes,
        "entry_candle_strength": entry_body_strength,
        "pre_entry_break_below_or_pct": max_break_below,
        "entry_from_open_pct": entry_from_open_pct,
    }
    return score, details, reasons


def passes_live_prefilter(symbol: str, day: pd.DataFrame, feats: dict[str, object], args: argparse.Namespace) -> tuple[bool, list[str], dict[str, float]]:
    reasons: list[str] = []
    open_price = float(day.iloc[0]["open"])
    extra = {
        **opening_range_stats(day, args.opening_range_minutes),
        **session_liquidity(day),
        **first_n_stats(day, 15),
    }

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
    if extra["or_high_pct"] < args.min_or_high_pct:
        reasons.append("opening_range_momentum_too_low")
    if extra["or_high_pct"] > args.max_or_high_pct:
        reasons.append("opening_range_momentum_too_extreme")

    gap = feats.get("gap_pct")
    if pd.notna(gap) and abs(float(gap)) > args.max_abs_gap:
        reasons.append("gap_abs_too_large")

    return not reasons, reasons, extra


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

        # Live-safe base quality: disable final intraday high as an entry filter.
        raw = pd.DataFrame([feats])
        for col in ["gap_pct", "first_5m_high_pct", "first_15m_high_pct", "rows_1m"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        clean, _ = apply_quality_filter(
            raw,
            min_open_price=args.min_entry_price,
            max_open_price=args.max_entry_price,
            max_abs_gap=args.max_abs_gap,
            max_intraday_high=10_000.0,  # disabled: future-aware
            max_first_5m_high=args.max_first_5m_high,
            min_dollar_volume=0.0,
            exclude_weird_symbols=args.exclude_weird_symbols,
            excluded_symbols=set() if args.include_leveraged_etfs else DEFAULT_EXCLUDED_SYMBOLS,
        )
        if clean.empty:
            add_reject(rejected, reject_counter, base_payload, "base_clean_filter")
            continue

        ok, reasons, extra = passes_live_prefilter(symbol, day, feats, args)
        if not ok:
            add_reject(rejected, reject_counter, {**base_payload, **extra}, ";".join(reasons))
            continue

        entry = find_or_breakout_entry(day, args.opening_range_minutes)
        if entry is None:
            add_reject(rejected, reject_counter, {**base_payload, **extra}, "no_or_breakout")
            continue

        entry_pos, entry_price = entry
        score, score_details, score_reasons = live_antifade_score(day, feats, extra, entry_pos, args)
        extra.update(score_details)
        if score < args.min_v45_score:
            score_reasons.append("v45_score_too_low")
        if score_reasons:
            add_reject(rejected, reject_counter, {**base_payload, **extra}, ";".join(score_reasons))
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

        for label, cfg in variants.items():
            pnl, reason, exit_time, max_pnl, min_pnl = simulate_exit(post_entry, entry_price, cfg)
            rows.append({
                **common,
                **extra,
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
    parser = argparse.ArgumentParser(description="v45 anti-fade live-like OR breakout momentum backtest.")
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
    parser.add_argument("--max-hold-minutes", type=int, default=None)
    parser.add_argument("--staged-take-profit", type=float, default=10.0)
    parser.add_argument("--staged-fraction", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = make_variants(args)

    print("Experiment: v45 anti-fade momentum OR breakout")
    print("Base: v41-style broad OR breakout, plus live-like anti-fade quality score.")
    print("No future filters: intraday_high/open_to_close/time_to_high are diagnostics only.")
    print(f"Data dir: {args.data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Entry: OR{args.opening_range_minutes} breakout")
    print("Filters:")
    print(f"- min_v45_score={args.min_v45_score:.2f}")
    print(f"- min_or_close_strength={args.min_or_close_strength:.2f}")
    print(f"- min_entry_candle_strength={args.min_entry_candle_strength:.2f}")
    print(f"- max_pre_entry_break_below_or_pct={args.max_pre_entry_break_below_or_pct:.2f}%")
    print(f"- max_entry_extension_from_open_pct={args.max_entry_extension_from_open_pct:.2f}%")
    print(f"- max_entry_minutes_from_open={args.max_entry_minutes_from_open:.2f}")

    trades_df, rejects_df, reject_counter, stats = run_backtest_single_pass(args, variants)

    summaries = []
    for label in variants:
        subset = trades_df[trades_df["variant"] == label] if not trades_df.empty else pd.DataFrame()
        summaries.append(summarize(label, subset))
    summary = pd.DataFrame(summaries)

    suffix = f"recent{args.recent_days}_or{args.opening_range_minutes}_score{args.min_v45_score:g}_minordv{int(args.min_or_dollar_volume)}"
    out_trades = output_dir / f"v45_antifade_momentum_or_breakout_trades_{suffix}.csv"
    out_summary = output_dir / f"v45_antifade_momentum_or_breakout_summary_{suffix}.csv"
    out_rejects = output_dir / f"v45_antifade_momentum_or_breakout_rejected_{suffix}.csv"
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
        best_name = summary.sort_values("median_pnl", ascending=False).iloc[0]["strategy"]
        best = trades_df[trades_df["variant"] == best_name]
        print("\n=== V45 score buckets, best median variant ===")
        tmp = best.copy()
        tmp["v45_score_bin"] = pd.cut(tmp["v45_score"], bins=[-999, 6, 7, 8, 9, 10, 12, 999])
        print(tmp.groupby("v45_score_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Entry timing buckets, best median variant ===")
        tmp["entry_minutes_bin"] = pd.cut(tmp["entry_minutes_from_open"], bins=[-1, 5, 10, 15, 30, 60, 90, 120, 999])
        print(tmp.groupby("entry_minutes_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Top symbols by total pnl, best median variant ===")
        print(f"Best median variant: {best_name}")
        print(best.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).head(30).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Recent trades, best median variant ===")
        cols = ["session_date", "symbol", "entry_time", "entry_minutes_from_open", "entry_price", "pnl_pct", "max_pnl_pct", "min_pnl_pct", "reason", "v45_score", "or_close_strength", "entry_candle_strength", "pre_entry_break_below_or_pct", "or_high_pct", "intraday_high_pct", "gap_pct"]
        print(best[cols].tail(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if reject_counter:
        print("\n=== Top rejection reasons ===")
        for reason, count in reject_counter.most_common(20):
            print(f"{reason:110s} {count}")

    print("\nInterpretation hints:")
    print("- v45 should be compared mainly to v41 trail_only and close_exit.")
    print("- Good outcome: fewer trades than v41, better median, no future filters in rejection reasons.")
    print("- If count is too low, lower --min-v45-score to 6 or increase --max-entry-minutes-from-open.")
    print("- If median improves but avg stays flat, next step is symbol-quality / premarket catalyst layer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
