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
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v45_antifade import (
    first_n_stats,
    live_antifade_score,
    minutes_from_open,
)


def add_reject(rejected: list[dict[str, object]], counter: Counter[str], payload: dict[str, object], reason: str) -> None:
    counter[reason] += 1
    if len(rejected) < 200_000:
        rejected.append({**payload, "reject_reason": reason})


def make_baseline_variants(args: argparse.Namespace) -> dict[str, ExitConfig]:
    return {
        "v45_trail_only": ExitConfig(args.stop_loss, None, args.trailing_activation, args.trailing_stop, args.max_hold_minutes, True, None, 0.0),
        "close_exit": ExitConfig(args.stop_loss, None, None, None, args.max_hold_minutes, True, None, 0.0),
        "wide_trail": ExitConfig(args.stop_loss, None, args.wide_trailing_activation, args.wide_trailing_stop, args.max_hold_minutes, True, None, 0.0),
        "tp_wide_trail": ExitConfig(args.stop_loss, args.take_profit, args.wide_trailing_activation, args.wide_trailing_stop, args.max_hold_minutes, True, None, 0.0),
    }


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


def simulate_trend_exit(day: pd.DataFrame, entry_price: float, args: argparse.Namespace, mode: str) -> tuple[float, str, pd.Timestamp, float, float]:
    """Trend-aware exit variants for v46.

    Modes:
    - breakeven_trend: initial risk, move stop to breakeven after profit, wider trend trail after stronger profit.
    - staged_runner: sell partial at first target, then trail rest wider.
    - ratchet: progressively tightens only after higher profit levels.
    """
    max_pnl = -999.0
    min_pnl = 999.0
    peak = entry_price
    stop_price = entry_price * (1.0 - args.stop_loss / 100.0)
    partial_done = False
    realized_pct = 0.0
    remaining = 1.0

    for _, row in day.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        ts = pd.Timestamp(row["datetime"])
        peak = max(peak, high)
        max_pnl = max(max_pnl, (high / entry_price - 1.0) * 100.0)
        min_pnl = min(min_pnl, (low / entry_price - 1.0) * 100.0)
        current_peak_pnl = (peak / entry_price - 1.0) * 100.0

        if mode in {"breakeven_trend", "staged_runner", "ratchet"} and current_peak_pnl >= args.breakeven_activation:
            stop_price = max(stop_price, entry_price * (1.0 + args.breakeven_lock_pct / 100.0))

        if mode == "staged_runner" and not partial_done and current_peak_pnl >= args.stage_profit:
            realized_pct += args.stage_fraction * args.stage_profit
            remaining = 1.0 - args.stage_fraction
            partial_done = True
            stop_price = max(stop_price, entry_price * (1.0 + args.stage_stop_lock_pct / 100.0))

        if mode == "breakeven_trend":
            if current_peak_pnl >= args.trend_activation:
                stop_price = max(stop_price, peak * (1.0 - args.trend_trailing_stop / 100.0))
            elif current_peak_pnl >= args.trailing_activation:
                stop_price = max(stop_price, peak * (1.0 - args.trailing_stop / 100.0))
        elif mode == "staged_runner":
            if current_peak_pnl >= args.runner_activation:
                stop_price = max(stop_price, peak * (1.0 - args.runner_trailing_stop / 100.0))
            elif current_peak_pnl >= args.trailing_activation:
                stop_price = max(stop_price, peak * (1.0 - args.trailing_stop / 100.0))
        elif mode == "ratchet":
            if current_peak_pnl >= 10.0:
                trail = args.ratchet_trail_10
            elif current_peak_pnl >= 6.0:
                trail = args.ratchet_trail_6
            elif current_peak_pnl >= args.trailing_activation:
                trail = args.trailing_stop
            else:
                trail = None
            if trail is not None:
                stop_price = max(stop_price, peak * (1.0 - trail / 100.0))

        if low <= stop_price:
            remaining_pnl = (stop_price / entry_price - 1.0) * 100.0
            pnl = realized_pct + remaining * remaining_pnl
            reason = f"{mode}_stop"
            return pnl, reason, ts, max_pnl, min_pnl

    last = day.iloc[-1]
    ts = pd.Timestamp(last["datetime"])
    close_pnl = (float(last["close"]) / entry_price - 1.0) * 100.0
    pnl = realized_pct + remaining * close_pnl
    return pnl, f"{mode}_close", ts, max_pnl, min_pnl


def run_backtest_single_pass(args: argparse.Namespace, baseline_variants: dict[str, ExitConfig]) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str], Counter[str]]:
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    reject_counter: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    trend_modes = ["breakeven_trend", "staged_runner", "ratchet"]

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
        for col in ["gap_pct", "first_5m_high_pct", "first_15m_high_pct", "rows_1m"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        clean, _ = apply_quality_filter(
            raw,
            min_open_price=args.min_entry_price,
            max_open_price=args.max_entry_price,
            max_abs_gap=args.max_abs_gap,
            max_intraday_high=10_000.0,
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

        for label, cfg in baseline_variants.items():
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

        for mode in trend_modes:
            pnl, reason, exit_time, max_pnl, min_pnl = simulate_trend_exit(post_entry, entry_price, args, mode)
            rows.append({
                **common,
                **extra,
                "variant": mode,
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


def fixed_position_profit(total_pnl_pct: float, position_size: float) -> float:
    return total_pnl_pct / 100.0 * position_size


def main() -> int:
    parser = argparse.ArgumentParser(description="v46 trend-aware exits on v45 anti-fade entries.")
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
    parser.add_argument("--fixed-position-size", type=float, default=1000.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_variants = make_baseline_variants(args)

    print("Experiment: v46 trend-aware exits on v45 entries")
    print("Goal: keep v45 selection but recover larger momentum runners with smarter exits.")
    print("No future filters: intraday_high/open_to_close/time_to_high are diagnostics only.")
    print(f"Data dir: {args.data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Entry: OR{args.opening_range_minutes} breakout + v45 anti-fade score")
    print("Exit variants include: v45_trail_only, close_exit, wide_trail, breakeven_trend, staged_runner, ratchet")

    trades_df, rejects_df, reject_counter, stats = run_backtest_single_pass(args, baseline_variants)

    variant_order = ["v45_trail_only", "close_exit", "wide_trail", "tp_wide_trail", "breakeven_trend", "staged_runner", "ratchet"]
    summaries = []
    for label in variant_order:
        subset = trades_df[trades_df["variant"] == label] if not trades_df.empty else pd.DataFrame()
        if not subset.empty or label in variant_order:
            row = summarize(label, subset)
            row["fixed_1000_profit_usd"] = fixed_position_profit(float(row.get("total_pnl", 0.0) or 0.0), args.fixed_position_size)
            summaries.append(row)
    summary = pd.DataFrame(summaries)

    suffix = f"recent{args.recent_days}_or{args.opening_range_minutes}_score{args.min_v45_score:g}_trendexit"
    out_trades = output_dir / f"v46_trend_exit_momentum_or_breakout_trades_{suffix}.csv"
    out_summary = output_dir / f"v46_trend_exit_momentum_or_breakout_summary_{suffix}.csv"
    out_rejects = output_dir / f"v46_trend_exit_momentum_or_breakout_rejected_{suffix}.csv"
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
        cols = ["strategy", "count", "active_days", "symbols", "win_rate", "avg_pnl", "median_pnl", "total_pnl", "fixed_1000_profit_usd", "avg_win", "avg_loss", "avg_max_pnl", "avg_min_pnl", "max_loss", "max_win"]
        print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not trades_df.empty:
        print("\n=== Exit reasons by variant ===")
        print(pd.crosstab(trades_df["variant"], trades_df["reason"]).to_string())
        best_name = summary.sort_values(["median_pnl", "avg_pnl"], ascending=False).iloc[0]["strategy"]
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
    print("- Compare v45_trail_only with breakeven_trend/staged_runner/ratchet.")
    print("- Good outcome: median stays positive while total_pnl/fixed-position profit improves.")
    print("- If close_exit still has much higher total but weak median, exits are still cutting/fading wrong names.")
    print("- If all exits are similar, next step is symbol quality / premarket catalyst layer, not exit tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
