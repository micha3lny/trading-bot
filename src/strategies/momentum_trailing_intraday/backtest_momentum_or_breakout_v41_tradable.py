from __future__ import annotations

import argparse
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


def looks_like_weird_security(symbol: str) -> bool:
    s = symbol.upper().strip()
    if len(s) >= 5 and (s.endswith("W") or s.endswith("WS") or s.endswith("WT") or s.endswith("WQ")):
        return True
    if len(s) >= 5 and s.endswith("U"):
        return True
    if len(s) >= 5 and s.endswith("R"):
        return True
    if "." in s or "-" in s or "/" in s:
        return True
    return False


def opening_range_stats(day: pd.DataFrame, opening_range_minutes: int) -> dict[str, float]:
    opening = day.iloc[:opening_range_minutes]
    open_price = float(day.iloc[0]["open"])
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())
    or_close = float(opening.iloc[-1]["close"])
    if "volume" in opening.columns:
        or_volume = float(opening["volume"].sum())
        or_dollar_volume = float((opening["close"] * opening["volume"]).sum())
    else:
        or_volume = 0.0
        or_dollar_volume = 0.0
    return {
        "or_high_pct": (or_high / open_price - 1.0) * 100.0,
        "or_low_pct": (or_low / open_price - 1.0) * 100.0,
        "or_close_pct": (or_close / open_price - 1.0) * 100.0,
        "or_range_pct": (or_high / or_low - 1.0) * 100.0 if or_low > 0 else float("nan"),
        "or_volume": or_volume,
        "or_dollar_volume": or_dollar_volume,
    }


def session_liquidity(day: pd.DataFrame) -> dict[str, float]:
    if "volume" not in day.columns:
        return {"session_volume": 0.0, "session_dollar_volume": 0.0, "median_1m_volume": 0.0}
    return {
        "session_volume": float(day["volume"].sum()),
        "session_dollar_volume": float((day["close"] * day["volume"]).sum()),
        "median_1m_volume": float(day["volume"].median()),
    }


def passes_v41_live_filter(symbol: str, day: pd.DataFrame, feats: dict[str, object], args: argparse.Namespace) -> tuple[bool, list[str], dict[str, float]]:
    reasons: list[str] = []
    open_price = float(day.iloc[0]["open"])
    ors = opening_range_stats(day, args.opening_range_minutes)
    liq = session_liquidity(day)
    extra = {**ors, **liq}

    if args.exclude_weird_symbols and looks_like_weird_security(symbol):
        reasons.append("weird_security_suffix")
    if not args.include_leveraged_etfs and symbol.upper() in DEFAULT_EXCLUDED_SYMBOLS:
        reasons.append("leveraged_or_excluded_etf")
    if open_price < args.min_entry_price:
        reasons.append("entry_price_too_low")
    if args.max_entry_price is not None and open_price > args.max_entry_price:
        reasons.append("entry_price_too_high")
    if ors["or_dollar_volume"] < args.min_or_dollar_volume:
        reasons.append("opening_range_dollar_volume_too_low")
    if ors["or_volume"] < args.min_or_volume:
        reasons.append("opening_range_volume_too_low")
    if args.min_session_dollar_volume > 0 and liq["session_dollar_volume"] < args.min_session_dollar_volume:
        reasons.append("session_dollar_volume_too_low")
    if ors["or_range_pct"] > args.max_or_range_pct:
        reasons.append("opening_range_too_wide")
    if ors["or_high_pct"] < args.min_or_high_pct:
        reasons.append("opening_range_momentum_too_low")
    if ors["or_high_pct"] > args.max_or_high_pct:
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

    return not reasons, reasons, extra


def run_backtest(args: argparse.Namespace, cfg: ExitConfig, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for symbol, day, prev_close in iter_days(Path(args.data_dir), args.recent_days):
        if len(day) < args.min_rows:
            continue

        feats = day_features(symbol, day, prev_close)
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
            rejected.append({**feats, "variant": label, "reject_reason": "base_clean_filter"})
            continue

        ok, reasons, extra = passes_v41_live_filter(symbol, day, feats, args)
        if not ok:
            rejected.append({**feats, **extra, "variant": label, "reject_reason": ";".join(reasons)})
            continue

        entry = find_or_breakout_entry(day, args.opening_range_minutes)
        if entry is None:
            rejected.append({**feats, **extra, "variant": label, "reject_reason": "no_or_breakout"})
            continue

        entry_pos, entry_price = entry
        post_entry = day.iloc[entry_pos:].copy()
        if post_entry.empty:
            continue

        pnl, reason, exit_time, max_pnl, min_pnl = simulate_exit(post_entry, entry_price, cfg)
        result = TradeResult(
            symbol=symbol,
            session_date=str(feats["session_date"]),
            entry_time=str(pd.Timestamp(post_entry.iloc[0]["datetime"])),
            entry_price=entry_price,
            exit_time=str(exit_time),
            exit_price=entry_price * (1.0 + pnl / 100.0),
            pnl_pct=pnl,
            max_pnl_pct=max_pnl,
            min_pnl_pct=min_pnl,
            reason=reason,
            intraday_high_pct=float(feats["intraday_high_pct"]),
            gap_pct=float(feats["gap_pct"]) if pd.notna(feats["gap_pct"]) else None,
            first_5m_high_pct=float(feats["first_5m_high_pct"]),
            first_15m_high_pct=float(feats["first_15m_high_pct"]),
            time_to_high_minutes=float(feats["time_to_high_minutes"]),
            opening_range_minutes=args.opening_range_minutes,
            open_to_close_pct=float(feats["open_to_close_pct"]),
        )
        rows.append({**result.__dict__, **extra, "variant": label})

    trades = pd.DataFrame(rows)
    rejects = pd.DataFrame(rejected)
    if not trades.empty:
        trades = trades.sort_values(["session_date", "symbol", "entry_time"]).reset_index(drop=True)
    return trades, rejects


def main() -> int:
    parser = argparse.ArgumentParser(description="v41 tradable momentum OR breakout backtest.")
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
    parser.add_argument("--min-session-dollar-volume", type=float, default=0.0)
    parser.add_argument("--research-min-intraday-high", type=float, default=None)
    parser.add_argument("--exclude-weird-symbols", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-leveraged-etfs", action="store_true")
    parser.add_argument("--stop-loss", type=float, default=8.0)
    parser.add_argument("--take-profit", type=float, default=25.0)
    parser.add_argument("--trailing-activation", type=float, default=2.5)
    parser.add_argument("--trailing-stop", type=float, default=1.5)
    parser.add_argument("--max-hold-minutes", type=int, default=None)
    parser.add_argument("--staged-take-profit", type=float, default=10.0)
    parser.add_argument("--staged-fraction", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "close_exit": ExitConfig(args.stop_loss, None, None, None, args.max_hold_minutes, True, None, 0.0),
        "trail_only": ExitConfig(args.stop_loss, None, args.trailing_activation, args.trailing_stop, args.max_hold_minutes, True, None, 0.0),
        "tp_trail": ExitConfig(args.stop_loss, args.take_profit, args.trailing_activation, args.trailing_stop, args.max_hold_minutes, True, None, 0.0),
        "staged": ExitConfig(args.stop_loss, args.take_profit, args.trailing_activation, args.trailing_stop, args.max_hold_minutes, True, args.staged_take_profit, args.staged_fraction),
    }

    print("Experiment: v41 tradable momentum OR breakout")
    print("Live-like filters use only symbol/static/opening-range information before entry.")
    print(f"Data dir: {args.data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Entry: OR{args.opening_range_minutes} breakout")
    print("Tradability filters:")
    print(f"- entry price: {args.min_entry_price:.2f} to {args.max_entry_price:.2f}")
    print(f"- OR high pct: {args.min_or_high_pct:.2f}% to {args.max_or_high_pct:.2f}%")
    print(f"- OR range <= {args.max_or_range_pct:.2f}%")
    print(f"- OR volume >= {args.min_or_volume:.0f}")
    print(f"- OR dollar volume >= {args.min_or_dollar_volume:.0f}")
    print(f"- abs gap <= {args.max_abs_gap:.2f}%")
    print(f"- exclude weird symbols: {args.exclude_weird_symbols}")
    print(f"- include leveraged ETFs: {args.include_leveraged_etfs}")
    if args.research_min_intraday_high is not None:
        print(f"Research-only future filter: intraday_high >= {args.research_min_intraday_high:.2f}%")

    all_trades: list[pd.DataFrame] = []
    all_rejects: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for label, cfg in variants.items():
        trades, rejects = run_backtest(args, cfg, label)
        all_trades.append(trades)
        all_rejects.append(rejects)
        summaries.append(summarize(label, trades))

    summary = pd.DataFrame(summaries)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    rejects_df = pd.concat(all_rejects, ignore_index=True) if all_rejects else pd.DataFrame()

    research = "research" if args.research_min_intraday_high is not None else "livelike"
    suffix = f"recent{args.recent_days}_or{args.opening_range_minutes}_{research}_minpx{args.min_entry_price:g}_minordv{int(args.min_or_dollar_volume)}"
    out_trades = output_dir / f"v41_tradable_momentum_or_breakout_trades_{suffix}.csv"
    out_summary = output_dir / f"v41_tradable_momentum_or_breakout_summary_{suffix}.csv"
    out_rejects = output_dir / f"v41_tradable_momentum_or_breakout_rejected_{suffix}.csv"
    trades_df.to_csv(out_trades, index=False)
    summary.to_csv(out_summary, index=False)
    rejects_df.to_csv(out_rejects, index=False)

    print(f"\nSaved trades CSV: {out_trades}")
    print(f"Saved summary CSV: {out_summary}")
    print(f"Saved rejected CSV: {out_rejects}")

    print("\n=== Variant comparison ===")
    if not summary.empty:
        cols = ["strategy", "count", "active_days", "symbols", "win_rate", "avg_pnl", "median_pnl", "total_pnl", "avg_win", "avg_loss", "avg_max_pnl", "avg_min_pnl", "max_loss", "max_win"]
        print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not trades_df.empty:
        print("\n=== Exit reasons by variant ===")
        print(pd.crosstab(trades_df["variant"], trades_df["reason"]).to_string())
        best_name = summary.sort_values("avg_pnl", ascending=False).iloc[0]["strategy"]
        best = trades_df[trades_df["variant"] == best_name]
        print("\n=== Top symbols by total pnl, best variant only ===")
        print(f"Best variant: {best_name}")
        print(best.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).head(30).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Liquidity buckets, best variant ===")
        tmp = best.copy()
        tmp["or_dollar_volume_bin"] = pd.cut(tmp["or_dollar_volume"], bins=[0, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, float("inf")])
        print(tmp.groupby("or_dollar_volume_bin", observed=True)["pnl_pct"].agg(["count", "mean", "median", "sum"]).to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== Recent trades, best variant ===")
        cols = ["session_date", "symbol", "entry_time", "entry_price", "pnl_pct", "max_pnl_pct", "min_pnl_pct", "reason", "or_high_pct", "or_dollar_volume", "intraday_high_pct", "gap_pct"]
        print(best[cols].tail(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not rejects_df.empty:
        print("\n=== Top rejection reasons ===")
        print(rejects_df["reject_reason"].value_counts().head(20).to_string())

    print("\nInterpretation hints:")
    print("- Run without --research-min-intraday-high for live-like behavior.")
    print("- Add --research-min-intraday-high 10 only to study known big-move days.")
    print("- If live-like count is huge, tighten --min-or-dollar-volume, --min-or-high-pct or --max-or-range-pct.")
    print("- If results depend only on low OR liquidity buckets, the strategy is probably not executable live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
