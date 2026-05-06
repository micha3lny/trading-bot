from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis.big_momentum_available_1m_clean_research import DEFAULT_EXCLUDED_SYMBOLS, is_weird_symbol
from src.analysis.big_momentum_available_1m_research import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, _load_1m_file
from src.strategies.momentum_trailing_intraday.backtest_momentum_or_breakout_v40 import (
    ExitConfig,
    pct,
    simulate_exit,
    summarize,
)


@dataclass(frozen=True)
class LiveTradeResult:
    symbol: str
    session_date: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    pnl_pct: float
    max_pnl_pct: float
    min_pnl_pct: float
    reason: str
    live_move_pct: float
    live_breakout_pct: float
    gap_pct: float | None
    first_5m_high_pct: float
    first_15m_high_pct: float
    cumulative_dollar_volume: float
    entry_minute: int
    opening_range_minutes: int
    day_intraday_high_pct: float
    day_open_to_close_pct: float


def iter_days(data_dir: Path, recent_days: int | None) -> Iterable[tuple[str, pd.DataFrame, float | None]]:
    files = sorted(data_dir.glob("*.csv"))
    loaded: list[tuple[str, pd.DataFrame]] = []
    latest_seen: pd.Timestamp | None = None

    for path in files:
        symbol = path.stem.upper()
        df = _load_1m_file(path)
        if df is None or df.empty:
            continue
        loaded.append((symbol, df))
        max_dt = pd.Timestamp(df["datetime"].max())
        if latest_seen is None or max_dt > latest_seen:
            latest_seen = max_dt

    cutoff = None
    if recent_days and latest_seen is not None:
        cutoff = latest_seen.normalize() - pd.Timedelta(days=recent_days)

    for symbol, df in loaded:
        if cutoff is not None:
            df = df[df["datetime"] >= cutoff]
        if df.empty:
            continue
        prev_close: float | None = None
        for _, day in df.groupby("session_date", sort=True):
            day = day.reset_index(drop=True)
            if len(day) < 30:
                prev_close = float(day.iloc[-1]["close"])
                continue
            yield symbol, day, prev_close
            prev_close = float(day.iloc[-1]["close"])


def cumulative_dollar_volume(day: pd.DataFrame, end_pos: int) -> float:
    if "volume" not in day.columns:
        return 0.0
    part = day.iloc[: end_pos + 1]
    typical = (part["high"].astype(float) + part["low"].astype(float) + part["close"].astype(float)) / 3.0
    volume = pd.to_numeric(part["volume"], errors="coerce").fillna(0.0)
    return float((typical * volume).sum())


def find_live_entry(
    day: pd.DataFrame,
    opening_range_minutes: int,
    min_live_move: float,
    max_entry_minute: int | None,
    min_cumulative_dollar_volume: float,
) -> tuple[int, float, float, float, float] | None:
    if len(day) <= opening_range_minutes + 1:
        return None

    open_price = float(day.iloc[0]["open"])
    opening = day.iloc[:opening_range_minutes]
    or_high = float(opening["high"].max())
    if open_price <= 0 or or_high <= 0:
        return None

    last_pos = len(day) - 1
    if max_entry_minute is not None:
        last_pos = min(last_pos, max_entry_minute)

    for pos in range(opening_range_minutes, last_pos + 1):
        bar = day.iloc[pos]
        high = float(bar["high"])
        if high <= or_high:
            continue
        live_move = pct(high, open_price)
        if live_move < min_live_move:
            continue
        dollar_vol = cumulative_dollar_volume(day, pos)
        if dollar_vol < min_cumulative_dollar_volume:
            continue
        live_breakout = pct(high, or_high)
        # Realistic fill assumption: enter at the worse of OR high and current bar open if the bar gaps through OR.
        entry_price = max(or_high, float(bar["open"]))
        return pos, entry_price, live_move, live_breakout, dollar_vol

    return None


def passes_live_quality_filters(args: argparse.Namespace, symbol: str, day: pd.DataFrame, prev_close: float | None) -> tuple[bool, dict[str, float | None]]:
    open_price = float(day.iloc[0]["open"])
    close_price = float(day.iloc[-1]["close"])
    high_price = float(day["high"].max())
    first_5 = day.iloc[:5]
    first_15 = day.iloc[:15]

    gap_pct = ((open_price / prev_close - 1.0) * 100.0) if prev_close and prev_close > 0 else None
    first_5m_high_pct = pct(float(first_5["high"].max()), open_price)
    first_15m_high_pct = pct(float(first_15["high"].max()), open_price)
    day_intraday_high_pct = pct(high_price, open_price)
    day_open_to_close_pct = pct(close_price, open_price)

    if args.exclude_weird_symbols and is_weird_symbol(symbol):
        return False, {}
    if not args.include_leveraged_etfs and symbol.upper() in DEFAULT_EXCLUDED_SYMBOLS:
        return False, {}
    if len(day) < args.min_rows:
        return False, {}
    if open_price < args.min_open_price:
        return False, {}
    if args.max_open_price is not None and open_price > args.max_open_price:
        return False, {}
    if gap_pct is not None and abs(gap_pct) > args.max_abs_gap:
        return False, {}
    # This is known after the first 5 minutes, so it is live-safe for OR5/OR15/OR30 entries.
    if first_5m_high_pct > args.max_first_5m_high:
        return False, {}
    if args.max_first_15m_high is not None and first_15m_high_pct > args.max_first_15m_high:
        return False, {}
    if args.min_first_15m_high is not None and first_15m_high_pct < args.min_first_15m_high:
        return False, {}

    return True, {
        "gap_pct": gap_pct,
        "first_5m_high_pct": first_5m_high_pct,
        "first_15m_high_pct": first_15m_high_pct,
        "day_intraday_high_pct": day_intraday_high_pct,
        "day_open_to_close_pct": day_open_to_close_pct,
    }


def run_backtest(args: argparse.Namespace, cfg: ExitConfig, label: str) -> pd.DataFrame:
    rows: list[LiveTradeResult] = []

    for symbol, day, prev_close in iter_days(Path(args.data_dir), args.recent_days):
        ok, feats = passes_live_quality_filters(args, symbol, day, prev_close)
        if not ok:
            continue

        entry = find_live_entry(
            day,
            opening_range_minutes=args.opening_range_minutes,
            min_live_move=args.min_live_move,
            max_entry_minute=args.max_entry_minute,
            min_cumulative_dollar_volume=args.min_cumulative_dollar_volume,
        )
        if entry is None:
            continue
        entry_pos, entry_price, live_move, live_breakout, dollar_vol = entry
        post_entry = day.iloc[entry_pos:].copy()
        if post_entry.empty:
            continue

        pnl, reason, exit_time, max_pnl, min_pnl = simulate_exit(post_entry, entry_price, cfg)
        rows.append(
            LiveTradeResult(
                symbol=symbol,
                session_date=str(day.iloc[0]["session_date"]),
                entry_time=str(pd.Timestamp(post_entry.iloc[0]["datetime"])),
                entry_price=entry_price,
                exit_time=str(exit_time),
                exit_price=entry_price * (1.0 + pnl / 100.0),
                pnl_pct=pnl,
                max_pnl_pct=max_pnl,
                min_pnl_pct=min_pnl,
                reason=reason,
                live_move_pct=live_move,
                live_breakout_pct=live_breakout,
                gap_pct=feats["gap_pct"],
                first_5m_high_pct=float(feats["first_5m_high_pct"]),
                first_15m_high_pct=float(feats["first_15m_high_pct"]),
                cumulative_dollar_volume=dollar_vol,
                entry_minute=entry_pos,
                opening_range_minutes=args.opening_range_minutes,
                day_intraday_high_pct=float(feats["day_intraday_high_pct"]),
                day_open_to_close_pct=float(feats["day_open_to_close_pct"]),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if not df.empty:
        df["variant"] = label
        df = df.sort_values(["session_date", "symbol", "entry_time"]).reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="v41 live-realistic momentum OR breakout backtest.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--opening-range-minutes", type=int, default=5, choices=[5, 15, 30])
    parser.add_argument("--min-live-move", type=float, default=5.0, help="Required live move vs open at or before entry; no future data used.")
    parser.add_argument("--max-entry-minute", type=int, default=180, help="Latest minute index allowed for entry after session open.")
    parser.add_argument("--min-cumulative-dollar-volume", type=float, default=500_000.0)
    parser.add_argument("--max-abs-gap", type=float, default=30.0)
    parser.add_argument("--max-first-5m-high", type=float, default=50.0)
    parser.add_argument("--min-first-15m-high", type=float, default=None)
    parser.add_argument("--max-first-15m-high", type=float, default=None)
    parser.add_argument("--min-open-price", type=float, default=1.0)
    parser.add_argument("--max-open-price", type=float, default=None)
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

    print("Experiment: v41 live-realistic momentum OR breakout")
    print("No future-day filters are used for entry.")
    print(f"Data dir: {args.data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Entry: OR{args.opening_range_minutes} breakout + live move >= {args.min_live_move:.2f}%")
    print("Live-safe filters:")
    print(f"- rows_1m >= {args.min_rows}")
    print(f"- max_entry_minute <= {args.max_entry_minute}")
    print(f"- cumulative dollar volume >= {args.min_cumulative_dollar_volume:,.0f}")
    print(f"- abs(gap_pct) <= {args.max_abs_gap:.2f}%")
    print(f"- first_5m_high_pct <= {args.max_first_5m_high:.2f}%")
    print(f"- min_open_price >= {args.min_open_price:.2f}")
    print(f"- exclude weird symbols: {args.exclude_weird_symbols}")
    print(f"- exclude leveraged ETFs: {not args.include_leveraged_etfs}")
    print("Exit variants:")
    print(f"- stop_loss={args.stop_loss:.2f}%")
    print(f"- take_profit={args.take_profit:.2f}%")
    print(f"- trailing_activation={args.trailing_activation:.2f}%, trailing_stop={args.trailing_stop:.2f}%")
    print(f"- staged: sell {args.staged_fraction:.0%} at +{args.staged_take_profit:.2f}%")

    all_trades = []
    summaries = []
    for label, cfg in variants.items():
        trades = run_backtest(args, cfg, label)
        all_trades.append(trades)
        summaries.append(summarize(label, trades))

    summary = pd.DataFrame(summaries)
    all_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    suffix = (
        f"recent{args.recent_days}_or{args.opening_range_minutes}_"
        f"livege{int(args.min_live_move)}_maxentry{args.max_entry_minute}_"
        f"maxgap{int(args.max_abs_gap)}"
    )
    out_trades = output_dir / f"v41_live_momentum_or_breakout_trades_{suffix}.csv"
    out_summary = output_dir / f"v41_live_momentum_or_breakout_summary_{suffix}.csv"
    all_df.to_csv(out_trades, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\nSaved trades CSV: {out_trades}")
    print(f"Saved summary CSV: {out_summary}")

    print("\n=== Variant comparison ===")
    if not summary.empty:
        cols = [
            "strategy", "count", "active_days", "symbols", "win_rate", "avg_pnl", "median_pnl",
            "total_pnl", "avg_win", "avg_loss", "avg_max_pnl", "avg_min_pnl", "max_loss", "max_win",
        ]
        print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not all_df.empty:
        print("\n=== Exit reasons by variant ===")
        print(pd.crosstab(all_df["variant"], all_df["reason"]).to_string())

        best_name = summary.sort_values("avg_pnl", ascending=False).iloc[0]["strategy"]
        best = all_df[all_df["variant"] == best_name]
        print("\n=== Live entry diagnostics, best variant ===")
        diag = best[["live_move_pct", "entry_minute", "cumulative_dollar_volume", "day_intraday_high_pct", "day_open_to_close_pct"]].describe()
        print(diag.to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Top symbols by total pnl, best variant only ===")
        sym = best.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).head(30)
        print(f"Best variant: {best_name}")
        print(sym.to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Recent trades, best variant ===")
        cols = [
            "session_date", "symbol", "entry_time", "entry_minute", "pnl_pct", "max_pnl_pct", "min_pnl_pct",
            "reason", "live_move_pct", "day_intraday_high_pct", "gap_pct", "cumulative_dollar_volume",
        ]
        print(best[cols].tail(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- This version does not filter on final intraday_high_pct before entry.")
    print("- day_intraday_high_pct is printed only for diagnostics after the simulated trade.")
    print("- If close_exit wins again, the edge is real momentum persistence, not future-day filtering.")
    print("- If trade count explodes, tighten min_live_move, dollar volume, or max_entry_minute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
