from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis.big_momentum_available_1m_clean_research import (
    DEFAULT_EXCLUDED_SYMBOLS,
    apply_quality_filter,
)
from src.analysis.big_momentum_available_1m_research import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    _load_1m_file,
)


@dataclass(frozen=True)
class ExitConfig:
    stop_loss: float
    take_profit: float | None
    trailing_activation: float | None
    trailing_stop: float | None
    max_hold_minutes: int | None
    exit_at_close: bool
    staged_take_profit: float | None
    staged_fraction: float


@dataclass(frozen=True)
class TradeResult:
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
    intraday_high_pct: float
    gap_pct: float | None
    first_5m_high_pct: float
    first_15m_high_pct: float
    time_to_high_minutes: float
    opening_range_minutes: int
    open_to_close_pct: float


def pct(price: float, entry: float) -> float:
    return (price / entry - 1.0) * 100.0


def day_features(symbol: str, day: pd.DataFrame, prev_close: float | None) -> dict[str, object]:
    open_price = float(day.iloc[0]["open"])
    close_price = float(day.iloc[-1]["close"])
    high_price = float(day["high"].max())
    high_idx = day["high"].idxmax()
    start_time = pd.Timestamp(day.iloc[0]["datetime"])
    high_time = pd.Timestamp(day.loc[high_idx, "datetime"])
    first_5 = day.iloc[:5]
    first_15 = day.iloc[:15]

    return {
        "session_date": str(day.iloc[0]["session_date"]),
        "symbol": symbol,
        "rows_1m": len(day),
        # Use NaN instead of None so pandas numeric filters like .abs() are safe.
        "gap_pct": ((open_price / prev_close - 1.0) * 100.0) if prev_close and prev_close > 0 else float("nan"),
        "intraday_high_pct": (high_price / open_price - 1.0) * 100.0,
        "open_to_close_pct": (close_price / open_price - 1.0) * 100.0,
        "first_5m_high_pct": (float(first_5["high"].max()) / open_price - 1.0) * 100.0,
        "first_15m_high_pct": (float(first_15["high"].max()) / open_price - 1.0) * 100.0,
        "time_to_high_minutes": (high_time - start_time).total_seconds() / 60.0,
    }


def find_or_breakout_entry(day: pd.DataFrame, opening_range_minutes: int) -> tuple[int, float] | None:
    if len(day) <= opening_range_minutes + 1:
        return None
    opening = day.iloc[:opening_range_minutes]
    rest = day.iloc[opening_range_minutes:]
    or_high = float(opening["high"].max())
    if or_high <= 0:
        return None
    breakout = rest[rest["high"] > or_high]
    if breakout.empty:
        return None
    idx = breakout.index[0]
    pos = day.index.get_loc(idx)
    return pos, or_high


def simulate_exit(post_entry: pd.DataFrame, entry_price: float, cfg: ExitConfig) -> tuple[float, str, pd.Timestamp, float, float]:
    peak = entry_price
    staged_realized: float | None = None
    staged_fraction = max(0.0, min(1.0, cfg.staged_fraction))
    entry_time = pd.Timestamp(post_entry.iloc[0]["datetime"])

    max_pnl = -999.0
    min_pnl = 999.0

    for _, bar in post_entry.iterrows():
        now = pd.Timestamp(bar["datetime"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        peak = max(peak, high)
        max_pnl = max(max_pnl, pct(high, entry_price))
        min_pnl = min(min_pnl, pct(low, entry_price))

        if cfg.staged_take_profit is not None and staged_realized is None:
            staged_price = entry_price * (1.0 + cfg.staged_take_profit / 100.0)
            if high >= staged_price:
                staged_realized = cfg.staged_take_profit

        stop_price = entry_price * (1.0 - cfg.stop_loss / 100.0)
        if low <= stop_price:
            raw = -cfg.stop_loss
            if staged_realized is not None:
                blended = staged_fraction * staged_realized + (1.0 - staged_fraction) * raw
                return blended, "staged_then_stop", now, max_pnl, min_pnl
            return raw, "stop_loss", now, max_pnl, min_pnl

        if cfg.take_profit is not None:
            tp_price = entry_price * (1.0 + cfg.take_profit / 100.0)
            if high >= tp_price:
                raw = cfg.take_profit
                if staged_realized is not None:
                    blended = staged_fraction * staged_realized + (1.0 - staged_fraction) * raw
                    return blended, "staged_then_take_profit", now, max_pnl, min_pnl
                return raw, "take_profit", now, max_pnl, min_pnl

        if cfg.trailing_activation is not None and cfg.trailing_stop is not None:
            if pct(peak, entry_price) >= cfg.trailing_activation:
                trail_price = peak * (1.0 - cfg.trailing_stop / 100.0)
                if low <= trail_price:
                    raw = pct(trail_price, entry_price)
                    if staged_realized is not None:
                        blended = staged_fraction * staged_realized + (1.0 - staged_fraction) * raw
                        return blended, "staged_then_trailing_stop", now, max_pnl, min_pnl
                    return raw, "trailing_stop", now, max_pnl, min_pnl

        if cfg.max_hold_minutes is not None:
            held = (now - entry_time).total_seconds() / 60.0
            if held >= cfg.max_hold_minutes:
                raw = pct(close, entry_price)
                if staged_realized is not None:
                    blended = staged_fraction * staged_realized + (1.0 - staged_fraction) * raw
                    return blended, "staged_then_time_exit", now, max_pnl, min_pnl
                return raw, "time_exit", now, max_pnl, min_pnl

    final_close = float(post_entry.iloc[-1]["close"])
    final_time = pd.Timestamp(post_entry.iloc[-1]["datetime"])
    raw = pct(final_close, entry_price)
    if staged_realized is not None:
        blended = staged_fraction * staged_realized + (1.0 - staged_fraction) * raw
        return blended, "staged_then_close", final_time, max_pnl, min_pnl
    return raw, "close_exit" if cfg.exit_at_close else "end_of_data", final_time, max_pnl, min_pnl


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
            if len(day) < 30:
                prev_close = float(day.iloc[-1]["close"])
                continue
            yield symbol, day.reset_index(drop=True), prev_close
            prev_close = float(day.iloc[-1]["close"])


def summarize(name: str, trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {"strategy": name, "count": 0}
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    return {
        "strategy": name,
        "count": len(trades),
        "active_days": trades["session_date"].nunique(),
        "symbols": trades["symbol"].nunique(),
        "win_rate": len(wins) / len(trades) * 100.0,
        "avg_pnl": trades["pnl_pct"].mean(),
        "median_pnl": trades["pnl_pct"].median(),
        "total_pnl": trades["pnl_pct"].sum(),
        "avg_win": wins["pnl_pct"].mean() if not wins.empty else 0.0,
        "avg_loss": losses["pnl_pct"].mean() if not losses.empty else 0.0,
        "avg_max_pnl": trades["max_pnl_pct"].mean(),
        "avg_min_pnl": trades["min_pnl_pct"].mean(),
        "max_loss": trades["pnl_pct"].min(),
        "max_win": trades["pnl_pct"].max(),
    }


def run_backtest(args: argparse.Namespace, cfg: ExitConfig, label: str) -> pd.DataFrame:
    rows: list[TradeResult] = []
    excluded = set() if args.include_leveraged_etfs else DEFAULT_EXCLUDED_SYMBOLS

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
            min_open_price=args.min_open_price,
            max_open_price=args.max_open_price,
            max_abs_gap=args.max_abs_gap,
            max_intraday_high=args.max_intraday_high,
            max_first_5m_high=args.max_first_5m_high,
            min_dollar_volume=0.0,
            exclude_weird_symbols=not args.include_weird_symbols,
            excluded_symbols=excluded,
        )
        if clean.empty:
            continue
        if float(feats["intraday_high_pct"]) < args.min_intraday_high:
            continue
        if args.max_first_15m_high is not None and float(feats["first_15m_high_pct"]) > args.max_first_15m_high:
            continue
        if args.min_first_15m_high is not None and float(feats["first_15m_high_pct"]) < args.min_first_15m_high:
            continue

        entry = find_or_breakout_entry(day, args.opening_range_minutes)
        if entry is None:
            continue
        entry_pos, entry_price = entry
        post_entry = day.iloc[entry_pos:].copy()
        if post_entry.empty:
            continue

        pnl, reason, exit_time, max_pnl, min_pnl = simulate_exit(post_entry, entry_price, cfg)
        rows.append(
            TradeResult(
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
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if not df.empty:
        df["variant"] = label
        df = df.sort_values(["session_date", "symbol", "entry_time"]).reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="v40 momentum OR breakout backtest on clean local 1m universe.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--min-intraday-high", type=float, default=10.0)
    parser.add_argument("--max-intraday-high", type=float, default=80.0)
    parser.add_argument("--max-abs-gap", type=float, default=30.0)
    parser.add_argument("--max-first-5m-high", type=float, default=50.0)
    parser.add_argument("--min-first-15m-high", type=float, default=None)
    parser.add_argument("--max-first-15m-high", type=float, default=None)
    parser.add_argument("--min-open-price", type=float, default=1.0)
    parser.add_argument("--max-open-price", type=float, default=None)
    parser.add_argument("--opening-range-minutes", type=int, default=5, choices=[5, 15, 30])
    parser.add_argument("--include-weird-symbols", action="store_true")
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
        "close_exit": ExitConfig(
            stop_loss=args.stop_loss,
            take_profit=None,
            trailing_activation=None,
            trailing_stop=None,
            max_hold_minutes=args.max_hold_minutes,
            exit_at_close=True,
            staged_take_profit=None,
            staged_fraction=0.0,
        ),
        "trail_only": ExitConfig(
            stop_loss=args.stop_loss,
            take_profit=None,
            trailing_activation=args.trailing_activation,
            trailing_stop=args.trailing_stop,
            max_hold_minutes=args.max_hold_minutes,
            exit_at_close=True,
            staged_take_profit=None,
            staged_fraction=0.0,
        ),
        "tp_trail": ExitConfig(
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trailing_activation=args.trailing_activation,
            trailing_stop=args.trailing_stop,
            max_hold_minutes=args.max_hold_minutes,
            exit_at_close=True,
            staged_take_profit=None,
            staged_fraction=0.0,
        ),
        "staged": ExitConfig(
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trailing_activation=args.trailing_activation,
            trailing_stop=args.trailing_stop,
            max_hold_minutes=args.max_hold_minutes,
            exit_at_close=True,
            staged_take_profit=args.staged_take_profit,
            staged_fraction=args.staged_fraction,
        ),
    }

    print("Experiment: v40 momentum OR breakout")
    print("Clean local 1m momentum-continuation strategy")
    print(f"Data dir: {args.data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Entry: OR{args.opening_range_minutes} breakout")
    print("Filters:")
    print(f"- rows_1m >= {args.min_rows}")
    print(f"- intraday_high_pct: {args.min_intraday_high:.2f}% to {args.max_intraday_high:.2f}%")
    print(f"- abs(gap_pct) <= {args.max_abs_gap:.2f}%")
    print(f"- first_5m_high_pct <= {args.max_first_5m_high:.2f}%")
    print(f"- exclude weird symbols: {not args.include_weird_symbols}")
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
        f"ge{int(args.min_intraday_high)}_maxhi{int(args.max_intraday_high)}_maxgap{int(args.max_abs_gap)}"
    )
    out_trades = output_dir / f"v40_momentum_or_breakout_trades_{suffix}.csv"
    out_summary = output_dir / f"v40_momentum_or_breakout_summary_{suffix}.csv"
    all_df.to_csv(out_trades, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"\nSaved trades CSV: {out_trades}")
    print(f"Saved summary CSV: {out_summary}")

    print("\n=== Variant comparison ===")
    if not summary.empty:
        cols = [
            "strategy",
            "count",
            "active_days",
            "symbols",
            "win_rate",
            "avg_pnl",
            "median_pnl",
            "total_pnl",
            "avg_win",
            "avg_loss",
            "avg_max_pnl",
            "avg_min_pnl",
            "max_loss",
            "max_win",
        ]
        print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not all_df.empty:
        print("\n=== Exit reasons by variant ===")
        print(pd.crosstab(all_df["variant"], all_df["reason"]).to_string())

        print("\n=== Top symbols by total pnl, best variant only ===")
        best_name = summary.sort_values("avg_pnl", ascending=False).iloc[0]["strategy"]
        best = all_df[all_df["variant"] == best_name]
        sym = best.groupby("symbol")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).head(30)
        print(f"Best variant: {best_name}")
        print(sym.to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n=== Recent trades, best variant ===")
        cols = ["session_date", "symbol", "entry_time", "pnl_pct", "max_pnl_pct", "min_pnl_pct", "reason", "intraday_high_pct", "gap_pct", "first_15m_high_pct"]
        print(best[cols].tail(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation hints:")
    print("- close_exit shows whether these clean momentum days trend until the close.")
    print("- trail_only shows whether simple trailing captures enough of the move.")
    print("- staged tests the practical idea: take partial profit on a large move and let the rest run.")
    print("- If avg_min_pnl is large negative, position sizing must be smaller than reversal strategy sizing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
