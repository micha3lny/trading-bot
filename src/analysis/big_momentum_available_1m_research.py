from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_DIR = Path("data/1m")
DEFAULT_OUTPUT_DIR = Path("data/backtests")


@dataclass(frozen=True)
class EntryResult:
    found: bool
    entry_time: str | None = None
    entry_price: float | None = None
    max_pnl_pct: float | None = None
    min_pnl_pct: float | None = None
    close_pnl_pct: float | None = None
    trail_10_07_pnl_pct: float | None = None
    trail_15_10_pnl_pct: float | None = None
    trail_25_15_pnl_pct: float | None = None


def _find_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _load_1m_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    dt_col = _find_col(df.columns, ["datetime", "date", "timestamp", "time"])
    open_col = _find_col(df.columns, ["open", "o"])
    high_col = _find_col(df.columns, ["high", "h"])
    low_col = _find_col(df.columns, ["low", "l"])
    close_col = _find_col(df.columns, ["close", "c"])
    volume_col = _find_col(df.columns, ["volume", "v"])

    required = [dt_col, open_col, high_col, low_col, close_col]
    if any(col is None for col in required):
        return None

    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(df[dt_col], errors="coerce"),
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df[volume_col], errors="coerce") if volume_col else 0,
        }
    )
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    if out.empty:
        return None

    out = out.sort_values("datetime").drop_duplicates("datetime")
    out["session_date"] = out["datetime"].dt.date
    return out


def _simulate_trailing(bars: pd.DataFrame, entry_price: float, activation_pct: float, trail_pct: float) -> float:
    activated = False
    peak = entry_price
    for _, bar in bars.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        if high > peak:
            peak = high
        if not activated and (peak / entry_price - 1.0) * 100.0 >= activation_pct:
            activated = True
        if activated:
            stop_price = peak * (1.0 - trail_pct / 100.0)
            if low <= stop_price:
                return (stop_price / entry_price - 1.0) * 100.0
    return (float(bars.iloc[-1]["close"]) / entry_price - 1.0) * 100.0


def _simulate_or_breakout(day: pd.DataFrame, opening_minutes: int) -> EntryResult:
    if len(day) <= opening_minutes + 1:
        return EntryResult(found=False)

    opening = day.iloc[:opening_minutes]
    rest = day.iloc[opening_minutes:]
    or_high = float(opening["high"].max())

    breakout_rows = rest[rest["high"] > or_high]
    if breakout_rows.empty:
        return EntryResult(found=False)

    first_idx = breakout_rows.index[0]
    entry_row_pos = day.index.get_loc(first_idx)
    post_entry = day.iloc[entry_row_pos:]
    entry_price = or_high

    if post_entry.empty or entry_price <= 0:
        return EntryResult(found=False)

    max_pnl = (float(post_entry["high"].max()) / entry_price - 1.0) * 100.0
    min_pnl = (float(post_entry["low"].min()) / entry_price - 1.0) * 100.0
    close_pnl = (float(post_entry.iloc[-1]["close"]) / entry_price - 1.0) * 100.0

    return EntryResult(
        found=True,
        entry_time=str(pd.Timestamp(post_entry.iloc[0]["datetime"])),
        entry_price=entry_price,
        max_pnl_pct=max_pnl,
        min_pnl_pct=min_pnl,
        close_pnl_pct=close_pnl,
        trail_10_07_pnl_pct=_simulate_trailing(post_entry, entry_price, 1.0, 0.7),
        trail_15_10_pnl_pct=_simulate_trailing(post_entry, entry_price, 1.5, 1.0),
        trail_25_15_pnl_pct=_simulate_trailing(post_entry, entry_price, 2.5, 1.5),
    )


def _day_shape(symbol: str, day: pd.DataFrame, prev_close: float | None) -> dict[str, object]:
    open_price = float(day.iloc[0]["open"])
    close_price = float(day.iloc[-1]["close"])
    high_price = float(day["high"].max())
    low_price = float(day["low"].min())
    high_idx = day["high"].idxmax()
    high_pos = day.index.get_loc(high_idx)
    high_time = pd.Timestamp(day.loc[high_idx, "datetime"])
    start_time = pd.Timestamp(day.iloc[0]["datetime"])

    first_5 = day.iloc[:5]
    first_15 = day.iloc[:15]
    first_30 = day.iloc[:30]

    or5 = _simulate_or_breakout(day, 5)
    or15 = _simulate_or_breakout(day, 15)
    or30 = _simulate_or_breakout(day, 30)

    return {
        "session_date": str(day.iloc[0]["session_date"]),
        "symbol": symbol,
        "rows_1m": len(day),
        "gap_pct": ((open_price / prev_close - 1.0) * 100.0) if prev_close and prev_close > 0 else None,
        "intraday_high_pct": (high_price / open_price - 1.0) * 100.0,
        "intraday_low_pct": (low_price / open_price - 1.0) * 100.0,
        "daily_return_pct": ((close_price / prev_close - 1.0) * 100.0) if prev_close and prev_close > 0 else None,
        "open_to_close_pct": (close_price / open_price - 1.0) * 100.0,
        "time_to_high_minutes": (high_time - start_time).total_seconds() / 60.0,
        "first_5m_high_pct": (float(first_5["high"].max()) / open_price - 1.0) * 100.0,
        "first_15m_high_pct": (float(first_15["high"].max()) / open_price - 1.0) * 100.0,
        "first_30m_high_pct": (float(first_30["high"].max()) / open_price - 1.0) * 100.0,
        "or5_found": or5.found,
        "or5_entry_time": or5.entry_time,
        "or5_max_pnl_pct": or5.max_pnl_pct,
        "or5_min_pnl_pct": or5.min_pnl_pct,
        "or5_close_pnl_pct": or5.close_pnl_pct,
        "or5_trail_10_07_pnl_pct": or5.trail_10_07_pnl_pct,
        "or5_trail_15_10_pnl_pct": or5.trail_15_10_pnl_pct,
        "or5_trail_25_15_pnl_pct": or5.trail_25_15_pnl_pct,
        "or15_found": or15.found,
        "or15_max_pnl_pct": or15.max_pnl_pct,
        "or15_min_pnl_pct": or15.min_pnl_pct,
        "or15_close_pnl_pct": or15.close_pnl_pct,
        "or30_found": or30.found,
        "or30_max_pnl_pct": or30.max_pnl_pct,
        "or30_min_pnl_pct": or30.min_pnl_pct,
        "or30_close_pnl_pct": or30.close_pnl_pct,
    }


def build_available_1m_opportunities(data_dir: Path, recent_days: int | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    files = sorted(data_dir.glob("*.csv"))
    latest_seen: pd.Timestamp | None = None

    loaded: list[tuple[str, pd.DataFrame]] = []
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
            rows.append(_day_shape(symbol, day.reset_index(drop=True), prev_close))
            prev_close = float(day.iloc[-1]["close"])

    return pd.DataFrame(rows)


def _print_metric_block(name: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n--- {name} ---")
    if df.empty:
        print("No rows")
        return
    for col in cols:
        if col in df.columns:
            print(f"avg {col}: {df[col].mean():.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze big momentum opportunities only from available local 1m candles.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recent-days", type=int, default=90, help="Filter by last N calendar days relative to newest local 1m candle.")
    parser.add_argument("--min-intraday-high", type=float, default=5.0)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum 1m rows required for a full-ish RTH session.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Available 1m big momentum research")
    print(f"Data dir: {data_dir}")
    print(f"Recent days: {args.recent_days}")
    print(f"Opportunity filter: intraday high >= {args.min_intraday_high:.2f}%")
    print(f"Top opportunities: {args.top}")

    all_days = build_available_1m_opportunities(data_dir, args.recent_days)
    if all_days.empty:
        print("No local 1m sessions found.")
        return 1

    full_days = all_days[all_days["rows_1m"] >= args.min_rows].copy()
    opps = full_days[full_days["intraday_high_pct"] >= args.min_intraday_high].copy()
    opps = opps.sort_values("intraday_high_pct", ascending=False).head(args.top)

    suffix = f"recent{args.recent_days}_intraday_ge_{int(args.min_intraday_high)}_top{args.top}"
    out_all = output_dir / f"big_momentum_available_1m_all_days_{suffix}.csv"
    out_opps = output_dir / f"big_momentum_available_1m_opportunities_{suffix}.csv"
    all_days.to_csv(out_all, index=False)
    opps.to_csv(out_opps, index=False)

    print(f"\nSaved all 1m day stats CSV: {out_all}")
    print(f"Saved opportunity CSV: {out_opps}")
    print("\n=== Coverage from local 1m files only ===")
    print(f"1m files found: {len(list(data_dir.glob('*.csv')))}")
    print(f"Analyzed sessions: {len(all_days)}")
    print(f"Full-ish sessions rows>={args.min_rows}: {len(full_days)}")
    print(f"Opportunities found: {len(opps)}")

    if opps.empty:
        print("\nNo opportunities matched the filter in available 1m data.")
        return 0

    print("\n=== Big momentum day shape from available 1m data ===")
    for col in [
        "gap_pct",
        "intraday_high_pct",
        "open_to_close_pct",
        "daily_return_pct",
        "time_to_high_minutes",
        "first_5m_high_pct",
        "first_15m_high_pct",
        "first_30m_high_pct",
    ]:
        if col in opps.columns:
            print(f"{col}: avg={opps[col].mean():.2f}, median={opps[col].median():.2f}")

    print("\n=== Diagnostic entry simulations ===")
    _print_metric_block("or5_breakout", opps[opps["or5_found"] == True], ["or5_max_pnl_pct", "or5_min_pnl_pct", "or5_close_pnl_pct", "or5_trail_10_07_pnl_pct", "or5_trail_15_10_pnl_pct", "or5_trail_25_15_pnl_pct"])
    _print_metric_block("or15_breakout", opps[opps["or15_found"] == True], ["or15_max_pnl_pct", "or15_min_pnl_pct", "or15_close_pnl_pct"])
    _print_metric_block("or30_breakout", opps[opps["or30_found"] == True], ["or30_max_pnl_pct", "or30_min_pnl_pct", "or30_close_pnl_pct"])

    print("\n=== Symbols with most available >= threshold moves ===")
    print(opps["symbol"].value_counts().head(30).to_string())

    print("\n=== Top available opportunities ===")
    cols = [
        "session_date",
        "symbol",
        "rows_1m",
        "gap_pct",
        "intraday_high_pct",
        "open_to_close_pct",
        "time_to_high_minutes",
        "first_5m_high_pct",
        "first_15m_high_pct",
        "first_30m_high_pct",
        "or5_max_pnl_pct",
        "or5_min_pnl_pct",
        "or5_close_pnl_pct",
        "or5_trail_25_15_pnl_pct",
    ]
    existing_cols = [c for c in cols if c in opps.columns]
    print(opps[existing_cols].head(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nInterpretation:")
    print("- This script ignores old daily-only opportunities and uses only local data/1m files.")
    print("- If this produces many rows, your backfill is useful already.")
    print("- Use the output CSV as the real 90-day learning set for the momentum-continuation strategy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
