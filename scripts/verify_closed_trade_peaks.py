#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_trading.analysis.common import fnum, load_session_candles, parse_dt


DEFAULT_HISTORY_DIR = Path("data/history/universe_1m")
DEFAULT_OUTPUT_DIR = Path("data/analysis")


@dataclass(frozen=True)
class ColumnMap:
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: str
    exit_price: str
    peak_pct: str
    final_pnl_pct: str | None = None


def parse_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def normalized_column_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("%", "pct")
        .replace("$", "dollars")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_norm = {normalized_column_name(col): col for col in columns}
    for candidate in candidates:
        hit = by_norm.get(normalized_column_name(candidate))
        if hit is not None:
            return hit
    return None


def build_column_map(df: pd.DataFrame) -> ColumnMap:
    columns = list(df.columns)
    symbol = find_column(columns, ["symbol", "Symbol"])
    entry_time = find_column(columns, ["entry_time", "Entry Time", "entry_fill_time", "entry_datetime"])
    exit_time = find_column(columns, ["exit_time", "Exit Time", "closed_at", "exit_fill_time", "exit_datetime"])
    entry_price = find_column(columns, ["buy", "Buy", "entry_price", "Entry Price", "avg_entry_price"])
    exit_price = find_column(columns, ["sell", "Sell", "exit_price", "Exit Price", "avg_exit_price"])
    peak_pct = find_column(columns, ["peak_pct", "Peak %", "mfe_pct", "MFE %", "peak_unrealized_pct"])
    final_pnl_pct = find_column(columns, ["net_pct", "Net %", "final_pnl_pct", "pnl_pct", "PnL %"])
    missing = [
        name
        for name, value in {
            "symbol": symbol,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "peak_pct": peak_pct,
        }.items()
        if value is None
    ]
    if missing:
        raise SystemExit(f"missing required export columns: {','.join(missing)} available={','.join(columns)}")
    return ColumnMap(
        symbol=str(symbol),
        entry_time=str(entry_time),
        exit_time=str(exit_time),
        entry_price=str(entry_price),
        exit_price=str(exit_price),
        peak_pct=str(peak_pct),
        final_pnl_pct=str(final_pnl_pct) if final_pnl_pct else None,
    )


def read_export_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"failed to read export CSV {path}: {exc!r}") from exc


def csv_has_peak_columns(path: Path, session_date: str) -> bool:
    if session_date not in path.name:
        return False
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return find_column(df.columns, ["peak_pct", "Peak %", "mfe_pct", "MFE %"]) is not None


def find_export_csv(session_date: str) -> Path | None:
    candidates: list[Path] = []
    for root in (Path("data/analysis"), Path("reports"), Path("data/exports"), Path(".")):
        if not root.exists():
            continue
        try:
            candidates.extend(root.glob(f"*{session_date}*.csv"))
        except Exception:
            continue
    candidates = [path for path in candidates if path.is_file() and csv_has_peak_columns(path, session_date)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def safe_iso(value: Any) -> str:
    parsed = parse_dt(value)
    return "" if parsed is None else parsed.isoformat()


def pct(price: float | None, base: float | None) -> float | None:
    if price is None or base is None or base <= 0:
        return None
    return (float(price) / float(base) - 1.0) * 100.0


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def row_value(row: dict[str, Any], column: str | None) -> Any:
    if not column:
        return None
    return row.get(column)


def verify_row(
    row: dict[str, Any],
    *,
    columns: ColumnMap,
    session_date: str,
    history_dir: Path,
) -> dict[str, Any]:
    symbol = str(row_value(row, columns.symbol) or "").strip().upper()
    entry_time = parse_dt(row_value(row, columns.entry_time))
    exit_time = parse_dt(row_value(row, columns.exit_time))
    entry_price = fnum(row_value(row, columns.entry_price))
    exit_price = fnum(row_value(row, columns.exit_price))
    exported_peak = fnum(row_value(row, columns.peak_pct))
    final_pnl_pct = fnum(row_value(row, columns.final_pnl_pct)) if columns.final_pnl_pct else None
    notes: list[str] = []

    candles = load_session_candles(history_dir, symbol, session_date, "RTH") if symbol else pd.DataFrame()
    candles_found = int(len(candles))
    if candles.empty:
        notes.append("missing_candles")
        window = pd.DataFrame()
    else:
        window = candles.copy()
        if entry_time is not None:
            window = window[window["timestamp"] >= entry_time]
        else:
            notes.append("missing_entry_time")
        if exit_time is not None:
            window = window[window["timestamp"] <= exit_time]
        else:
            notes.append("missing_exit_time")
        if window.empty:
            notes.append("no_candles_between_entry_exit")

    max_high = None
    recalculated_peak = None
    time_of_peak = None
    close_after_peak_to_exit_drawdown_pct = None
    if not window.empty and entry_price is not None and entry_price > 0:
        high_series = pd.to_numeric(window["high"], errors="coerce")
        if high_series.notna().any():
            peak_idx = high_series.idxmax()
            max_high = float(window.loc[peak_idx, "high"])
            time_of_peak = window.loc[peak_idx, "timestamp"]
            recalculated_peak = pct(max_high, entry_price)
            after_peak = window[window["timestamp"] >= time_of_peak]
            if not after_peak.empty and max_high and max_high > 0:
                exit_close = fnum(after_peak.iloc[-1].get("close"))
                close_after_peak_to_exit_drawdown_pct = pct(exit_close, max_high)
    elif entry_price is None or entry_price <= 0:
        notes.append("missing_entry_price")

    full_day_peak = None
    if not candles.empty and entry_price is not None and entry_price > 0:
        full_day_high = fnum(pd.to_numeric(candles["high"], errors="coerce").max())
        full_day_peak = pct(full_day_high, entry_price)

    peak_diff = None
    peak_matches = False
    looks_like_full_day_peak = False
    if exported_peak is not None and recalculated_peak is not None:
        peak_diff = exported_peak - recalculated_peak
        peak_matches = abs(peak_diff) <= 0.5
    if exported_peak is not None and full_day_peak is not None:
        rec_diff = abs(exported_peak - recalculated_peak) if recalculated_peak is not None else float("inf")
        full_diff = abs(exported_peak - full_day_peak)
        looks_like_full_day_peak = full_diff < rec_diff

    giveback = None
    if recalculated_peak is not None and final_pnl_pct is not None:
        giveback = recalculated_peak - final_pnl_pct

    if exported_peak is None:
        notes.append("missing_exported_peak")
    if recalculated_peak is None:
        notes.append("missing_recalculated_peak")
    if looks_like_full_day_peak and not peak_matches:
        notes.append("suspected_full_day_peak")

    return {
        "date": session_date,
        "symbol": symbol,
        "entry_time": safe_iso(row_value(row, columns.entry_time)),
        "exit_time": safe_iso(row_value(row, columns.exit_time)),
        "entry_price": round_or_none(entry_price),
        "exit_price": round_or_none(exit_price),
        "exported_peak_pct": round_or_none(exported_peak),
        "recalculated_peak_pct": round_or_none(recalculated_peak),
        "peak_diff_pct": round_or_none(peak_diff),
        "peak_matches": int(peak_matches),
        "time_of_peak": "" if time_of_peak is None else pd.Timestamp(time_of_peak).isoformat(),
        "full_day_peak_pct": round_or_none(full_day_peak),
        "looks_like_full_day_peak": int(looks_like_full_day_peak),
        "final_pnl_pct": round_or_none(final_pnl_pct),
        "giveback_pct": round_or_none(giveback),
        "candles_found": candles_found,
        "notes": ";".join(notes),
        "max_high_between_entry_exit": round_or_none(max_high),
        "close_after_peak_to_exit_drawdown_pct": round_or_none(close_after_peak_to_exit_drawdown_pct),
    }


def print_summary(out: pd.DataFrame) -> None:
    trades_checked = len(out)
    matches = int(pd.to_numeric(out.get("peak_matches", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not out.empty else 0
    mismatches = trades_checked - matches
    suspected_full_day = int(pd.to_numeric(out.get("looks_like_full_day_peak", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not out.empty else 0
    print(
        f"PEAK_VERIFY_SUMMARY trades_checked={trades_checked} "
        f"peak_matches_count={matches} peak_mismatch_count={mismatches} "
        f"suspected_full_day_peak_count={suspected_full_day}",
        flush=True,
    )
    if out.empty:
        return
    mism = out.copy()
    mism["_abs_diff"] = pd.to_numeric(mism["peak_diff_pct"], errors="coerce").abs()
    mism = mism.sort_values("_abs_diff", ascending=False).head(10)
    top = ",".join(
        f"{row.symbol}:{row.peak_diff_pct}"
        for row in mism.itertuples()
        if pd.notna(getattr(row, "peak_diff_pct", None))
    )
    print(f"PEAK_VERIFY_TOP_MISMATCHES {top}", flush=True)
    give = out.copy()
    give["_giveback"] = pd.to_numeric(give["giveback_pct"], errors="coerce")
    give = give[give["peak_matches"].astype(int) == 1].sort_values("_giveback", ascending=False).head(10)
    biggest = ",".join(
        f"{row.symbol}:{row.giveback_pct}"
        for row in give.itertuples()
        if pd.notna(getattr(row, "giveback_pct", None))
    )
    print(f"PEAK_VERIFY_BIGGEST_GIVEBACKS_CONFIRMED {biggest}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify dashboard-exported closed-trade Peak % against 1m candles.")
    parser.add_argument("--date", required=True, type=parse_date)
    parser.add_argument("--export-csv", type=Path, default=None)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--min-peak-pct", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    session_date = args.date
    export_csv = args.export_csv or find_export_csv(session_date)
    if export_csv is None:
        print(
            f"PEAK_VERIFY_FAILED date={session_date} reason=export_csv_not_found "
            f"hint='pass --export-csv path/to/dashboard_export.csv'",
            flush=True,
        )
        return 2

    output = args.output or DEFAULT_OUTPUT_DIR / f"closed_trade_peak_verify_{session_date}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"PEAK_VERIFY_START date={session_date} export_csv={export_csv} "
        f"min_peak_pct={args.min_peak_pct} output={output}",
        flush=True,
    )
    started = time.monotonic()
    export = read_export_csv(export_csv)
    columns = build_column_map(export)
    export["_exported_peak_for_filter"] = pd.to_numeric(export[columns.peak_pct], errors="coerce")
    candidates = export[export["_exported_peak_for_filter"] >= float(args.min_peak_pct)].copy()
    rows: list[dict[str, Any]] = []
    total = len(candidates)
    for idx, row in enumerate(candidates.to_dict("records"), start=1):
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"PEAK_VERIFY_PROGRESS date={session_date} processed={idx}/{total}", flush=True)
        rows.append(verify_row(row, columns=columns, session_date=session_date, history_dir=args.history_dir))

    out = pd.DataFrame(rows)
    ordered_cols = [
        "date",
        "symbol",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "exported_peak_pct",
        "recalculated_peak_pct",
        "peak_diff_pct",
        "peak_matches",
        "time_of_peak",
        "full_day_peak_pct",
        "looks_like_full_day_peak",
        "final_pnl_pct",
        "giveback_pct",
        "candles_found",
        "notes",
        "max_high_between_entry_exit",
        "close_after_peak_to_exit_drawdown_pct",
    ]
    for col in ordered_cols:
        if col not in out.columns:
            out[col] = None
    out = out[ordered_cols]
    out.to_csv(output, index=False)
    print_summary(out)
    print(
        f"PEAK_VERIFY_DONE date={session_date} trades_checked={len(out)} "
        f"elapsed_seconds={time.monotonic() - started:.2f} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
