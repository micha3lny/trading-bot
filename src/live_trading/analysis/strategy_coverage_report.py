from __future__ import annotations

import argparse
import glob
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.live_trading.analysis.common import (
    calculate_runner_stats,
    load_top100,
    load_universe_symbols,
    normalize_symbol,
    read_sql_table,
)
from src.live_trading.analysis.missed_runners_analyzer import (
    DEFAULT_HISTORY_DIR,
    DEFAULT_RECORDER_DIR,
    DEFAULT_SQLITE_PATH,
    DEFAULT_UNIVERSE,
    no_signal_diagnostics,
)
from src.live_trading.ranking.daily_top100_builder import normalize_history_df


DEFAULT_OUTPUT_DIR = Path("data/analysis")
DEFAULT_THRESHOLDS = (5.0, 10.0, 15.0, 20.0)
NEEDED_PARQUET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CoverageProgress:
    date: str
    processed: int
    total: int
    started_at: float

    def log(self) -> None:
        elapsed = time.monotonic() - self.started_at
        print(f"COVERAGE_PROGRESS date={self.date} processed={self.processed}/{self.total} elapsed={elapsed:.1f}", flush=True)


def threshold_label(threshold: float) -> str:
    value = int(threshold) if float(threshold).is_integer() else threshold
    return str(value).replace(".", "_")


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _is_blank(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(True, index=index)
    return series.fillna("").astype(str).str.strip().eq("")


def dated_history_glob(history_dir: Path, session_date: str, session_type: str = "RTH") -> str:
    d = pd.Timestamp(session_date).date()
    return str(
        Path(history_dir)
        / f"session_type={session_type.upper()}"
        / "symbol=*"
        / f"year={d.year:04d}"
        / f"month={d.month:02d}"
        / f"day={d.day:02d}.parquet"
    )


def symbol_from_parquet_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("symbol="):
            return normalize_symbol(part.split("=", 1)[1])
    return normalize_symbol(path.parent.parent.parent.name.replace("symbol=", ""))


def find_history_files(history_dir: Path, session_date: str, session_type: str = "RTH") -> dict[str, Path]:
    files = sorted(Path(value) for value in glob.glob(dated_history_glob(history_dir, session_date, session_type)))
    out: dict[str, Path] = {}
    for path in files:
        symbol = symbol_from_parquet_path(path)
        if symbol:
            out[symbol] = path
    return out


def read_history_for_coverage(path: Path) -> pd.DataFrame:
    try:
        try:
            raw = pd.read_parquet(path, columns=NEEDED_PARQUET_COLUMNS)
        except Exception:
            raw = pd.read_parquet(path)
        return normalize_history_df(raw)
    except Exception:
        return pd.DataFrame()


def load_bought_symbols(sqlite_path: Path, session_date: str) -> set[str]:
    executions = read_sql_table(
        sqlite_path,
        "executions",
        columns=["symbol", "side", "session_date", "executed_at", "recorded_at"],
        where="session_date = ? OR substr(executed_at, 1, 10) = ? OR substr(recorded_at, 1, 10) = ?",
        params=[session_date, session_date, session_date],
    )
    if executions.empty:
        return set()
    side = executions.get("side", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    buys = executions[side.isin(["BOT", "BUY", "BOUGHT"])]
    return set(buys.get("symbol", pd.Series(dtype=str)).map(normalize_symbol).dropna().tolist())


def load_runtime_evidence_symbols(sqlite_path: Path, session_date: str) -> set[str]:
    symbols: set[str] = set()
    table_specs = {
        "runtime_events": ("event_time", ["symbol", "event_time", "raw_json"]),
        "risk_events": ("event_time", ["symbol", "event_time", "raw_json"]),
        "orders": ("created_at", ["symbol", "created_at", "updated_at", "raw_json"]),
        "trades": ("session_date", ["symbol", "session_date", "entry_fill_time", "exit_fill_time", "raw_json"]),
        "executions": ("session_date", ["symbol", "session_date", "executed_at", "recorded_at", "raw_json"]),
    }
    for table, (time_col, columns) in table_specs.items():
        try:
            if time_col == "session_date":
                where = "session_date = ? OR substr(entry_fill_time, 1, 10) = ? OR substr(exit_fill_time, 1, 10) = ?"
                params = [session_date, session_date, session_date]
            else:
                where = f"substr({time_col}, 1, 10) = ?"
                params = [session_date]
            frame = read_sql_table(sqlite_path, table, columns=columns, where=where, params=params)
        except (sqlite3.Error, Exception):
            continue
        if frame.empty or "symbol" not in frame.columns:
            continue
        symbols.update(frame["symbol"].map(normalize_symbol).dropna().tolist())
    return {symbol for symbol in symbols if symbol}


def build_runner_rows(
    *,
    session_date: str,
    history_dir: Path,
    universe_path: Path,
    top100_path: Path,
    sqlite_path: Path,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    max_symbols: int | None = None,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, dict[str, int]]:
    started = time.monotonic()
    universe_symbols = set(load_universe_symbols(universe_path))
    top100 = load_top100(top100_path)
    top100_symbols = set(top100.get("symbol", pd.Series(dtype=str)).map(normalize_symbol).dropna().tolist()) if not top100.empty else set()
    bought_symbols = load_bought_symbols(sqlite_path, session_date)
    runtime_evidence_symbols = load_runtime_evidence_symbols(sqlite_path, session_date)
    history_files = find_history_files(history_dir, session_date)
    symbols = sorted(symbol for symbol in universe_symbols if symbol in history_files)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    print(
        f"COVERAGE_START date={session_date} universe_files_found={len(history_files)} "
        f"top100_symbols={len(top100_symbols)} symbols_to_process={len(symbols)}",
        flush=True,
    )
    rows: list[dict[str, object]] = []
    base_threshold = min(float(value) for value in thresholds)
    for idx, symbol in enumerate(symbols, start=1):
        candles = read_history_for_coverage(history_files[symbol])
        stats = calculate_runner_stats(candles)
        if stats is not None and stats.open_to_high_pct >= base_threshold:
            no_signal = no_signal_diagnostics(candles) if symbol in top100_symbols and symbol not in bought_symbols else {}
            should_have = no_signal.get("top100_no_signal_reason") == "should_have_signaled"
            rows.append({
                "date": session_date,
                "symbol": symbol,
                "source_bucket": "top100" if symbol in top100_symbols else "outside_top100",
                "open_to_high_pct": stats.open_to_high_pct,
                "was_bought": int(symbol in bought_symbols),
                "was_detectable_from_history": int(should_have),
                "top100_no_signal_reason": no_signal.get("top100_no_signal_reason", ""),
                "signal_time": "",
                "ready_since": "",
                "blocked_reason": "",
                "rejection_reason": "",
                "runtime_evidence": int(symbol in runtime_evidence_symbols),
            })
        if idx % progress_every == 0:
            CoverageProgress(session_date, idx, len(symbols), started).log()
    if symbols and len(symbols) % progress_every:
        CoverageProgress(session_date, len(symbols), len(symbols), started).log()
    diagnostics = {
        "universe_total_symbols": len(universe_symbols),
        "universe_files_found": len(history_files),
        "top100_total_symbols": len(top100_symbols),
        "bought_symbols": len(bought_symbols),
        "runtime_evidence_symbols": len(runtime_evidence_symbols),
        "processed_symbols": len(symbols),
    }
    return pd.DataFrame(rows), diagnostics


def summarize_coverage_from_missed(
    missed: pd.DataFrame,
    *,
    session_date: str,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    """Summarize strategy coverage from runner rows.

    Compatible with analyze_missed_runners output and the optimized coverage
    runner rows produced by this module.
    """
    thresholds = tuple(float(value) for value in thresholds)
    df = missed.copy()
    row: dict[str, object] = {
        "date": session_date,
        "runner_threshold_base_pct": min(thresholds),
        "universe_total_symbols": "",
        "top100_total_symbols": "",
    }
    if df.empty:
        for threshold in thresholds:
            label = threshold_label(float(threshold))
            row[f"universe_gt_{label}"] = 0
            row[f"universe_runner_count_gt_{label}"] = 0
            row[f"top100_gt_{label}"] = 0
            row[f"top100_runner_count_gt_{label}"] = 0
            row[f"coverage_gt_{label}_pct"] = 0.0
            row[f"bought_gt_{label}"] = 0
            row[f"bought_runner_count_gt_{label}"] = 0
            row[f"capture_gt_{label}_pct"] = 0.0
            row[f"missed_gt_{label}"] = 0
        row.update({
            "missed_detectable": 0,
            "missed_undetectable": 0,
            "missed_should_have_signaled": 0,
            "missed_runtime_missing": 0,
        })
        return row

    open_to_high = _numeric(df.get("open_to_high_pct")).fillna(-1.0)
    was_bought = _numeric(df.get("was_bought")).fillna(0).astype(int)
    source_bucket = df.get("source_bucket", pd.Series("", index=df.index)).fillna("").astype(str)
    for threshold in thresholds:
        label = threshold_label(float(threshold))
        above = open_to_high >= float(threshold)
        universe_count = int(above.sum())
        top100_count = int((above & source_bucket.eq("top100")).sum())
        bought_count = int((above & was_bought.eq(1)).sum())
        row[f"universe_gt_{label}"] = universe_count
        row[f"universe_runner_count_gt_{label}"] = universe_count
        row[f"top100_gt_{label}"] = top100_count
        row[f"top100_runner_count_gt_{label}"] = top100_count
        row[f"coverage_gt_{label}_pct"] = round((top100_count / universe_count * 100.0), 2) if universe_count else 0.0
        row[f"bought_gt_{label}"] = bought_count
        row[f"bought_runner_count_gt_{label}"] = bought_count
        row[f"capture_gt_{label}_pct"] = round((bought_count / top100_count * 100.0), 2) if top100_count else 0.0
        row[f"missed_gt_{label}"] = universe_count - bought_count

    base = open_to_high >= min(thresholds)
    missed_mask = base & was_bought.ne(1)
    detectable = _numeric(df.get("was_detectable_from_history")).fillna(0).astype(int).eq(1)
    top100_no_signal_reason = df.get("top100_no_signal_reason", pd.Series("", index=df.index)).fillna("").astype(str)
    signal_time_blank = _is_blank(df.get("signal_time"), df.index)
    ready_since_blank = _is_blank(df.get("ready_since"), df.index)
    blocked_blank = _is_blank(df.get("blocked_reason"), df.index)
    rejected_blank = _is_blank(df.get("rejection_reason"), df.index)
    if "runtime_evidence" in df.columns:
        runtime_evidence = _numeric(df.get("runtime_evidence")).fillna(0).astype(int).eq(1)
    else:
        runtime_evidence = pd.Series(False, index=df.index)
    should_have = top100_no_signal_reason.eq("should_have_signaled")

    row["missed_detectable"] = int((missed_mask & detectable).sum())
    row["missed_undetectable"] = int((missed_mask & ~detectable).sum())
    row["missed_should_have_signaled"] = int((missed_mask & should_have).sum())
    row["missed_runtime_missing"] = int((
        missed_mask
        & should_have
        & ~runtime_evidence
        & signal_time_blank
        & ready_since_blank
        & blocked_blank
        & rejected_blank
    ).sum())
    return row


def build_coverage_report(
    *,
    session_date: str,
    history_dir: Path = DEFAULT_HISTORY_DIR,
    universe_path: Path = DEFAULT_UNIVERSE,
    top100_path: Path | None = None,
    sqlite_path: Path = DEFAULT_SQLITE_PATH,
    recorder_dir: Path = DEFAULT_RECORDER_DIR,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    max_symbols: int | None = None,
) -> pd.DataFrame:
    del recorder_dir
    thresholds = tuple(float(value) for value in thresholds)
    top100 = top100_path or Path(f"data/universe/daily_top100_{session_date}.csv")
    runners, diagnostics = build_runner_rows(
        session_date=session_date,
        history_dir=history_dir,
        universe_path=universe_path,
        top100_path=top100,
        sqlite_path=sqlite_path,
        thresholds=thresholds,
        max_symbols=max_symbols,
    )
    summary = summarize_coverage_from_missed(runners, session_date=session_date, thresholds=thresholds)
    summary.update(diagnostics)
    return pd.DataFrame([summary])


def report_path(output_dir: Path, session_date: str) -> Path:
    return output_dir / f"coverage_report_{session_date}.csv"


def update_coverage_history(output_dir: Path, report: pd.DataFrame) -> Path:
    path = output_dir / "coverage_history.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, report], ignore_index=True)
    combined = combined.drop_duplicates("date", keep="last").sort_values("date")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return path


def write_coverage_report(
    *,
    session_date: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output: Path | None = None,
    force: bool = False,
    slow_seconds: float = 120.0,
    **kwargs: object,
) -> Path:
    path = output or report_path(output_dir, session_date)
    if path.exists() and not force:
        print(f"COVERAGE_SKIPPED_EXISTING date={session_date} output={path}", flush=True)
        return path
    started = time.monotonic()
    df = build_coverage_report(session_date=session_date, **kwargs)
    elapsed = time.monotonic() - started
    if elapsed > slow_seconds:
        print(f"COVERAGE_SLOW_DATE date={session_date} elapsed={elapsed:.1f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    history_path = update_coverage_history(output_dir, df)
    processed = int(df.iloc[0].get("processed_symbols", 0) or 0) if not df.empty else 0
    print(f"COVERAGE_DONE date={session_date} processed_symbols={processed} elapsed_seconds={elapsed:.1f} output={path} history={history_path}", flush=True)
    return path


def iter_dates(start_date: str, end_date: str) -> Iterable[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    step = timedelta(days=1)
    current: date = start
    while current <= end:
        yield current.isoformat()
        current += step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build daily Strategy Coverage KPI reports.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Session date, YYYY-MM-DD. Shortcut for --start-date DATE --end-date DATE.")
    group.add_argument("--start-date", help="Start date for an inclusive date range, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for --start-date range, YYYY-MM-DD.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--top100", type=Path, default=None, help="Top100 CSV for single-date runs. Defaults to daily_top100_DATE.csv.")
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Single-date output path override.")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit processed symbols for quick testing.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing coverage_report_YYYY-MM-DD.csv.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dates = [args.date] if args.date else list(iter_dates(args.start_date, args.end_date or args.start_date))
    written: list[Path] = []
    for session_date in dates:
        output = args.output if args.date else None
        path = write_coverage_report(
            session_date=session_date,
            output_dir=args.output_dir,
            output=output,
            force=args.force,
            history_dir=args.history_dir,
            universe_path=args.universe,
            top100_path=args.top100 if args.date else None,
            sqlite_path=args.sqlite_path,
            recorder_dir=args.recorder_dir,
            max_symbols=args.max_symbols,
        )
        written.append(path)
        print(f"STRATEGY_COVERAGE_REPORT_WRITTEN date={session_date} output={path}", flush=True)
    if len(written) > 1:
        frames = [pd.read_csv(path) for path in written if path.exists()]
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined_path = args.output_dir / f"coverage_report_{dates[0]}_{dates[-1]}.csv"
            combined.to_csv(combined_path, index=False)
            print(f"STRATEGY_COVERAGE_REPORT_COMBINED output={combined_path} rows={len(combined)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
