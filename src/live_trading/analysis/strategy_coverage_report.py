from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.live_trading.analysis.missed_runners_analyzer import (
    DEFAULT_HISTORY_DIR,
    DEFAULT_RECORDER_DIR,
    DEFAULT_SQLITE_PATH,
    DEFAULT_UNIVERSE,
    analyze_missed_runners,
)


DEFAULT_OUTPUT_DIR = Path("data/analysis")
DEFAULT_THRESHOLDS = (5.0, 10.0, 15.0, 20.0)


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


def summarize_coverage_from_missed(
    missed: pd.DataFrame,
    *,
    session_date: str,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    """Summarize strategy coverage from missed-runners rows.

    The input is expected to be the same shape as analyze_missed_runners(...,
    threshold_pct=min(thresholds)). It may contain extra rows; each threshold is
    applied independently.
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
            row[f"top100_gt_{label}"] = 0
            row[f"coverage_gt_{label}_pct"] = 0.0
            row[f"bought_gt_{label}"] = 0
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
        row[f"top100_gt_{label}"] = top100_count
        row[f"coverage_gt_{label}_pct"] = round((top100_count / universe_count * 100.0), 2) if universe_count else 0.0
        row[f"bought_gt_{label}"] = bought_count
        row[f"missed_gt_{label}"] = universe_count - bought_count

    base = open_to_high >= min(thresholds)
    missed_mask = base & was_bought.ne(1)
    detectable = _numeric(df.get("was_detectable_from_history")).fillna(0).astype(int).eq(1)
    top100_no_signal_reason = df.get("top100_no_signal_reason", pd.Series("", index=df.index)).fillna("").astype(str)
    signal_time_blank = _is_blank(df.get("signal_time"), df.index)
    ready_since_blank = _is_blank(df.get("ready_since"), df.index)
    blocked_blank = _is_blank(df.get("blocked_reason"), df.index)
    rejected_blank = _is_blank(df.get("rejection_reason"), df.index)
    should_have = top100_no_signal_reason.eq("should_have_signaled")

    row["missed_detectable"] = int((missed_mask & detectable).sum())
    row["missed_undetectable"] = int((missed_mask & ~detectable).sum())
    row["missed_should_have_signaled"] = int((missed_mask & should_have).sum())
    row["missed_runtime_missing"] = int((
        missed_mask
        & should_have
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
) -> pd.DataFrame:
    thresholds = tuple(float(value) for value in thresholds)
    top100 = top100_path or Path(f"data/universe/daily_top100_{session_date}.csv")
    missed = analyze_missed_runners(
        session_date=session_date,
        history_dir=history_dir,
        universe_path=universe_path,
        top100_path=top100,
        sqlite_path=sqlite_path,
        recorder_dir=recorder_dir,
        threshold_pct=min(thresholds),
    )
    summary = summarize_coverage_from_missed(missed, session_date=session_date, thresholds=thresholds)
    summary["universe_total_symbols"] = len(pd.read_csv(universe_path)) if universe_path.exists() else ""
    summary["top100_total_symbols"] = len(pd.read_csv(top100)) if top100.exists() else ""
    return pd.DataFrame([summary])


def report_path(output_dir: Path, session_date: str) -> Path:
    return output_dir / f"coverage_report_{session_date}.csv"


def write_coverage_report(
    *,
    session_date: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output: Path | None = None,
    **kwargs: object,
) -> Path:
    df = build_coverage_report(session_date=session_date, **kwargs)
    path = output or report_path(output_dir, session_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
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
    group.add_argument("--date", help="Session date, YYYY-MM-DD.")
    group.add_argument("--start-date", help="Start date for an inclusive date range, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for --start-date range, YYYY-MM-DD.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--top100", type=Path, default=None, help="Top100 CSV for single-date runs. Defaults to daily_top100_DATE.csv.")
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--recorder-dir", type=Path, default=DEFAULT_RECORDER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Single-date output path override.")
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
            history_dir=args.history_dir,
            universe_path=args.universe,
            top100_path=args.top100 if args.date else None,
            sqlite_path=args.sqlite_path,
            recorder_dir=args.recorder_dir,
        )
        written.append(path)
        print(f"STRATEGY_COVERAGE_REPORT_WRITTEN date={session_date} output={path}")
    if len(written) > 1:
        combined = pd.concat([pd.read_csv(path) for path in written], ignore_index=True)
        combined_path = args.output_dir / f"coverage_report_{dates[0]}_{dates[-1]}.csv"
        combined.to_csv(combined_path, index=False)
        print(f"STRATEGY_COVERAGE_REPORT_COMBINED output={combined_path} rows={len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
