#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.v62_live_data_recorder import resolved_record_session_date

TEXT_SUFFIXES = {".csv", ".jsonl", ".json"}


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            return list(csv.DictReader(fh))
    except Exception as exc:
        return [{"_load_error": repr(exc)}]


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except Exception as exc:
                rows.append({"_line_no": line_no, "_load_error": repr(exc), "raw_line": line})
                continue
            if isinstance(parsed, dict):
                parsed.setdefault("_line_no", line_no)
                rows.append(parsed)
    except Exception as exc:
        return [{"_load_error": repr(exc)}]
    return rows


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return [{"_load_error": repr(exc)}]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    return [{"value": parsed}]


def rows_for_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        return load_csv_rows(path)
    if path.suffix == ".jsonl":
        return load_jsonl_rows(path)
    if path.suffix == ".json":
        return load_json_rows(path)
    return []


def audit_date(session_dir: Path) -> list[dict[str, Any]]:
    directory_date = session_dir.name
    out: list[dict[str, Any]] = []
    for path in sorted(session_dir.iterdir()):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        rows = rows_for_file(path)
        for idx, row in enumerate(rows, start=1):
            actual = resolved_record_session_date(row, fallback_session_date="")
            mismatch = bool(actual and actual != directory_date)
            if mismatch:
                out.append({
                    "directory_date": directory_date,
                    "file": path.name,
                    "row_number": row.get("_line_no") or idx,
                    "actual_session_date": actual,
                    "symbol": row.get("symbol") or row.get("contract_symbol") or "",
                    "event_type": row.get("event") or row.get("event_type") or row.get("status") or "",
                    "timestamp": row.get("recorded_at") or row.get("event_time") or row.get("timestamp") or row.get("bar_time") or row.get("executed_at") or "",
                    "payload_excerpt": json.dumps(row, ensure_ascii=False, default=str, sort_keys=True)[:1000],
                })
    return out


def session_dirs(root: Path, *, date: str | None, all_dates: bool) -> list[Path]:
    if date:
        return [root / date]
    if all_dates:
        return sorted(p for p in root.iterdir() if p.is_dir() and len(p.name) == 10)
    raise RuntimeError("Specify --date or --all-dates")


def run(args: argparse.Namespace) -> int:
    root = Path(args.recorder_root)
    all_mismatches: list[dict[str, Any]] = []
    files_scanned = 0
    rows_scanned = 0
    for session_dir in session_dirs(root, date=args.date, all_dates=args.all_dates):
        if not session_dir.exists():
            continue
        for path in session_dir.iterdir():
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files_scanned += 1
                rows_scanned += len(rows_for_file(path))
        all_mismatches.extend(audit_date(session_dir))
    output = Path(args.output_csv) if args.output_csv else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        columns = ["directory_date", "file", "row_number", "actual_session_date", "symbol", "event_type", "timestamp", "payload_excerpt"]
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in all_mismatches:
                writer.writerow({k: row.get(k, "") for k in columns})
    by_file = Counter((row["directory_date"], row["file"], row["actual_session_date"]) for row in all_mismatches)
    print(
        f"RECORDER_SESSION_AUDIT_DONE recorder_root={root} files_scanned={files_scanned} rows_scanned={rows_scanned} "
        f"mismatch_rows={len(all_mismatches)} output={output or ''}",
        flush=True,
    )
    for (directory_date, file_name, actual), count in sorted(by_file.items()):
        print(
            f"RECORDER_SESSION_MISMATCH directory_date={directory_date} file={file_name} actual_session_date={actual} rows={count}",
            flush=True,
        )
    return 1 if all_mismatches and args.fail_on_mismatch else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit recorder files for rows written under the wrong session-date directory.")
    parser.add_argument("--recorder-root", default="data/live/recorder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date")
    group.add_argument("--all-dates", action="store_true")
    parser.add_argument("--output-csv")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
