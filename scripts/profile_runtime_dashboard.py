#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sqlite3
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.runtime_queries import (
    DateWindow,
    capture_dashboard_performance,
    load_dashboard_snapshot,
    process_rss_mb,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Profile the read-only runtime dashboard snapshot builder.")
    value.add_argument("--sqlite-path", type=Path, required=True)
    value.add_argument("--date")
    value.add_argument("--start-date")
    value.add_argument("--end-date")
    value.add_argument("--strategy", default="All")
    value.add_argument("--include-reconstructed", action="store_true")
    value.add_argument("--recorder-root", type=Path)
    value.add_argument("--cprofile-output", type=Path)
    value.add_argument("--output-json", type=Path)
    value.add_argument("--slow-query-ms", type=float, default=100.0)
    return value


def explain_slow_queries(sqlite_path: Path, queries: list[dict[str, Any]], threshold_ms: float) -> list[dict[str, Any]]:
    explained: list[dict[str, Any]] = []
    seen: set[str] = set()
    uri = f"file:{sqlite_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        for query in queries:
            fingerprint = str(query.get("fingerprint") or "")
            if fingerprint in seen or float(query.get("duration_ms") or 0.0) <= threshold_ms:
                continue
            seen.add(fingerprint)
            try:
                plan = [str(row[3]) for row in conn.execute(
                    f"EXPLAIN QUERY PLAN {query['sql']}", query.get("params") or []
                ).fetchall()]
            except Exception as exc:
                plan = [f"EXPLAIN_FAILED: {type(exc).__name__}: {exc}"]
            explained.append({
                "table": query.get("table"),
                "duration_ms": query.get("duration_ms"),
                "rows": query.get("rows"),
                "plan": plan,
            })
    return explained


def main() -> int:
    args = parser().parse_args()
    if args.date:
        start_date = end_date = args.date
    else:
        if not args.start_date or not args.end_date:
            raise SystemExit("Provide --date or both --start-date and --end-date")
        start_date, end_date = args.start_date, args.end_date
    if args.recorder_root:
        os.environ["DASHBOARD_RECORDER_DIR"] = str(args.recorder_root)

    profiler = cProfile.Profile() if args.cprofile_output else None
    rss_before = process_rss_mb()
    tracemalloc.start()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    with capture_dashboard_performance() as trace:
        if profiler is not None:
            profiler.enable()
        snapshot = load_dashboard_snapshot(
            args.sqlite_path,
            DateWindow(start_date, end_date),
            args.strategy,
            include_reconstructed=args.include_reconstructed,
            read_only=True,
        )
        if profiler is not None:
            profiler.disable()
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process_rss_mb()

    if profiler is not None:
        args.cprofile_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.cprofile_output)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)

    performance = trace.summary()
    slow_plans = explain_slow_queries(args.sqlite_path, performance["queries"], args.slow_query_ms)
    frame_rows = {
        key: len(value)
        for key, value in snapshot.items()
        if hasattr(value, "columns") and hasattr(value, "__len__")
    }
    result = {
        "sqlite_path": str(args.sqlite_path.resolve()),
        "start_date": start_date,
        "end_date": end_date,
        "strategy": args.strategy,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "tracemalloc_current_mb": current_bytes / (1024 * 1024),
        "tracemalloc_peak_mb": peak_bytes / (1024 * 1024),
        "snapshot_rows": frame_rows,
        "query_count": performance["query_count"],
        "repeated_identical_query_count": performance["repeated_identical_query_count"],
        "rows_read_by_table": performance["rows_read_by_table"],
        "phase_timings": performance["phases"],
        "slow_query_plans": slow_plans,
    }
    print("DASHBOARD_PROFILE_RESULT " + json.dumps(result, sort_keys=True, default=str))
    for item in slow_plans:
        print("DASHBOARD_SLOW_QUERY_PLAN " + json.dumps(item, sort_keys=True, default=str))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
