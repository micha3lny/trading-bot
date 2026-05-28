#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, resolve_sqlite_path


def print_rows(rows: list[dict[str, Any]]) -> None:
    print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description="Query live runtime SQLite diagnostics.")
    ap.add_argument("--sqlite-path", default=None)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("open-trades")
    sub.add_parser("unresolved-risk")
    p = sub.add_parser("daily-summary")
    p.add_argument("--date", required=True)
    p = sub.add_parser("executions")
    p.add_argument("--date", required=True)
    p = sub.add_parser("reconciliation")
    p.add_argument("--date", required=True)
    p = sub.add_parser("peak-diagnostics")
    p.add_argument("--date", required=True)
    sub.add_parser("collector-runs")
    args = ap.parse_args()

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    try:
        if args.command == "open-trades":
            print_rows(store.get_open_trades())
        elif args.command == "unresolved-risk":
            print_rows(store.get_unresolved_risk_events())
        elif args.command == "daily-summary":
            print_rows(store.query(
                """
                SELECT
                    ? AS session_date,
                    (SELECT COUNT(*) FROM trades WHERE session_date = ?) AS trades,
                    (SELECT COUNT(*) FROM executions WHERE session_date = ?) AS executions,
                    (SELECT COUNT(*) FROM risk_events WHERE session_date = ?) AS risk_events,
                    (SELECT COUNT(*) FROM reconciliation_runs WHERE substr(started_at, 1, 10) = ?) AS reconciliation_runs
                """,
                [args.date, args.date, args.date, args.date, args.date],
            ))
        elif args.command == "executions":
            print_rows(store.query("SELECT * FROM executions WHERE session_date = ? ORDER BY executed_at, recorded_at", [args.date]))
        elif args.command == "reconciliation":
            print_rows(store.query("SELECT * FROM reconciliation_runs WHERE substr(started_at, 1, 10) = ? OR substr(finished_at, 1, 10) = ? ORDER BY started_at", [args.date, args.date]))
        elif args.command == "peak-diagnostics":
            print_rows(store.query(
                """
                SELECT
                    ? AS session_date,
                    (SELECT COUNT(*) FROM trades WHERE session_date = ? AND UPPER(COALESCE(status, '')) IN ('CLOSED', 'DONE', 'EXIT_FILLED', 'FLAT')) AS closed_trades,
                    (SELECT COUNT(*) FROM trades WHERE session_date = ? AND UPPER(COALESCE(status, '')) IN ('CLOSED', 'DONE', 'EXIT_FILLED', 'FLAT') AND mfe_pct IS NOT NULL) AS trades_with_mfe,
                    (SELECT COUNT(*) FROM runtime_events WHERE COALESCE(session_date, substr(event_time, 1, 10)) = ? AND (UPPER(event_type) LIKE '%PEAK%' OR raw_json LIKE '%peak_price%' OR raw_json LIKE '%mfe%')) AS peak_events,
                    (SELECT COUNT(DISTINCT symbol) FROM runtime_events WHERE COALESCE(session_date, substr(event_time, 1, 10)) = ? AND (UPPER(event_type) LIKE '%PEAK%' OR raw_json LIKE '%peak_price%' OR raw_json LIKE '%mfe%')) AS peak_symbols
                """,
                [args.date, args.date, args.date, args.date, args.date],
            ))
        elif args.command == "collector-runs":
            print_rows(store.query("SELECT * FROM collector_runs ORDER BY started_at DESC LIMIT 100"))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
