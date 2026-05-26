#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any

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
        elif args.command == "collector-runs":
            print_rows(store.query("SELECT * FROM collector_runs ORDER BY started_at DESC LIMIT 100"))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
