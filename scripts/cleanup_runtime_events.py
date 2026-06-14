#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, resolve_sqlite_path


def cutoff_date(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=max(0, days))).strftime("%F")


def cleanup_runtime_events(store: SQLiteRuntimeStore, *, older_than_days: int = 14, apply: bool = False) -> dict:
    cutoff = cutoff_date(older_than_days)
    before = store.query("SELECT COUNT(*) AS n FROM runtime_events WHERE substr(event_time, 1, 10) < ?", [cutoff])[0]["n"]
    if apply and before:
        with store.transaction():
            store.execute("DELETE FROM runtime_events WHERE substr(event_time, 1, 10) < ?", [cutoff])
    after = store.query("SELECT COUNT(*) AS n FROM runtime_events WHERE substr(event_time, 1, 10) < ?", [cutoff])[0]["n"]
    return {
        "apply": bool(apply),
        "cutoff_date": cutoff,
        "runtime_events_matching_before": int(before or 0),
        "runtime_events_matching_after": int(after or 0),
        "ledger_tables_preserved": ["orders", "executions", "trades"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete old runtime_events detail rows while preserving ledger tables.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--older-than-days", type=int, default=14)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    try:
        print(json.dumps(cleanup_runtime_events(store, older_than_days=args.older_than_days, apply=args.apply), indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
