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


def checkpoint_truncate(store: SQLiteRuntimeStore) -> None:
    try:
        store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass


def cleanup_runtime_events(
    store: SQLiteRuntimeStore,
    *,
    older_than_days: int = 14,
    apply: bool = False,
    batch_size: int = 5000,
    checkpoint_every_batches: int = 1,
) -> dict:
    cutoff = cutoff_date(older_than_days)
    before = store.query("SELECT COUNT(*) AS n FROM runtime_events WHERE substr(event_time, 1, 10) < ?", [cutoff])[0]["n"]
    deleted = 0
    if apply and before:
        batch = 0
        while True:
            rows = store.query(
                """
                SELECT rowid AS runtime_event_rowid
                FROM runtime_events
                WHERE substr(event_time, 1, 10) < ?
                ORDER BY rowid
                LIMIT ?
                """,
                [cutoff, max(1, int(batch_size))],
            )
            rowids = [int(row["runtime_event_rowid"]) for row in rows]
            if not rowids:
                break
            placeholders = ",".join("?" for _ in rowids)
            store.execute(f"DELETE FROM runtime_events WHERE rowid IN ({placeholders})", rowids)
            deleted += len(rowids)
            batch += 1
            if checkpoint_every_batches > 0 and batch % checkpoint_every_batches == 0:
                checkpoint_truncate(store)
            print(json.dumps({"deleted": deleted, "remaining_estimate": max(0, int(before or 0) - deleted)}), flush=True)
        checkpoint_truncate(store)
    after = store.query("SELECT COUNT(*) AS n FROM runtime_events WHERE substr(event_time, 1, 10) < ?", [cutoff])[0]["n"]
    return {
        "apply": bool(apply),
        "cutoff_date": cutoff,
        "runtime_events_matching_before": int(before or 0),
        "runtime_events_matching_after": int(after or 0),
        "runtime_events_deleted": int(deleted),
        "ledger_tables_preserved": ["orders", "executions", "trades"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete old runtime_events detail rows while preserving ledger tables.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--older-than-days", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--checkpoint-every-batches", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    try:
        print(json.dumps(
            cleanup_runtime_events(
                store,
                older_than_days=args.older_than_days,
                apply=args.apply,
                batch_size=args.batch_size,
                checkpoint_every_batches=args.checkpoint_every_batches,
            ),
            indent=2,
        ))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
