#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, resolve_sqlite_path


def mb(value: int | float | None) -> float:
    return round(float(value or 0) / (1024 * 1024), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SQLite runtime DB row counts and raw_json size.")
    parser.add_argument("--sqlite-path", default=None)
    args = parser.parse_args()

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    try:
        tables = [
            row["name"]
            for row in store.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            if not str(row["name"]).startswith("sqlite_")
        ]
        table_rows = []
        for table in tables:
            count = store.query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
            columns = {row["name"] for row in store.query(f"PRAGMA table_info({table})")}
            raw_mb = 0.0
            if "raw_json" in columns:
                raw_mb = mb(store.query(f"SELECT SUM(LENGTH(COALESCE(raw_json, ''))) AS bytes FROM {table}")[0]["bytes"])
            table_rows.append({"table": table, "rows": int(count or 0), "raw_json_mb": raw_mb})

        event_rows = []
        if "runtime_events" in tables:
            event_rows = store.query(
                """
                SELECT event_type,
                       COUNT(*) AS rows,
                       ROUND(SUM(LENGTH(COALESCE(raw_json, ''))) / 1048576.0, 3) AS raw_json_mb
                FROM runtime_events
                GROUP BY event_type
                ORDER BY rows DESC
                LIMIT 50
                """
            )
        print(json.dumps({"tables": table_rows, "runtime_events_by_type": event_rows}, indent=2, default=str))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
