#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, resolve_sqlite_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VACUUM on the runtime SQLite DB after cleanup.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    path = Path(resolve_sqlite_path(args.sqlite_path))
    before = path.stat().st_size if path.exists() else 0
    free = shutil.disk_usage(path.parent if path.parent.exists() else Path(".")).free
    if args.apply:
        if before and free < before * 1.2:
            print(json.dumps({
                "apply": False,
                "path": str(path),
                "bytes_before": before,
                "free_bytes": free,
                "error": "not_enough_free_space_for_vacuum",
                "hint": "Run cleanup_runtime_events first and free disk space; VACUUM may need roughly database-size free space.",
            }, indent=2))
            return 2
        store = SQLiteRuntimeStore(path)
        try:
            store.conn.execute("VACUUM")
        finally:
            store.close()
    after = path.stat().st_size if path.exists() else 0
    print(json.dumps({"apply": bool(args.apply), "path": str(path), "bytes_before": before, "bytes_after": after}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
