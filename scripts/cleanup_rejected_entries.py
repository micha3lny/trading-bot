#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rejected_symbols_from_lifecycle(session_dir: Path) -> set[str]:
    path = session_dir / "trade_lifecycle.csv"
    if not path.exists():
        return set()
    import csv

    symbols: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            if row.get("event") == "ENTRY_ORDER_REJECTED":
                symbol = str(row.get("symbol") or "").upper().strip()
                if symbol:
                    symbols.add(symbol)
    return symbols


def load_positions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    positions = data.get("positions", data) if isinstance(data, dict) else {}
    return positions if isinstance(positions, dict) else {}


def cleanup_session(session_dir: Path, *, apply: bool) -> dict[str, Any]:
    managed_path = session_dir / "managed_positions.json"
    positions = load_positions(managed_path)
    rejected = rejected_symbols_from_lifecycle(session_dir)
    remove = sorted(symbol for symbol in rejected if symbol in positions)
    if apply and remove:
        for symbol in remove:
            positions.pop(symbol, None)
        managed_path.write_text(json.dumps({"positions": positions}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"session": session_dir.name, "managed_positions": len(positions), "rejected_found": len(rejected), "would_remove": remove, "applied": apply}


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove ENTRY_REJECTED entries from managed_positions.json without touching audit logs")
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--recorder-root", default="data/live/recorder")
    parser.add_argument("--date", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.session_dir:
        sessions = [Path(args.session_dir)]
    elif args.date:
        sessions = [Path(args.recorder_root) / args.date]
    else:
        sessions = sorted(Path(args.recorder_root).glob("20??-??-??"))
    for session_dir in sessions:
        if session_dir.exists():
            print(json.dumps(cleanup_session(session_dir, apply=bool(args.apply)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

