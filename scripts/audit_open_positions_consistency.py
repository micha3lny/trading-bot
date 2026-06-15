#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import DEFAULT_SQLITE_PATH, resolve_sqlite_path  # noqa: E402


OPEN_STATUSES = ("OPEN", "EXIT_ORDER")


def parse_raw(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def active_rows(conn: sqlite3.Connection, selected_date: str) -> list[sqlite3.Row]:
    if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'").fetchone():
        return []
    return conn.execute(
        f"""
        SELECT position_key, strategy_name, session_date, symbol, status, quantity,
               source, ibkr_quantity, updated_at, raw_json
        FROM positions
        WHERE COALESCE(session_date, '') <= ?
          AND COALESCE(active, 0) = 1
          AND UPPER(COALESCE(status, '')) IN ({",".join("?" for _ in OPEN_STATUSES)})
        ORDER BY UPPER(symbol), COALESCE(updated_at, '') DESC, position_key
        """,
        (selected_date, *OPEN_STATUSES),
    ).fetchall()


def audit(sqlite_path: str | Path, selected_date: str) -> dict[str, Any]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = active_rows(conn, selected_date)
    finally:
        conn.close()

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    stale_active: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    mixed_sources: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["symbol"] = str(item.get("symbol") or "").upper()
        raw = parse_raw(item.get("raw_json"))
        item["raw_flags"] = {
            "entry_fill_verified": raw.get("entry_fill_verified"),
            "ibkr_entry_confirmed": raw.get("ibkr_entry_confirmed"),
            "ibkr_position_flat_confirmed": raw.get("ibkr_position_flat_confirmed"),
        }
        by_symbol.setdefault(item["symbol"], []).append(item)
        if str(item.get("session_date") or "") < selected_date:
            stale_active.append(item)
        ibkr_qty = item.get("ibkr_quantity")
        confirmed = False
        try:
            confirmed = ibkr_qty is not None and abs(float(ibkr_qty)) > 1e-9
        except Exception:
            confirmed = False
        if not confirmed and not raw.get("ibkr_entry_confirmed") and not raw.get("entry_fill_verified"):
            unconfirmed.append(item)

    duplicates = {symbol: items for symbol, items in by_symbol.items() if len(items) > 1}
    for symbol, items in duplicates.items():
        sources = {str(item.get("source") or "") for item in items}
        if "live_buy" in sources and "sqlite_execution_reducer" in sources:
            mixed_sources.append({"symbol": symbol, "sources": sorted(sources), "rows": items})

    return {
        "date": selected_date,
        "active_open_rows": len(rows),
        "duplicate_active_symbol_count": len(duplicates),
        "duplicate_active_symbols": duplicates,
        "stale_active_open_count": len(stale_active),
        "stale_active_opens": stale_active,
        "unconfirmed_active_open_count": len(unconfirmed),
        "unconfirmed_active_opens": unconfirmed,
        "mixed_live_buy_execution_reducer_count": len(mixed_sources),
        "mixed_live_buy_execution_reducer": mixed_sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit duplicate/stale active open positions in runtime SQLite.")
    parser.add_argument("--sqlite-path", default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(resolve_sqlite_path(args.sqlite_path), args.date), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
