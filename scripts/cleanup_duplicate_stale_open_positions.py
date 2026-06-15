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

from src.live_trading.storage.sqlite_store import DEFAULT_SQLITE_PATH, resolve_sqlite_path, utc_now_iso  # noqa: E402


OPEN_STATUSES = ("OPEN", "EXIT_ORDER")


def parse_raw(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def dumps_raw(raw: dict[str, Any]) -> str:
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)


def priority(row: sqlite3.Row) -> tuple[int, str, str]:
    raw = parse_raw(row["raw_json"])
    source = str(row["source"] or "").lower()
    ibkr_qty = row["ibkr_quantity"]
    if source == "sqlite_execution_reducer":
        score = 40
    else:
        try:
            score = 30 if ibkr_qty is not None and abs(float(ibkr_qty)) > 1e-9 else 0
        except Exception:
            score = 0
        if raw.get("ibkr_entry_confirmed"):
            score = max(score, 25)
        if raw.get("entry_fill_verified"):
            score = max(score, 20)
        if source == "live_buy":
            score = max(score, 10)
    return score, str(row["updated_at"] or ""), str(row["position_key"] or "")


def fetch_active(conn: sqlite3.Connection, selected_date: str) -> list[sqlite3.Row]:
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


def cleanup(sqlite_path: str | Path, selected_date: str, apply: bool = False) -> dict[str, Any]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    now = utc_now_iso()
    suppressed_duplicates: list[dict[str, Any]] = []
    stale_unconfirmed: list[dict[str, Any]] = []
    try:
        rows = fetch_active(conn, selected_date)
        by_symbol: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_symbol.setdefault(str(row["symbol"] or "").upper(), []).append(row)

        updates: list[tuple[str, str, str, str]] = []
        for symbol, items in by_symbol.items():
            if len(items) > 1:
                keep = sorted(items, key=priority, reverse=True)[0]
                for row in items:
                    if row["position_key"] == keep["position_key"]:
                        continue
                    raw = parse_raw(row["raw_json"])
                    raw.update(
                        {
                            "active": False,
                            "stale_duplicate_suppressed": True,
                            "duplicate_suppressed_reason": "cleanup_duplicate_stale_open_positions",
                            "duplicate_suppressed_by": keep["position_key"],
                            "duplicate_suppressed_at": now,
                        }
                    )
                    updates.append((now, dumps_raw(raw), row["position_key"], "STALE_DUPLICATE_SUPPRESSED"))
                    suppressed_duplicates.append({"symbol": symbol, "position_key": row["position_key"], "kept": keep["position_key"]})

            canonical = sorted(items, key=priority, reverse=True)[0]
            raw = parse_raw(canonical["raw_json"])
            confirmed = bool(raw.get("ibkr_entry_confirmed") or raw.get("entry_fill_verified"))
            try:
                confirmed = confirmed or (canonical["ibkr_quantity"] is not None and abs(float(canonical["ibkr_quantity"])) > 1e-9)
            except Exception:
                pass
            if str(canonical["session_date"] or "") < selected_date and not confirmed:
                raw.update(
                    {
                        "active": False,
                        "stale_carry_open": True,
                        "requires_ibkr_confirmation": True,
                        "stale_carry_suppressed_at": now,
                    }
                )
                updates.append((now, dumps_raw(raw), canonical["position_key"], "STALE_CARRY_OPEN"))
                stale_unconfirmed.append({"symbol": symbol, "position_key": canonical["position_key"], "session_date": canonical["session_date"]})

        if apply and updates:
            conn.execute("BEGIN")
            for updated_at, raw_json, position_key, status in updates:
                conn.execute(
                    """
                    UPDATE positions
                    SET active = 0,
                        status = ?,
                        exit_sent = 0,
                        updated_at = ?,
                        raw_json = ?
                    WHERE position_key = ?
                    """,
                    (status, updated_at, raw_json, position_key),
                )
            conn.commit()
    finally:
        conn.close()

    return {
        "apply": apply,
        "date": selected_date,
        "duplicate_rows_to_suppress": len(suppressed_duplicates),
        "duplicate_rows": suppressed_duplicates,
        "stale_unconfirmed_to_suppress": len(stale_unconfirmed),
        "stale_unconfirmed": stale_unconfirmed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Suppress duplicate/stale active open position rows.")
    parser.add_argument("--sqlite-path", default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--date", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(cleanup(resolve_sqlite_path(args.sqlite_path), args.date, apply=args.apply), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
