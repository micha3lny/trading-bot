#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import DEFAULT_SQLITE_PATH, connect_sqlite, resolve_sqlite_path, utc_now_iso  # noqa: E402


OPEN_STATUSES = ("OPEN", "EXIT_ORDER")
DEFAULT_ORPHAN_STALE_DAYS = 7


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


def parse_dt(value: Any) -> datetime | None:
    try:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def stale_age_days(row: sqlite3.Row, selected_date: str) -> float | None:
    raw = parse_raw(row["raw_json"])
    entry_time = raw.get("entry_time") or raw.get("buy_time") or row["session_date"]
    if isinstance(entry_time, str) and entry_time.startswith("adopted_on_restart:"):
        entry_time = entry_time.split(":", 1)[1]
    start = parse_dt(entry_time if "T" in str(entry_time) else f"{entry_time}T00:00:00+00:00")
    end = parse_dt(f"{selected_date}T23:59:59+00:00")
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds() / 86400.0)


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


def cleanup(sqlite_path: str | Path, selected_date: str, apply: bool = False, stale_days: int = DEFAULT_ORPHAN_STALE_DAYS) -> dict[str, Any]:
    conn = connect_sqlite(sqlite_path)
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
            age = stale_age_days(canonical, selected_date)
            if str(canonical["session_date"] or "") < selected_date and not confirmed and age is not None and age > stale_days:
                raw.update(
                    {
                        "active": False,
                        "stale_carry_open": True,
                        "orphan_stale_position": True,
                        "requires_ibkr_confirmation": True,
                        "stale_carry_suppressed_at": now,
                        "orphan_stale_suppressed_at": now,
                        "orphan_stale_age_days": age,
                    }
                )
                updates.append((now, dumps_raw(raw), canonical["position_key"], "ORPHAN_STALE_POSITION"))
                stale_unconfirmed.append({"symbol": symbol, "position_key": canonical["position_key"], "session_date": canonical["session_date"], "age_days": age})

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
        "stale_days": stale_days,
        "duplicate_rows_to_suppress": len(suppressed_duplicates),
        "duplicate_rows": suppressed_duplicates,
        "stale_unconfirmed_to_suppress": len(stale_unconfirmed),
        "stale_unconfirmed": stale_unconfirmed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Suppress duplicate/stale active open position rows.")
    parser.add_argument("--sqlite-path", default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--date", required=True)
    parser.add_argument("--stale-days", type=int, default=DEFAULT_ORPHAN_STALE_DAYS)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(cleanup(resolve_sqlite_path(args.sqlite_path), args.date, apply=args.apply, stale_days=args.stale_days), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
