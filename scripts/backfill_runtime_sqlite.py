#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, resolve_sqlite_path, safe_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def session_dirs(root: Path, start_date: str | None, end_date: str | None) -> list[Path]:
    out = []
    for path in sorted(root.iterdir() if root.exists() else []):
        if not path.is_dir():
            continue
        name = path.name
        if start_date and name < start_date:
            continue
        if end_date and name > end_date:
            continue
        out.append(path)
    return out


EventKey = tuple[str, str, str, str, str, str]


def runtime_event_key(*, event_time: str, event_type: str, symbol: str, order_id: str, execution_id: str, source: str) -> EventKey:
    return (event_time, event_type, symbol or "", order_id or "", execution_id or "", source)


def load_existing_event_keys(store: SQLiteRuntimeStore, session_date: str) -> set[EventKey]:
    rows = store.query(
        """
        SELECT event_time, event_type, COALESCE(symbol, '') AS symbol,
               COALESCE(order_id, '') AS order_id, COALESCE(execution_id, '') AS execution_id,
               source
        FROM runtime_events
        WHERE session_date = ?
        """,
        [session_date],
    )
    return {
        runtime_event_key(
            event_time=str(row.get("event_time") or ""),
            event_type=str(row.get("event_type") or ""),
            symbol=str(row.get("symbol") or ""),
            order_id=str(row.get("order_id") or ""),
            execution_id=str(row.get("execution_id") or ""),
            source=str(row.get("source") or ""),
        )
        for row in rows
    }


def runtime_event_exists(store: SQLiteRuntimeStore, *, event_time: str, event_type: str, symbol: str, order_id: str, execution_id: str, source: str) -> bool:
    rows = store.query(
        """
        SELECT id FROM runtime_events
        WHERE event_time = ? AND event_type = ? AND COALESCE(symbol, '') = ?
          AND COALESCE(order_id, '') = ? AND COALESCE(execution_id, '') = ? AND source = ?
        LIMIT 1
        """,
        [event_time, event_type, symbol or "", order_id or "", execution_id or "", source],
    )
    return bool(rows)


def record_event_once(
    store: SQLiteRuntimeStore,
    *,
    session_date: str,
    event_time: str,
    event_type: str,
    symbol: str = "",
    order_id: str = "",
    execution_id: str = "",
    reason: str = "",
    source: str,
    raw_json: Any = None,
    existing_keys: set[EventKey] | None = None,
) -> None:
    key = runtime_event_key(
        event_time=event_time,
        event_type=event_type,
        symbol=symbol,
        order_id=order_id,
        execution_id=execution_id,
        source=source,
    )
    if existing_keys is not None and key in existing_keys:
        return
    if existing_keys is None and runtime_event_exists(store, event_time=event_time, event_type=event_type, symbol=symbol, order_id=order_id, execution_id=execution_id, source=source):
        return
    store.record_runtime_event(
        event_time=event_time,
        severity="WARN" if any(x in event_type for x in ("FAILED", "ERROR", "DRIFT", "ORPHAN")) else "INFO",
        event_type=event_type,
        session_date=session_date,
        symbol=symbol,
        order_id=order_id,
        execution_id=execution_id,
        source=source,
        reason=reason,
        action_required=1 if "MANUAL_REQUIRED" in event_type else 0,
        raw_json=raw_json,
    )
    if existing_keys is not None:
        existing_keys.add(key)


def row_event_time(row: dict[str, Any], session_date: str) -> str:
    return (
        row.get("recorded_at")
        or row.get("timestamp")
        or row.get("event_time")
        or row.get("time")
        or f"{session_date}T00:00:00+00:00"
    )


def import_session(store: SQLiteRuntimeStore, session: Path) -> dict[str, int]:
    session_date = session.name
    counts = {"executions": 0, "events": 0, "positions": 0, "reconciliation": 0}
    existing_keys = load_existing_event_keys(store, session_date)
    print(f"BACKFILL_SQLITE_SESSION_START session={session_date}", flush=True)

    for row in read_csv_rows(session / "fills.csv"):
        row["session_date"] = session_date
        store.upsert_execution(row)
        counts["executions"] += 1
    print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=fills executions={counts['executions']}", flush=True)

    trade_lifecycle_rows = read_csv_rows(session / "trade_lifecycle.csv")
    for row in trade_lifecycle_rows:
        event_time = row_event_time(row, session_date)
        record_event_once(
            store,
            session_date=session_date,
            event_time=event_time,
            event_type=row.get("event") or "TRADE_LIFECYCLE_EVENT",
            symbol=str(row.get("symbol") or "").upper(),
            order_id=str(row.get("order_id") or ""),
            execution_id=str(row.get("execution_id") or ""),
            reason=str(row.get("reason") or ""),
            source="backfill_trade_lifecycle",
            raw_json=row,
            existing_keys=existing_keys,
        )
        counts["events"] += 1
    print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=trade_lifecycle rows={len(trade_lifecycle_rows)} events={counts['events']}", flush=True)

    lifecycle_jsonl = session / "order_lifecycle.jsonl"
    if lifecycle_jsonl.exists():
        jsonl_rows = 0
        for line in lifecycle_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            jsonl_rows += 1
            event_time = row_event_time(row, session_date)
            record_event_once(
                store,
                session_date=session_date,
                event_time=event_time,
                event_type=row.get("event_type") or row.get("event") or "ORDER_LIFECYCLE_EVENT",
                symbol=str(row.get("symbol") or "").upper(),
                order_id=str(row.get("ib_order_id") or row.get("order_id") or ""),
                execution_id=str(row.get("execution_id") or ""),
                reason=str(row.get("reason") or ""),
                source="backfill_order_lifecycle",
                raw_json=row,
                existing_keys=existing_keys,
            )
            counts["events"] += 1
        print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=order_lifecycle rows={jsonl_rows} events={counts['events']}", flush=True)

    managed = read_json(session / "managed_positions.json") or {}
    managed_positions = managed.get("positions") if isinstance(managed.get("positions"), dict) else managed
    for symbol, pos in (managed_positions or {}).items():
        if not isinstance(pos, dict):
            continue
        store.upsert_position({
            "strategy_name": pos.get("strategy") or managed.get("strategy") or "v67",
            "session_date": session_date,
            "symbol": symbol,
            "status": "OPEN" if pos.get("active", True) else "CLOSED",
            "quantity": pos.get("quantity"),
            "avg_price": pos.get("entry_price"),
            "source": pos.get("source"),
            "active": pos.get("active", True),
            "exit_sent": pos.get("exit_sent", False),
            "updated_at": managed.get("recorded_at") or utc_now_iso(),
            "raw_json": pos,
        })
        counts["positions"] += 1
    print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=managed_positions positions={counts['positions']}", flush=True)

    eod = read_json(session / "eod_summary.json")
    if isinstance(eod, dict):
        eod_time = eod.get("recorded_at") or eod.get("timestamp") or f"{session_date}T23:59:59+00:00"
        store.record_reconciliation_run(
            run_id=f"backfill:eod:{session_date}",
            started_at=eod_time,
            finished_at=eod_time,
            mode="eod",
            clean=eod.get("clean"),
            ibkr_positions_count=eod.get("open_positions"),
            managed_positions_count=eod.get("managed_open"),
            orphan_count=len(eod.get("fractional_orphans") or []) + len(eod.get("whole_share_orphans") or []),
            fractional_orphan_count=len(eod.get("fractional_orphans") or []),
            drift_count=0,
            pending_orders_count=eod.get("pending_orders"),
            details_json=eod,
        )
        record_event_once(
            store,
            session_date=session_date,
            event_time=eod_time,
            event_type="EOD_FINAL_STATUS",
            source="backfill_eod_summary",
            raw_json=eod,
            existing_keys=existing_keys,
        )
        counts["reconciliation"] += 1

    run_metadata_rows = read_csv_rows(session / "run_metadata.csv")
    for row in run_metadata_rows:
        record_event_once(
            store,
            session_date=session_date,
            event_time=row_event_time(row, session_date),
            event_type="RUN_METADATA",
            source="backfill_run_metadata",
            raw_json=row.get("metadata_json") or row,
            existing_keys=existing_keys,
        )
        counts["events"] += 1
    print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=run_metadata rows={len(run_metadata_rows)} events={counts['events']} reconciliation={counts['reconciliation']}", flush=True)

    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill live recorder CSV/JSONL artifacts into runtime SQLite.")
    ap.add_argument("--recorder-root", default="data/live/recorder")
    ap.add_argument("--sqlite-path", default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    args = ap.parse_args()

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    totals = {"sessions": 0, "executions": 0, "events": 0, "positions": 0, "reconciliation": 0}
    try:
        for session in session_dirs(Path(args.recorder_root), args.start_date, args.end_date):
            counts = import_session(store, session)
            totals["sessions"] += 1
            for key, value in counts.items():
                totals[key] += value
            print(f"BACKFILL_SQLITE_SESSION session={session.name} counts={safe_json(counts)}", flush=True)
    finally:
        store.close()
    print(f"BACKFILL_SQLITE_DONE totals={safe_json(totals)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
