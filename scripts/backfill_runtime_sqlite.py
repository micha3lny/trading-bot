#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, resolve_sqlite_path, safe_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_session_date() -> str:
    return datetime.now(timezone.utc).strftime("%F")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        yield from csv.DictReader(fh)


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
    severity = "WARN" if any(x in event_type for x in ("FAILED", "ERROR", "DRIFT", "ORPHAN")) else "INFO"
    action_required = 1 if "MANUAL_REQUIRED" in event_type else 0
    if existing_keys is not None:
        store.conn.execute(
            """
            INSERT INTO runtime_events (
                event_time, severity, event_type, session_date, symbol, order_id,
                execution_id, source, reason, action_required, first_seen_at,
                last_seen_at, repeat_count, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            [
                event_time,
                severity,
                event_type,
                session_date,
                symbol or None,
                order_id or None,
                execution_id or None,
                source,
                reason or None,
                action_required,
                event_time,
                event_time,
                safe_json(raw_json),
            ],
        )
        existing_keys.add(key)
        return
    store.record_runtime_event(
        event_time=event_time,
        severity=severity,
        event_type=event_type,
        session_date=session_date,
        symbol=symbol,
        order_id=order_id,
        execution_id=execution_id,
        source=source,
        reason=reason,
        action_required=action_required,
        raw_json=raw_json,
    )


def row_event_time(row: dict[str, Any], session_date: str) -> str:
    return (
        row.get("recorded_at")
        or row.get("timestamp")
        or row.get("event_time")
        or row.get("time")
        or f"{session_date}T00:00:00+00:00"
    )


def normalized_side(value: Any) -> str:
    text = str(value or "").upper()
    if text in {"BOT", "BUY"}:
        return "BUY"
    if text in {"SLD", "SELL"}:
        return "SELL"
    return text


def as_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def as_optional_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def parse_dt(value: Any) -> datetime | None:
    try:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def parse_raw_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def raw_float(raw: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = as_optional_float(raw.get(key))
        if value is not None:
            return value
    return None


def raw_peak_pct(raw: dict[str, Any]) -> float | None:
    peak_pct = raw_float(raw, "mfe_pct", "peak_pct", "peak_gain_pct", "max_gain_pct")
    if peak_pct is not None:
        return peak_pct
    entry_price = raw_float(raw, "entry_price", "buy", "buy_price")
    peak_price = raw_float(raw, "peak_price", "high_watermark", "mfe_price")
    if entry_price and peak_price is not None:
        return ((peak_price / entry_price) - 1.0) * 100.0
    return None


def pct_from_prices(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return ((numerator / denominator) - 1.0) * 100.0


def reconstruct_trades_from_execution_pairs(store: SQLiteRuntimeStore, session_date: str) -> int:
    executions = store.query(
        """
        SELECT execution_id, trade_id, COALESCE(strategy_name, 'unknown') AS strategy_name,
               session_date, symbol, side, quantity, price, executed_at, recorded_at,
               commission, commission_source, raw_json
        FROM executions
        WHERE session_date = ?
        ORDER BY COALESCE(executed_at, recorded_at), execution_id
        """,
        [session_date],
    )
    existing = store.query(
        "SELECT symbol, trade_id FROM trades WHERE session_date = ?",
        [session_date],
    )
    existing_symbols = {str(row.get("symbol") or "").upper() for row in existing}
    open_lots: dict[str, list[dict[str, Any]]] = {}
    created = 0
    for row in executions:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol or symbol in existing_symbols:
            continue
        side = normalized_side(row.get("side"))
        qty = as_float(row.get("quantity"))
        price = as_float(row.get("price"))
        if qty <= 0 or price <= 0:
            continue
        if side == "BUY":
            lot = dict(row)
            lot["remaining_qty"] = qty
            lot["original_qty"] = qty
            open_lots.setdefault(symbol, []).append(lot)
            continue
        if side != "SELL":
            continue
        remaining = qty
        while remaining > 1e-9 and open_lots.get(symbol):
            lot = open_lots[symbol][0]
            matched_qty = min(remaining, as_float(lot.get("remaining_qty")))
            buy_price = as_float(lot.get("price"))
            sell_price = price
            gross = (sell_price - buy_price) * matched_qty
            buy_exec = str(lot.get("execution_id") or "")
            sell_exec = str(row.get("execution_id") or "")
            trade_id = f"reconstructed:{session_date}:{symbol}:{buy_exec}:{sell_exec}"
            store.upsert_trade({
                "trade_id": trade_id,
                "strategy_name": row.get("strategy_name") or lot.get("strategy_name") or "unknown",
                "session_date": session_date,
                "symbol": symbol,
                "status": "CLOSED",
                "entry_fill_time": lot.get("executed_at"),
                "exit_fill_time": row.get("executed_at"),
                "closed_at": row.get("executed_at"),
                "entry_price": buy_price,
                "exit_price": sell_price,
                "quantity": matched_qty,
                "gross_pnl": gross,
                "commission": as_float(lot.get("commission")) + as_float(row.get("commission")),
                "net_pnl": gross - as_float(lot.get("commission")) - as_float(row.get("commission")),
                "ibkr_entry_confirmed": True,
                "ibkr_exit_confirmed": True,
                "ibkr_position_flat_confirmed": True,
                "raw_json": {
                    "reconstruction_source": "executions_pair",
                    "buy_execution_id": buy_exec,
                    "sell_execution_id": sell_exec,
                    "entry_executed_at": lot.get("executed_at"),
                    "exit_executed_at": row.get("executed_at"),
                },
            })
            created += 1
            lot["remaining_qty"] = as_float(lot.get("remaining_qty")) - matched_qty
            remaining -= matched_qty
            if as_float(lot.get("remaining_qty")) <= 1e-9:
                open_lots[symbol].pop(0)
    if created:
        store.conn.commit()
    return created


def enrich_trades_from_runtime_events(store: SQLiteRuntimeStore, session_date: str) -> int:
    trades = store.query(
        """
        SELECT trade_id, strategy_name, session_date, symbol, entry_fill_time, exit_fill_time,
               closed_at, entry_price, exit_price, raw_json
        FROM trades
        WHERE session_date = ?
          AND UPPER(COALESCE(status, '')) IN ('CLOSED', 'DONE', 'EXIT_FILLED', 'FLAT', 'COMMISSION_PENDING', 'PNL_PENDING')
        """,
        [session_date],
    )
    if not trades:
        return 0
    events = store.query(
        """
        SELECT event_time, event_type, COALESCE(strategy_name, 'unknown') AS strategy_name,
               symbol, raw_json
        FROM runtime_events
        WHERE COALESCE(session_date, substr(event_time, 1, 10)) = ?
        ORDER BY event_time
        """,
        [session_date],
    )
    events_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        symbol = str(event.get("symbol") or "").upper()
        if not symbol:
            continue
        row = dict(event)
        row["raw"] = parse_raw_json(row.get("raw_json"))
        row["event_dt"] = parse_dt(row.get("event_time"))
        events_by_symbol.setdefault(symbol, []).append(row)
    trade_counts: dict[str, int] = {}
    for trade in trades:
        symbol = str(trade.get("symbol") or "").upper()
        trade_counts[symbol] = trade_counts.get(symbol, 0) + 1
    updated = 0
    for trade in trades:
        symbol = str(trade.get("symbol") or "").upper()
        symbol_events = events_by_symbol.get(symbol) or []
        if not symbol_events:
            continue
        start_dt = parse_dt(trade.get("entry_fill_time"))
        end_dt = parse_dt(trade.get("exit_fill_time") or trade.get("closed_at"))
        scoped = symbol_events
        if start_dt or end_dt:
            scoped = [
                event for event in symbol_events
                if event.get("event_dt") is None
                or ((start_dt is None or event["event_dt"] >= start_dt) and (end_dt is None or event["event_dt"] <= end_dt))
            ]
            if not scoped:
                scoped = symbol_events
        elif trade_counts.get(symbol, 0) > 1:
            continue

        entry_event_time = trade.get("entry_fill_time")
        exit_event_time = trade.get("exit_fill_time") or trade.get("closed_at")
        best_peak_pct: float | None = None
        best_peak_price: float | None = None
        entry_price = as_optional_float(trade.get("entry_price"))
        exit_price = as_optional_float(trade.get("exit_price"))
        for event in scoped:
            raw = event.get("raw") or {}
            event_type = str(event.get("event_type") or "").upper()
            if not entry_event_time and event_type in {"BUY_ORDER_SENT", "ENTRY_ORDER_SUBMITTED", "ENTRY_ORDER_FILLED", "POSITION_OPENED"}:
                entry_event_time = event.get("event_time")
            entry_price = entry_price or raw_float(raw, "entry_price", "buy", "buy_price")
            peak_price = raw_float(raw, "peak_price", "high_watermark", "mfe_price")
            peak_pct = raw_peak_pct(raw)
            if peak_price is None and entry_price is not None and peak_pct is not None:
                peak_price = entry_price * (1.0 + peak_pct / 100.0)
            if peak_pct is None:
                peak_pct = pct_from_prices(peak_price, entry_price)
            if peak_pct is not None and (best_peak_pct is None or peak_pct > best_peak_pct):
                best_peak_pct = peak_pct
                best_peak_price = peak_price
            if event_type in {"SELL_ORDER_SENT", "POSITION_VERIFIED_CLOSED", "POSITION_CLOSED"}:
                exit_event_time = exit_event_time or event.get("event_time")
                exit_price = exit_price or raw_float(raw, "exit_price", "sell_price", "decision_price", "price")
        if best_peak_pct is None and entry_event_time == trade.get("entry_fill_time") and exit_event_time == (trade.get("exit_fill_time") or trade.get("closed_at")):
            continue
        drop_from_peak_pct = pct_from_prices(exit_price, best_peak_price)
        raw = parse_raw_json(trade.get("raw_json"))
        raw.update({
            "peak_source": "runtime_events",
            "peak_match_quality": "trade_time_window" if (start_dt or end_dt) else "symbol_session_unique",
            "peak_price": best_peak_price,
            "peak_gain_pct": best_peak_pct,
            "drop_from_peak_pct": drop_from_peak_pct,
            "time_source": "runtime_events",
        })
        store.conn.execute(
            """
            UPDATE trades
            SET entry_fill_time = COALESCE(entry_fill_time, ?),
                exit_fill_time = COALESCE(exit_fill_time, ?),
                closed_at = COALESCE(closed_at, ?),
                mfe_pct = COALESCE(?, mfe_pct),
                raw_json = ?
            WHERE trade_id = ?
            """,
            [entry_event_time, exit_event_time, exit_event_time, best_peak_pct, safe_json(raw), trade.get("trade_id")],
        )
        updated += 1
    if updated:
        store.conn.commit()
    return updated


def import_session(store: SQLiteRuntimeStore, session: Path, *, progress_interval: int = 500) -> dict[str, int]:
    session_date = session.name
    counts = {"executions": 0, "events": 0, "positions": 0, "reconciliation": 0}
    existing_keys = load_existing_event_keys(store, session_date)
    print(f"BACKFILL_SQLITE_SESSION_START session={session_date}", flush=True)

    execution_symbols: set[str] = set()
    for row in iter_csv_rows(session / "fills.csv"):
        row["session_date"] = session_date
        row.setdefault("strategy_name", "v67")
        if row.get("symbol"):
            execution_symbols.add(str(row.get("symbol") or "").upper())
        store.upsert_execution(row)
        counts["executions"] += 1
        if progress_interval > 0 and counts["executions"] % progress_interval == 0:
            print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=fills executions={counts['executions']}", flush=True)
    print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=fills executions={counts['executions']}", flush=True)
    if execution_symbols and session_date == utc_session_date():
        rebuild_result = store.rebuild_positions_from_executions(sorted(execution_symbols), allow_historical_open_lots=False)
        print(
            "BACKFILL_SQLITE_SESSION_PROGRESS "
            f"session={session_date} artifact=position_reducer "
            f"symbols={len(execution_symbols)} result={json.dumps(rebuild_result, sort_keys=True, default=str)}",
            flush=True,
        )
    reconstructed_trades = reconstruct_trades_from_execution_pairs(store, session_date)
    if reconstructed_trades:
        print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=reconstructed_trades trades={reconstructed_trades}", flush=True)

    trade_lifecycle_rows = 0
    for row in iter_csv_rows(session / "trade_lifecycle.csv"):
        trade_lifecycle_rows += 1
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
        if progress_interval > 0 and trade_lifecycle_rows % progress_interval == 0:
            store.conn.commit()
            print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=trade_lifecycle rows={trade_lifecycle_rows} events={counts['events']}", flush=True)
    store.conn.commit()
    print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=trade_lifecycle rows={trade_lifecycle_rows} events={counts['events']}", flush=True)

    lifecycle_jsonl = session / "order_lifecycle.jsonl"
    if lifecycle_jsonl.exists():
        jsonl_rows = 0
        with lifecycle_jsonl.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
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
                if progress_interval > 0 and jsonl_rows % progress_interval == 0:
                    store.conn.commit()
                    print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=order_lifecycle rows={jsonl_rows} events={counts['events']}", flush=True)
        store.conn.commit()
        print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=order_lifecycle rows={jsonl_rows} events={counts['events']}", flush=True)

    enriched_trades = enrich_trades_from_runtime_events(store, session_date)
    if enriched_trades:
        print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=trade_metrics trades={enriched_trades}", flush=True)

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

    run_metadata_rows = 0
    for row in iter_csv_rows(session / "run_metadata.csv"):
        run_metadata_rows += 1
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
        if progress_interval > 0 and run_metadata_rows % progress_interval == 0:
            store.conn.commit()
            print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=run_metadata rows={run_metadata_rows} events={counts['events']} reconciliation={counts['reconciliation']}", flush=True)
    store.conn.commit()
    print(f"BACKFILL_SQLITE_SESSION_PROGRESS session={session_date} artifact=run_metadata rows={run_metadata_rows} events={counts['events']} reconciliation={counts['reconciliation']}", flush=True)

    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill live recorder CSV/JSONL artifacts into runtime SQLite.")
    ap.add_argument("--recorder-root", default="data/live/recorder")
    ap.add_argument("--sqlite-path", default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--progress-interval", type=int, default=500)
    args = ap.parse_args()

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    totals = {"sessions": 0, "executions": 0, "events": 0, "positions": 0, "reconciliation": 0}
    try:
        for session in session_dirs(Path(args.recorder_root), args.start_date, args.end_date):
            counts = import_session(store, session, progress_interval=max(0, int(args.progress_interval)))
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
