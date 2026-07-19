#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_OUTPUT_DIR = Path("data/analysis")


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def parse_raw(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def first_present(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def norm_status(value: Any) -> str:
    return str(value or "").strip().upper()


def query_rows(conn: sqlite3.Connection, sql: str, params: list[Any] | tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def classify_placeholder(*, trade: dict[str, Any], orders: list[dict[str, Any]], direct_executions: list[dict[str, Any]], canonical_closed_count: int) -> str:
    if direct_executions:
        return "INVESTIGATE"
    order_statuses = {norm_status(row.get("status") or row.get("ibkr_status")) for row in orders}
    if any("CANCEL" in status or "REJECT" in status or "INACTIVE" in status for status in order_statuses):
        return "CANCELLED"
    if canonical_closed_count > 0:
        return "STALE_PLACEHOLDER"
    return "INVESTIGATE"


def diagnose(conn: sqlite3.Connection, *, date: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    trade_cols = table_columns(conn, "trades")
    if not trade_cols:
        return []
    order_cols = table_columns(conn, "orders")
    exec_cols = table_columns(conn, "executions")
    where = ["UPPER(COALESCE(status, '')) = 'ENTRY_PENDING'"]
    params: list[Any] = []
    if date:
        where.append("COALESCE(session_date, substr(entry_order_time, 1, 10), substr(updated_at, 1, 10), substr(created_at, 1, 10)) = ?")
        params.append(date)
    sql = f"""
        SELECT trade_id, symbol, session_date, status, entry_order_id, entry_perm_id,
               entry_order_time, entry_fill_time, exit_fill_time, entry_price, exit_price,
               quantity, raw_json, created_at, updated_at
        FROM trades
        WHERE {' AND '.join(where)}
        ORDER BY COALESCE(updated_at, created_at, entry_order_time, session_date, ''), symbol, trade_id
    """
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = []
    for trade in query_rows(conn, sql, params):
        raw = parse_raw(trade.get("raw_json"))
        trade_id = str(trade.get("trade_id") or "")
        symbol = str(trade.get("symbol") or "").upper()
        session_date = str(trade.get("session_date") or "")
        order_id = first_present(trade.get("entry_order_id"), raw.get("entry_order_id"), raw.get("order_id"), raw.get("orderId"))
        order_rows: list[dict[str, Any]] = []
        if order_cols:
            clauses = []
            order_params: list[Any] = []
            if trade_id and "trade_id" in order_cols:
                clauses.append("trade_id = ?")
                order_params.append(trade_id)
            if order_id and "order_id" in order_cols:
                clauses.append("CAST(order_id AS TEXT) = ?")
                order_params.append(order_id)
            if clauses:
                selected = ", ".join(col for col in ["order_key", "trade_id", "symbol", "side", "status", "ibkr_status", "order_id", "perm_id", "submitted_at", "updated_at"] if col in order_cols)
                order_rows = query_rows(conn, f"SELECT {selected} FROM orders WHERE {' OR '.join(clauses)} ORDER BY COALESCE(updated_at, submitted_at, '')", order_params)
        direct_execution_rows: list[dict[str, Any]] = []
        symbol_session_execution_rows: list[dict[str, Any]] = []
        if exec_cols:
            selected = ", ".join(col for col in ["execution_id", "trade_id", "symbol", "side", "quantity", "price", "order_id", "perm_id", "executed_at", "recorded_at", "realized_pnl"] if col in exec_cols)
            direct_clauses = []
            direct_params: list[Any] = []
            if trade_id and "trade_id" in exec_cols:
                direct_clauses.append("trade_id = ?")
                direct_params.append(trade_id)
            if order_id and "order_id" in exec_cols:
                direct_clauses.append("CAST(order_id AS TEXT) = ?")
                direct_params.append(order_id)
            if direct_clauses:
                direct_execution_rows = query_rows(
                    conn,
                    f"SELECT {selected} FROM executions WHERE {' OR '.join(direct_clauses)} ORDER BY COALESCE(executed_at, recorded_at, '')",
                    direct_params,
                )
            if symbol and session_date:
                symbol_session_execution_rows = query_rows(
                    conn,
                    f"SELECT {selected} FROM executions WHERE UPPER(symbol) = ? AND session_date = ? ORDER BY COALESCE(executed_at, recorded_at, '')",
                    [symbol, session_date],
                )
        canonical_closed_count = 0
        if symbol and session_date:
            canonical_closed_count = int((query_rows(conn, """
                SELECT COUNT(*) AS n
                FROM trades
                WHERE UPPER(symbol) = ?
                  AND session_date = ?
                  AND UPPER(COALESCE(status, '')) IN ('CLOSED', 'COMMISSION_PENDING', 'PNL_PENDING')
                  AND NULLIF(entry_fill_time, '') IS NOT NULL
                  AND NULLIF(exit_fill_time, '') IS NOT NULL
                  AND entry_price IS NOT NULL
                  AND exit_price IS NOT NULL
            """, [symbol, session_date]) or [{"n": 0}])[0].get("n") or 0)
        classification = classify_placeholder(
            trade=trade,
            orders=order_rows,
            direct_executions=direct_execution_rows,
            canonical_closed_count=canonical_closed_count,
        )
        rows.append({
            "trade_id": trade_id,
            "symbol": symbol,
            "session_date": session_date,
            "status": trade.get("status"),
            "created_at": trade.get("created_at"),
            "updated_at": trade.get("updated_at"),
            "entry_order_time": trade.get("entry_order_time"),
            "entry_order_id": order_id,
            "entry_perm_id": first_present(trade.get("entry_perm_id"), raw.get("entry_perm_id"), raw.get("perm_id"), raw.get("permId")),
            "entry_fill_time": trade.get("entry_fill_time"),
            "exit_fill_time": trade.get("exit_fill_time"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "quantity": trade.get("quantity"),
            "orders_count": len(order_rows),
            "order_statuses": ";".join(sorted({first_present(row.get("status"), row.get("ibkr_status")) for row in order_rows if first_present(row.get("status"), row.get("ibkr_status"))})),
            "direct_executions_count": len(direct_execution_rows),
            "direct_execution_ids": ";".join(str(row.get("execution_id") or "") for row in direct_execution_rows if row.get("execution_id")),
            "symbol_session_executions_count": len(symbol_session_execution_rows),
            "symbol_session_execution_ids": ";".join(str(row.get("execution_id") or "") for row in symbol_session_execution_rows if row.get("execution_id")),
            "canonical_closed_same_symbol_day_count": canonical_closed_count,
            "recommended_classification": classification,
        })
    return rows


def write_outputs(rows: list[dict[str, Any]], *, output_dir: Path, suffix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"stale_entry_pending_diagnostic_{suffix}.csv"
    md_path = output_dir / f"stale_entry_pending_diagnostic_{suffix}.md"
    fieldnames = [
        "trade_id", "symbol", "session_date", "status", "created_at", "updated_at", "entry_order_time",
        "entry_order_id", "entry_perm_id", "entry_fill_time", "exit_fill_time", "entry_price", "exit_price", "quantity",
        "orders_count", "order_statuses", "direct_executions_count", "direct_execution_ids",
        "symbol_session_executions_count", "symbol_session_execution_ids", "canonical_closed_same_symbol_day_count",
        "recommended_classification",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("recommended_classification") or "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# Stale ENTRY_PENDING Diagnostic",
        "",
        f"generated_at={datetime.now(timezone.utc).isoformat()}",
        f"rows={len(rows)}",
        f"classification_counts={json.dumps(counts, sort_keys=True)}",
        "",
        "## First 10",
        "",
    ]
    for row in rows[:10]:
        lines.append(
            f"- {row.get('symbol')} trade_id={row.get('trade_id')} order_id={row.get('entry_order_id')} "
            f"updated_at={row.get('updated_at') or row.get('created_at') or ''} "
            f"direct_executions={row.get('direct_executions_count')} symbol_session_executions={row.get('symbol_session_executions_count')} "
            f"canonical_closed_same_day={row.get('canonical_closed_same_symbol_day_count')} "
            f"classification={row.get('recommended_classification')}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only diagnostic for stale ENTRY_PENDING trade placeholders.")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_DB))
    parser.add_argument("--date")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows; 0 means no limit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    sqlite_path = Path(args.sqlite_path)
    with connect(sqlite_path) as conn:
        rows = diagnose(conn, date=args.date, limit=args.limit if args.limit and args.limit > 0 else None)
    suffix = args.date or "all"
    csv_path, md_path = write_outputs(rows, output_dir=Path(args.output_dir), suffix=suffix)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("recommended_classification") or "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    print(
        f"STALE_ENTRY_PENDING_DIAGNOSTIC_DONE rows={len(rows)} classification_counts={json.dumps(counts, sort_keys=True)} output={csv_path} summary={md_path}",
        flush=True,
    )
    for row in rows[:10]:
        print(
            f"STALE_ENTRY_PENDING_SAMPLE symbol={row.get('symbol')} trade_id={row.get('trade_id')} "
            f"order_id={row.get('entry_order_id')} direct_executions={row.get('direct_executions_count')} "
            f"symbol_session_executions={row.get('symbol_session_executions_count')} "
            f"canonical_closed_same_day={row.get('canonical_closed_same_symbol_day_count')} "
            f"classification={row.get('recommended_classification')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
