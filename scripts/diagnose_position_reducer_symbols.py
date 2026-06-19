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

from src.live_trading.storage.sqlite_store import connect_sqlite, parse_jsonish, resolve_sqlite_path, safe_float  # noqa: E402


def rows(conn: sqlite3.Connection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def print_table(title: str, data: list[dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    if not data:
        print("(none)")
        return
    headers = list(data[0].keys())
    print(",".join(headers))
    for row in data:
        print(",".join("" if row.get(col) is None else str(row.get(col)) for col in headers))


def active_quantity(position_rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in position_rows:
        if int(row.get("active") or 0) != 1:
            continue
        status = str(row.get("status") or "").upper()
        if status not in {"OPEN", "EXIT_ORDER"}:
            continue
        total += safe_float(row.get("quantity")) or 0.0
    return total


def raw_lot_ids(row: dict[str, Any]) -> str:
    raw = parse_jsonish(row.get("raw_json"))
    ids = raw.get("open_lot_execution_ids")
    if isinstance(ids, list):
        return "|".join(str(item) for item in ids)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose SQLite execution reducer quantities for selected symbols.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--date", default="")
    parser.add_argument("--symbols", default="RXT,SRPT,PDYN,VIR")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")
    placeholders = ",".join("?" for _ in symbols)
    path = resolve_sqlite_path(args.sqlite_path)
    conn = connect_sqlite(path, read_only=True)

    date_filter = ""
    params: list[Any] = symbols[:]
    if args.date:
        date_filter = "AND COALESCE(substr(executed_at, 1, 10), substr(recorded_at, 1, 10), session_date) = ?"
        params.append(args.date)

    grouped = rows(
        conn,
        f"""
        SELECT
            symbol,
            COALESCE(order_id, '') AS order_id,
            COALESCE(perm_id, '') AS perm_id,
            UPPER(COALESCE(side, '')) AS side,
            COUNT(*) AS executions,
            SUM(COALESCE(quantity, 0)) AS quantity,
            MIN(COALESCE(executed_at, recorded_at, '')) AS first_time,
            MAX(COALESCE(executed_at, recorded_at, '')) AS last_time,
            SUM(CASE WHEN COALESCE(commission_source, '') = 'ibkr' THEN 1 ELSE 0 END) AS ibkr_commission_rows,
            SUM(CASE WHEN trade_id IS NOT NULL AND trade_id != '' THEN 1 ELSE 0 END) AS linked_trade_rows
        FROM executions
        WHERE UPPER(symbol) IN ({placeholders})
        {date_filter}
        GROUP BY symbol, COALESCE(order_id, ''), COALESCE(perm_id, ''), UPPER(COALESCE(side, ''))
        ORDER BY symbol, first_time, order_id, side
        """,
        params,
    )
    details = rows(
        conn,
        f"""
        SELECT
            symbol,
            execution_id,
            COALESCE(order_id, '') AS order_id,
            COALESCE(perm_id, '') AS perm_id,
            side,
            quantity,
            price,
            executed_at,
            recorded_at,
            commission,
            commission_source,
            realized_pnl,
            COALESCE(trade_id, '') AS trade_id
        FROM executions
        WHERE UPPER(symbol) IN ({placeholders})
        {date_filter}
        ORDER BY symbol, COALESCE(executed_at, recorded_at, ''), execution_id
        """,
        params,
    )
    net_rows = rows(
        conn,
        f"""
        SELECT
            symbol,
            SUM(CASE
                WHEN UPPER(COALESCE(side, '')) IN ('BOT', 'BUY') THEN COALESCE(quantity, 0)
                WHEN UPPER(COALESCE(side, '')) IN ('SLD', 'SELL') THEN -COALESCE(quantity, 0)
                ELSE 0
            END) AS execution_net_quantity,
            SUM(CASE WHEN UPPER(COALESCE(side, '')) IN ('BOT', 'BUY') THEN COALESCE(quantity, 0) ELSE 0 END) AS buy_quantity,
            SUM(CASE WHEN UPPER(COALESCE(side, '')) IN ('SLD', 'SELL') THEN COALESCE(quantity, 0) ELSE 0 END) AS sell_quantity,
            COUNT(*) AS execution_rows
        FROM executions
        WHERE UPPER(symbol) IN ({placeholders})
        {date_filter}
        GROUP BY symbol
        ORDER BY symbol
        """,
        params,
    )
    pos = rows(
        conn,
        f"""
        SELECT
            symbol,
            position_key,
            status,
            active,
            quantity,
            avg_price,
            ibkr_quantity,
            source,
            updated_at,
            raw_json
        FROM positions
        WHERE UPPER(symbol) IN ({placeholders})
        ORDER BY symbol, COALESCE(active, 0) DESC, COALESCE(updated_at, '') DESC, position_key
        """,
        symbols,
    )
    pos_display = []
    for row in pos:
        out = dict(row)
        out["open_lot_execution_ids"] = raw_lot_ids(row)
        out.pop("raw_json", None)
        pos_display.append(out)

    active_by_symbol: dict[str, float] = {}
    for symbol in symbols:
        active_by_symbol[symbol] = active_quantity([row for row in pos if str(row.get("symbol") or "").upper() == symbol])
    net_by_symbol = {str(row.get("symbol") or "").upper(): safe_float(row.get("execution_net_quantity")) or 0.0 for row in net_rows}
    summary = []
    for symbol in symbols:
        exec_net = net_by_symbol.get(symbol, 0.0)
        sqlite_active = active_by_symbol.get(symbol, 0.0)
        summary.append(
            {
                "symbol": symbol,
                "execution_net_quantity": exec_net,
                "sqlite_active_quantity": sqlite_active,
                "difference_exec_net_minus_sqlite_active": exec_net - sqlite_active,
            }
        )

    payload = {
        "sqlite_path": path,
        "date": args.date,
        "symbols": symbols,
        "summary": summary,
        "executions_grouped": grouped,
        "positions": pos_display,
        "executions": details,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"SQLITE_PATH {path}")
    if args.date:
        print(f"DATE_FILTER {args.date}")
    print_table("SUMMARY", summary)
    print_table("EXECUTIONS_GROUPED_BY_SYMBOL_ORDER_SIDE", grouped)
    print_table("POSITIONS_FOR_SYMBOLS", pos_display)
    print_table("EXECUTIONS_DETAIL", details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
