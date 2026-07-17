#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_trading.storage.canonical_fifo import build_canonical_fifo, sort_execution_rows  # noqa: E402
from src.live_trading.storage.sqlite_store import connect_sqlite  # noqa: E402


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_OUTPUT_DIR = Path("data/analysis")
QTY_TOLERANCE = 1e-6
RESIDUAL_DUST_QTY = 1.0


def norm_symbol(value: Any) -> str:
    return str(value or "").upper().strip()


def norm_side(value: Any) -> str:
    text = str(value or "").upper().strip()
    if text in {"BOT", "BUY", "BOUGHT", "B"}:
        return "BUY"
    if text in {"SLD", "SELL", "SOLD", "S"}:
        return "SELL"
    return text


def fnum(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def event_time(row: dict[str, Any]) -> str:
    return str(row.get("executed_at") or row.get("recorded_at") or "")


def date_part(value: Any) -> str:
    return str(value or "")[:10]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def qmarks(values: list[Any]) -> str:
    return ",".join("?" for _ in values)


def read_rows(conn: sqlite3.Connection, sql: str, params: list[Any] | tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def selected_symbols(conn: sqlite3.Connection, *, date: str | None, symbols: list[str]) -> list[str]:
    if symbols:
        return sorted({norm_symbol(symbol) for symbol in symbols if norm_symbol(symbol)})
    if date:
        rows = read_rows(
            conn,
            """
            SELECT DISTINCT UPPER(symbol) AS symbol
            FROM executions
            WHERE session_date = ?
               OR substr(COALESCE(executed_at, recorded_at), 1, 10) = ?
            ORDER BY symbol
            """,
            [date, date],
        )
    else:
        rows = read_rows(conn, "SELECT DISTINCT UPPER(symbol) AS symbol FROM executions ORDER BY symbol")
    return [norm_symbol(row.get("symbol")) for row in rows if norm_symbol(row.get("symbol"))]


def execution_rows(conn: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    return read_rows(
        conn,
        """
        SELECT execution_id, trade_id, order_key, order_id, perm_id,
               COALESCE(strategy_name, 'unknown') AS strategy_name,
               session_date, symbol, side, quantity, price,
               executed_at, recorded_at, commission, commission_source,
               realized_pnl, raw_json
        FROM executions
        WHERE UPPER(symbol) = ?
        ORDER BY COALESCE(executed_at, recorded_at, ''), COALESCE(recorded_at, ''), execution_id
        """,
        [symbol],
    )


def component_rows(conn: sqlite3.Connection, symbol: str, date: str | None) -> list[dict[str, Any]]:
    if not table_exists(conn, "trade_components"):
        return []
    params: list[Any] = [symbol]
    clause = ""
    if date:
        clause = "AND (session_date = ? OR substr(COALESCE(exit_time, entry_time), 1, 10) = ?)"
        params.extend([date, date])
    return read_rows(
        conn,
        f"""
        SELECT trade_id AS canonical_trade_id, symbol, buy_execution_id, sell_execution_id,
               entry_time AS buy_time, exit_time AS sell_time, matched_qty,
               buy_price, sell_price, session_date
        FROM trade_components
        WHERE UPPER(symbol) = ? {clause}
        ORDER BY COALESCE(exit_time, entry_time), sell_execution_id, entry_time, buy_execution_id
        """,
        params,
    )


def latest_position_quantity(conn: sqlite3.Connection, symbol: str) -> tuple[float | None, str]:
    if not table_exists(conn, "positions"):
        return None, ""
    rows = read_rows(
        conn,
        """
        SELECT quantity, status, active, source, updated_at, raw_json
        FROM positions
        WHERE UPPER(symbol) = ?
        ORDER BY COALESCE(updated_at, '') DESC, rowid DESC
        LIMIT 1
        """,
        [symbol],
    )
    if not rows:
        return None, ""
    row = rows[0]
    return fnum(row.get("quantity")), json.dumps(row, sort_keys=True, default=str)


@dataclass
class InventoryEvent:
    symbol: str
    event_type: str
    event_time: str
    execution_id: str
    side: str
    execution_qty: float
    position_before: float
    position_after: float
    status: str
    note: str = ""


def classify_cycle(
    *,
    initial_history_date: str,
    position_before: float,
    position_after: float,
    current_open_qty: float,
    final_reconstructed_qty: float,
    latest_position_qty: float | None,
) -> str:
    if abs(position_after) <= QTY_TOLERANCE and abs(position_before) > QTY_TOLERANCE:
        return "COMPLETE_HISTORY_CYCLE"
    if abs(position_after) > QTY_TOLERANCE and abs(position_after) <= RESIDUAL_DUST_QTY:
        return "RESIDUAL_DUST"
    if abs(position_before) > QTY_TOLERANCE and abs(position_after) > QTY_TOLERANCE:
        return "CARRIED_POSITION_CONFIRMED"
    if initial_history_date and abs(position_before) <= QTY_TOLERANCE and position_after < -QTY_TOLERANCE:
        return "OPENING_INVENTORY_UNKNOWN"
    if latest_position_qty is not None and abs(final_reconstructed_qty - latest_position_qty) > QTY_TOLERANCE:
        return "POSSIBLE_MISSING_EXECUTION"
    if current_open_qty > QTY_TOLERANCE:
        return "CARRIED_POSITION_CONFIRMED"
    return "COMPLETE_HISTORY_CYCLE"


def build_inventory_events(conn: sqlite3.Connection, symbol: str) -> tuple[list[InventoryEvent], dict[str, Any]]:
    executions = execution_rows(conn, symbol)
    rebuild = build_canonical_fifo(executions, symbol=symbol)
    executions = sort_execution_rows(executions)
    latest_qty, latest_position_raw = latest_position_quantity(conn, symbol)
    running = 0.0
    events: list[InventoryEvent] = []
    initial_history_date = date_part(event_time(executions[0])) if executions else ""
    for row in executions:
        side = norm_side(row.get("side"))
        qty = abs(fnum(row.get("quantity")) or 0.0)
        if qty <= QTY_TOLERANCE:
            continue
        before = running
        if side == "BUY":
            running += qty
        elif side == "SELL":
            running -= qty
        else:
            continue
        after = running
        transition = ""
        if abs(before) <= QTY_TOLERANCE and after > QTY_TOLERANCE:
            transition = "ZERO_TO_POSITIVE"
        elif before > QTY_TOLERANCE and abs(after) <= QTY_TOLERANCE:
            transition = "POSITIVE_TO_ZERO"
        elif side == "SELL":
            transition = "SELL_PARTIAL_OR_CARRY"
        elif side == "BUY":
            transition = "BUY_ADD_LOT"
        status = classify_cycle(
            initial_history_date=initial_history_date,
            position_before=before,
            position_after=after,
            current_open_qty=rebuild.open_quantity,
            final_reconstructed_qty=running,
            latest_position_qty=latest_qty,
        )
        events.append(
            InventoryEvent(
                symbol=symbol,
                event_type=transition,
                event_time=event_time(row),
                execution_id=str(row.get("execution_id") or ""),
                side=side,
                execution_qty=qty,
                position_before=before,
                position_after=after,
                status=status,
            )
        )
    summary = {
        "symbol": symbol,
        "initial_history_date": initial_history_date,
        "execution_count": len(executions),
        "final_reconstructed_quantity": running,
        "canonical_open_quantity": rebuild.open_quantity,
        "latest_position_quantity": latest_qty,
        "latest_position_raw": latest_position_raw,
        "unmatched_sell_count": len(rebuild.unmatched_sells),
        "unmatched_sell_quantity": sum(float(row.get("unmatched_sell_quantity") or 0.0) for row in rebuild.unmatched_sells),
        "transition_count": len(events),
        **rebuild.timestamp_diagnostics,
    }
    return events, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summaries: list[dict[str, Any]], component_details: list[dict[str, Any]]) -> None:
    lines = [
        "# Inventory Continuity Diagnostic",
        "",
        "This report separates accounting FIFO inventory from logical strategy lifecycle / entry order reporting.",
        "",
        "## Symbol Summary",
        "",
    ]
    for row in summaries:
        lines.append(
            "- {symbol}: initial_history_date={initial_history_date} final_reconstructed_quantity={final_reconstructed_quantity} "
            "latest_position_quantity={latest_position_quantity} unmatched_sell_quantity={unmatched_sell_quantity}".format(**row)
        )
    lines.extend(["", "## Component Details", ""])
    if component_details:
        lines.append("| symbol | canonical_trade_id | buy_execution_id | buy_time | sell_execution_id | sell_time | matched_qty |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in component_details:
            lines.append(
                f"| {row.get('symbol','')} | {row.get('canonical_trade_id','')} | {row.get('buy_execution_id','')} | "
                f"{row.get('buy_time','')} | {row.get('sell_execution_id','')} | {row.get('sell_time','')} | {row.get('matched_qty','')} |"
            )
    else:
        lines.append("_No component rows._")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only inventory continuity diagnostic from SQLite executions/components.")
    parser.add_argument("--date")
    parser.add_argument("--symbol", action="append", help="Symbol to inspect. Can be repeated.")
    parser.add_argument("--symbols", help="Comma-separated symbols to inspect.")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_SQLITE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    symbols = list(args.symbol or [])
    if args.symbols:
        symbols.extend(part.strip() for part in args.symbols.split(",") if part.strip())
    sqlite_path = Path(args.sqlite_path)
    suffix = args.date or ("_".join(sorted(norm_symbol(x) for x in symbols)) if symbols else "all")
    with connect_sqlite(sqlite_path, read_only=True) as conn:
        selected = selected_symbols(conn, date=args.date, symbols=symbols)
        all_events: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        components: list[dict[str, Any]] = []
        for symbol in selected:
            events, summary = build_inventory_events(conn, symbol)
            summaries.append(summary)
            all_events.extend([event.__dict__ for event in events])
            components.extend(component_rows(conn, symbol, args.date))

    output_dir = Path(args.output_dir)
    events_path = output_dir / f"inventory_continuity_events_{suffix}.csv"
    summary_csv_path = output_dir / f"inventory_continuity_summary_{suffix}.csv"
    components_path = output_dir / f"inventory_continuity_components_{suffix}.csv"
    summary_md_path = output_dir / f"inventory_continuity_summary_{suffix}.md"
    write_csv(events_path, all_events)
    write_csv(summary_csv_path, summaries)
    write_csv(components_path, components)
    write_summary(summary_md_path, summaries, components)
    print(
        "INVENTORY_CONTINUITY_DONE "
        f"symbols={len(selected)} events={len(all_events)} components={len(components)} "
        f"events_output={events_path} summary={summary_md_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
