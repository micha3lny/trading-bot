#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.canonical_fifo import build_canonical_fifo
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, connect_sqlite


DEFAULT_DB = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_OUTPUT_DIR = Path("data/analysis")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def selected_symbols(conn: sqlite3.Connection, date: str | None, symbol: str | None) -> list[str]:
    if symbol:
        return [symbol.upper()]
    where = ""
    params: list[Any] = []
    if date:
        where = "WHERE session_date = ? OR substr(COALESCE(executed_at, recorded_at), 1, 10) = ?"
        params = [date, date]
    rows = conn.execute(f"SELECT DISTINCT UPPER(symbol) AS symbol FROM executions {where} ORDER BY symbol", params).fetchall()
    return [str(row[0] or "").upper() for row in rows if row[0]]


def execution_rows(conn: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT execution_id, trade_id, order_key, order_id, perm_id,
               COALESCE(strategy_name, 'unknown') AS strategy_name,
               session_date, symbol, side, quantity, price, exchange, liquidity,
               executed_at, recorded_at, commission, commission_currency,
               realized_pnl, commission_source, exit_reason, exit_reason_source, raw_json
        FROM executions
        WHERE UPPER(symbol) = ?
        ORDER BY COALESCE(executed_at, recorded_at, ''), COALESCE(recorded_at, ''), execution_id
        """,
        (symbol.upper(),),
    ).fetchall()
    return [dict(row) for row in rows]


def current_trade_count(conn: sqlite3.Connection, symbol: str | None = None, date: str | None = None) -> int:
    if not table_exists(conn, "trades"):
        return 0
    clauses: list[str] = []
    params: list[Any] = []
    if symbol:
        clauses.append("UPPER(symbol) = ?")
        params.append(symbol.upper())
    if date:
        clauses.append("(session_date = ? OR substr(COALESCE(exit_fill_time, closed_at, entry_fill_time), 1, 10) = ?)")
        params.extend([date, date])
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    row = conn.execute(f"SELECT COUNT(*) FROM trades {where}", params).fetchone()
    return int(row[0] or 0)


def trader_process_running() -> bool:
    try:
        result = subprocess.run(["pgrep", "-af", "v67_live_top100_expansion_paper_trader|v67-trader"], capture_output=True, text=True, timeout=5)
    except Exception:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def production_db_path() -> Path:
    return REPO_ROOT / DEFAULT_DB


def resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()


def is_production_db_path(sqlite_path: Path) -> bool:
    return resolved_path(sqlite_path) == resolved_path(production_db_path())


def enforce_apply_guard(sqlite_path: Path) -> None:
    absolute_path = resolved_path(sqlite_path)
    if is_production_db_path(sqlite_path):
        if trader_process_running():
            raise RuntimeError("v67 trader process appears active on production DB; refusing --apply")
        return
    print(f"NON_PRODUCTION_DATABASE_APPLY sqlite_path={absolute_path}", flush=True)


def build_plan(sqlite_path: Path, *, date: str | None, symbol: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with connect_sqlite(sqlite_path, read_only=True) as conn:
        symbols = selected_symbols(conn, date, symbol)
        rows: list[dict[str, Any]] = []
        summary = {
            "symbols": len(symbols),
            "current_trades": current_trade_count(conn, symbol=symbol, date=date),
            "planned_trades": 0,
            "planned_components": 0,
            "unmatched_sell_count": 0,
            "unmatched_sell_quantity": 0.0,
            "mixed_timestamp_format_symbols": 0,
            "timestamp_parse_failures": 0,
            "raw_string_order_diff_symbols": 0,
        }
        for sym in symbols:
            executions = execution_rows(conn, sym)
            rebuild = build_canonical_fifo(executions, symbol=sym)
            ts_diag = rebuild.timestamp_diagnostics
            matching_trades = [
                trade for trade in rebuild.trades
                if not date or trade.session_date == date or str(trade.exit_time).startswith(date)
            ]
            matching_components = [
                component for component in rebuild.components
                if not date or component.session_date == date or str(component.exit_time).startswith(date)
            ]
            unmatched = [
                item for item in rebuild.unmatched_sells
                if not date or str(item.get("session_date") or "").startswith(date) or str(item.get("executed_at") or item.get("recorded_at") or "").startswith(date)
            ]
            summary["planned_trades"] += len(matching_trades)
            summary["planned_components"] += len(matching_components)
            summary["unmatched_sell_count"] += len(unmatched)
            summary["unmatched_sell_quantity"] += sum(float(item.get("unmatched_sell_quantity") or 0.0) for item in unmatched)
            summary["mixed_timestamp_format_symbols"] += int(bool(ts_diag.get("mixed_timestamp_formats")))
            summary["timestamp_parse_failures"] += int(ts_diag.get("timestamp_parse_failures") or 0)
            summary["raw_string_order_diff_symbols"] += int(bool(ts_diag.get("raw_string_order_differs_from_parsed")))
            rows.append(
                {
                    "symbol": sym,
                    "execution_rows": len(executions),
                    "planned_trades": len(matching_trades),
                    "planned_components": len(matching_components),
                    "open_quantity": rebuild.open_quantity,
                    "unmatched_sell_count": len(unmatched),
                    "unmatched_sell_quantity": sum(float(item.get("unmatched_sell_quantity") or 0.0) for item in unmatched),
                    "unmatched_sell_execution_ids": ",".join(str(item.get("execution_id") or "") for item in unmatched),
                    "mixed_timestamp_formats": int(bool(ts_diag.get("mixed_timestamp_formats"))),
                    "timestamp_parse_failures": int(ts_diag.get("timestamp_parse_failures") or 0),
                    "raw_string_order_differs_from_parsed": int(bool(ts_diag.get("raw_string_order_differs_from_parsed"))),
                }
            )
    return rows, summary


def write_reports(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path, suffix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"repair_trades_from_executions_dry_run_{suffix}.csv"
    summary_path = output_dir / f"repair_trades_from_executions_dry_run_{suffix}.md"
    import csv

    fieldnames = list(rows[0].keys()) if rows else ["symbol", "execution_rows", "planned_trades", "planned_components", "open_quantity", "unmatched_sell_count", "unmatched_sell_quantity", "unmatched_sell_execution_ids"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Repair Trades From Executions Dry Run",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Unmatched SELL Quantities",
        "",
    ]
    unmatched = [row for row in rows if float(row.get("unmatched_sell_quantity") or 0.0) > 0]
    if unmatched:
        for row in unmatched:
            lines.append(f"- {row['symbol']}: qty={row['unmatched_sell_quantity']} sells={row['unmatched_sell_execution_ids']}")
    else:
        lines.append("- none")
    summary_path.write_text("\n".join(lines) + "\n")
    return csv_path, summary_path


def apply_repair(sqlite_path: Path, *, date: str | None, symbol: str | None) -> dict[str, Any]:
    enforce_apply_guard(sqlite_path)
    backup_path = sqlite_path.with_suffix(sqlite_path.suffix + f".backup_{utc_stamp()}")
    shutil.copy2(sqlite_path, backup_path)
    store = SQLiteRuntimeStore(sqlite_path)
    try:
        symbols = selected_symbols(store.conn, date, symbol)
        before = current_trade_count(store.conn, symbol=symbol, date=date)
        with store.transaction():
            for sym in symbols:
                store.rebuild_symbol_trade_state(sym, allow_historical_open_lots=True)
        after = current_trade_count(store.conn, symbol=symbol, date=date)
        return {"backup_path": str(backup_path), "symbols": len(symbols), "trades_before": before, "trades_after": after}
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run or repair SQLite trades from execution-ledger canonical FIFO.")
    parser.add_argument("--date", help="Report/repair trades whose canonical trade exits or sessions intersect this date.")
    parser.add_argument("--symbol")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_DB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--apply", action="store_true", help="Apply repair. Default is dry-run only.")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_path)
    suffix = args.date or (args.symbol.upper() if args.symbol else "all")
    rows, summary = build_plan(sqlite_path, date=args.date, symbol=args.symbol)
    csv_path, summary_path = write_reports(rows, summary, Path(args.output_dir), suffix)
    print(
        "REPAIR_TRADES_FROM_EXECUTIONS_DRY_RUN "
        f"symbols={summary['symbols']} current_trades={summary['current_trades']} "
        f"planned_trades={summary['planned_trades']} planned_components={summary['planned_components']} "
        f"unmatched_sell_count={summary['unmatched_sell_count']} "
        f"unmatched_sell_quantity={summary['unmatched_sell_quantity']} "
        f"output={csv_path} summary={summary_path}",
        flush=True,
    )
    if not args.apply:
        return 0
    result = apply_repair(sqlite_path, date=args.date, symbol=args.symbol)
    print("REPAIR_TRADES_FROM_EXECUTIONS_APPLIED " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
