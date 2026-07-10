#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")


def rows_by_symbol_for_date(store: SQLiteRuntimeStore, session_date: str | None) -> list[str]:
    if not session_date:
        rows = store.query(
            """
            SELECT symbol FROM executions WHERE COALESCE(symbol, '') != ''
            UNION
            SELECT symbol FROM trades WHERE COALESCE(symbol, '') != ''
            ORDER BY symbol
            """
        )
    else:
        rows = store.query(
            """
            SELECT symbol
            FROM executions
            WHERE COALESCE(symbol, '') != ''
              AND (
                session_date = ?
                OR substr(COALESCE(executed_at, ''), 1, 10) = ?
                OR substr(COALESCE(recorded_at, ''), 1, 10) = ?
              )
            UNION
            SELECT symbol
            FROM trades
            WHERE COALESCE(symbol, '') != ''
              AND (
                session_date = ?
                OR substr(COALESCE(entry_fill_time, ''), 1, 10) = ?
                OR substr(COALESCE(exit_fill_time, ''), 1, 10) = ?
                OR substr(COALESCE(closed_at, ''), 1, 10) = ?
              )
            ORDER BY symbol
            """,
            [session_date, session_date, session_date, session_date, session_date, session_date, session_date],
        )
    return sorted({str(row.get("symbol") or "").upper() for row in rows if row.get("symbol")})


def suspicious_trade_groups(store: SQLiteRuntimeStore, session_date: str | None, symbols: list[str]) -> list[dict[str, Any]]:
    params: list[Any] = []
    filters = ["COALESCE(entry_order_id, '') != ''"]
    if session_date:
        filters.append(
            """
            (
              session_date = ?
              OR substr(COALESCE(entry_fill_time, ''), 1, 10) = ?
              OR substr(COALESCE(exit_fill_time, ''), 1, 10) = ?
              OR substr(COALESCE(closed_at, ''), 1, 10) = ?
            )
            """
        )
        params.extend([session_date, session_date, session_date, session_date])
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        filters.append(f"UPPER(symbol) IN ({placeholders})")
        params.extend(symbols)
    rows = store.query(
        f"""
        SELECT
            UPPER(symbol) AS symbol,
            entry_order_id,
            COUNT(*) AS rows,
            SUM(CASE WHEN trade_id LIKE 'entry:%' THEN 1 ELSE 0 END) AS entry_rows,
            SUM(CASE WHEN trade_id LIKE 'reconstructed:%' OR raw_json LIKE '%sqlite_execution_reducer%' THEN 1 ELSE 0 END) AS reconstructed_rows,
            SUM(CASE WHEN UPPER(COALESCE(status, '')) = 'CLOSED' THEN 1 ELSE 0 END) AS closed_rows,
            GROUP_CONCAT(trade_id) AS trade_ids
        FROM trades
        WHERE {' AND '.join(filters)}
        GROUP BY UPPER(symbol), entry_order_id
        HAVING rows > 1 OR (entry_rows > 0 AND reconstructed_rows > 0)
        ORDER BY symbol, entry_order_id
        """,
        params,
    )
    return [dict(row) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair SQLite canonical trade reconstruction from executions. Dry-run by default."
    )
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--date", help="Session/exit date to repair, YYYY-MM-DD.")
    parser.add_argument("--symbol", action="append", default=[], help="Limit to one symbol. Can be passed multiple times.")
    parser.add_argument("--apply", action="store_true", help="Apply rebuild. Without this flag the script is read-only.")
    parser.add_argument(
        "--allow-historical-open-lots",
        action="store_true",
        help="Allow historical unmatched BUY lots to remain active. Default keeps live-safe suppression.",
    )
    args = parser.parse_args(argv)

    store = SQLiteRuntimeStore(args.sqlite_path)
    try:
        symbols = sorted({str(symbol).upper().strip() for symbol in args.symbol if str(symbol).strip()})
        if not symbols:
            symbols = rows_by_symbol_for_date(store, args.date)
        suspicious_before = suspicious_trade_groups(store, args.date, symbols)
        result: dict[str, Any] = {
            "apply": bool(args.apply),
            "date": args.date,
            "sqlite_path": str(args.sqlite_path),
            "symbols_count": len(symbols),
            "symbols": symbols,
            "suspicious_groups_before_count": len(suspicious_before),
            "suspicious_groups_before": suspicious_before[:50],
            "rebuilt": [],
        }
        if args.apply:
            for symbol in symbols:
                result["rebuilt"].append(
                    store.rebuild_symbol_trade_state(
                        symbol,
                        allow_historical_open_lots=args.allow_historical_open_lots,
                    )
                )
            suspicious_after = suspicious_trade_groups(store, args.date, symbols)
            result["suspicious_groups_after_count"] = len(suspicious_after)
            result["suspicious_groups_after"] = suspicious_after[:50]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
