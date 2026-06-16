#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, resolve_sqlite_path


def ledger_net_positions(store: SQLiteRuntimeStore, symbols: list[str] | None = None) -> dict[str, float]:
    params: list[Any] = []
    symbol_filter = ""
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        symbol_filter = f"WHERE UPPER(symbol) IN ({placeholders})"
        params = [symbol.upper() for symbol in symbols]
    rows = store.query(
        f"""
        SELECT UPPER(symbol) AS symbol,
               SUM(CASE
                   WHEN UPPER(COALESCE(side, '')) IN ('BOT', 'BUY') THEN COALESCE(quantity, 0)
                   WHEN UPPER(COALESCE(side, '')) IN ('SLD', 'SELL') THEN -COALESCE(quantity, 0)
                   ELSE 0
               END) AS net_qty
        FROM executions
        {symbol_filter}
        GROUP BY UPPER(symbol)
        HAVING ABS(net_qty) > 0.000001
        ORDER BY UPPER(symbol)
        """,
        params,
    )
    return {str(row["symbol"]).upper(): float(row["net_qty"] or 0.0) for row in rows}


def sqlite_active_positions(store: SQLiteRuntimeStore, symbols: list[str] | None = None) -> dict[str, float]:
    params: list[Any] = []
    symbol_filter = ""
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        symbol_filter = f"AND UPPER(symbol) IN ({placeholders})"
        params = [symbol.upper() for symbol in symbols]
    rows = store.query(
        f"""
        SELECT UPPER(symbol) AS symbol,
               SUM(COALESCE(ibkr_quantity, quantity, 0)) AS active_qty,
               COUNT(*) AS rows_count
        FROM positions
        WHERE COALESCE(active, 0) = 1
          AND UPPER(COALESCE(status, '')) IN ('OPEN', 'EXIT_ORDER')
          {symbol_filter}
        GROUP BY UPPER(symbol)
        ORDER BY UPPER(symbol)
        """,
        params,
    )
    return {str(row["symbol"]).upper(): float(row["active_qty"] or 0.0) for row in rows}


def diff_positions(ledger: dict[str, float], active: dict[str, float]) -> list[dict[str, Any]]:
    symbols = sorted(set(ledger) | set(active))
    diffs: list[dict[str, Any]] = []
    for symbol in symbols:
        ledger_qty = ledger.get(symbol, 0.0)
        active_qty = active.get(symbol, 0.0)
        if abs(ledger_qty - active_qty) <= 1e-6:
            continue
        diffs.append(
            {
                "symbol": symbol,
                "ledger_net_qty": ledger_qty,
                "sqlite_active_qty": active_qty,
                "difference": active_qty - ledger_qty,
            }
        )
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair SQLite positions.active state from immutable executions ledger.")
    parser.add_argument("--sqlite-path", default=resolve_sqlite_path(None))
    parser.add_argument("--symbol", action="append", default=[], help="Limit repair to one symbol; repeatable.")
    parser.add_argument("--allow-historical-open-lots", action="store_true", help="Allow old unmatched BUY lots to become active.")
    parser.add_argument("--apply", action="store_true", help="Apply repair. Without this flag, only reports differences.")
    args = parser.parse_args()

    symbols = sorted({symbol.upper().strip() for symbol in args.symbol if symbol.strip()}) or None
    store = SQLiteRuntimeStore(args.sqlite_path)
    try:
        before_ledger = ledger_net_positions(store, symbols)
        before_active = sqlite_active_positions(store, symbols)
        before_diff = diff_positions(before_ledger, before_active)
        result: dict[str, Any] = {
            "apply": args.apply,
            "sqlite_path": str(args.sqlite_path),
            "symbols_limited": symbols or [],
            "mismatches_before": len(before_diff),
            "mismatch_rows_before": before_diff,
        }
        if args.apply:
            repair = store.rebuild_positions_from_executions(
                symbols,
                allow_historical_open_lots=args.allow_historical_open_lots,
            )
            after_ledger = ledger_net_positions(store, symbols)
            after_active = sqlite_active_positions(store, symbols)
            after_diff = diff_positions(after_ledger, after_active)
            result.update(
                {
                    "repair": repair,
                    "mismatches_after": len(after_diff),
                    "mismatch_rows_after": after_diff,
                }
            )
            store.record_runtime_event(
                event_type="POSITION_REDUCER_REPAIR",
                source="repair_positions_from_executions",
                reason="manual_repair",
                raw_json=result,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result.get("mismatches_after") else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
