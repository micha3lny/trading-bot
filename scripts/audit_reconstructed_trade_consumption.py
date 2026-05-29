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

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, parse_jsonish, resolve_sqlite_path


def as_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def audit_consumption(store: SQLiteRuntimeStore, date: str | None = None) -> dict[str, Any]:
    params: list[Any] = []
    date_clause = ""
    if date:
        date_clause = "AND (substr(exit_fill_time, 1, 10) = ? OR substr(closed_at, 1, 10) = ?)"
        params.extend([date, date])
    trades = store.query(
        f"""
        SELECT trade_id, symbol, quantity, entry_fill_time, exit_fill_time, raw_json
        FROM trades
        WHERE UPPER(COALESCE(status, '')) IN ('CLOSED', 'DONE', 'EXIT_FILLED', 'FLAT')
          AND (
            trade_id LIKE 'reconstructed:%'
            OR COALESCE(raw_json, '') LIKE '%buy_execution_id%'
            OR COALESCE(raw_json, '') LIKE '%sell_execution_id%'
          )
          {date_clause}
        ORDER BY symbol, entry_fill_time, exit_fill_time, trade_id
        """,
        params,
    )
    executions = {
        str(row.get("execution_id") or ""): as_float(row.get("quantity"))
        for row in store.query("SELECT execution_id, quantity FROM executions")
    }
    pair_counts: dict[tuple[str, str], list[str]] = {}
    consumed: dict[str, float] = {}
    missing_execution_ids: set[str] = set()
    for trade in trades:
        raw = parse_jsonish(trade.get("raw_json"))
        buy_exec = str(raw.get("buy_execution_id") or "")
        sell_exec = str(raw.get("sell_execution_id") or "")
        qty = as_float(raw.get("matched_quantity")) or as_float(trade.get("quantity"))
        if buy_exec and sell_exec:
            pair_counts.setdefault((buy_exec, sell_exec), []).append(str(trade.get("trade_id") or ""))
        for exec_id in (buy_exec, sell_exec):
            if not exec_id:
                continue
            if exec_id not in executions:
                missing_execution_ids.add(exec_id)
            consumed[exec_id] = consumed.get(exec_id, 0.0) + qty

    duplicate_pairs = [
        {"buy_execution_id": buy, "sell_execution_id": sell, "trade_ids": ids}
        for (buy, sell), ids in sorted(pair_counts.items())
        if len(ids) > 1
    ]
    over_consumed = [
        {
            "execution_id": exec_id,
            "execution_quantity": executions.get(exec_id, 0.0),
            "consumed_quantity": qty,
        }
        for exec_id, qty in sorted(consumed.items())
        if exec_id in executions and qty > executions[exec_id] + 1e-8
    ]
    return {
        "date": date,
        "trades_checked": len(trades),
        "duplicate_pair_count": len(duplicate_pairs),
        "over_consumed_count": len(over_consumed),
        "missing_execution_id_count": len(missing_execution_ids),
        "duplicate_pairs": duplicate_pairs,
        "over_consumed": over_consumed,
        "missing_execution_ids": sorted(missing_execution_ids),
        "ok": not duplicate_pairs and not over_consumed and not missing_execution_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reconstructed trades for reused or over-consumed execution quantities.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    try:
        result = audit_consumption(store, args.date)
    finally:
        store.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
