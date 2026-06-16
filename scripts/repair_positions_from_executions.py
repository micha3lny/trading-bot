#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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


def diff_positions(target: dict[str, float], active: dict[str, float]) -> list[dict[str, Any]]:
    symbols = sorted(set(target) | set(active))
    diffs: list[dict[str, Any]] = []
    for symbol in symbols:
        target_qty = target.get(symbol, 0.0)
        active_qty = active.get(symbol, 0.0)
        if abs(target_qty - active_qty) <= 1e-6:
            continue
        diffs.append(
            {
                "symbol": symbol,
                "target_qty": target_qty,
                "sqlite_active_qty": active_qty,
                "difference": active_qty - target_qty,
            }
        )
    return diffs


def load_broker_positions_csv(path: str | Path) -> dict[str, float]:
    rows = list(csv.DictReader(Path(path).read_text().splitlines()))
    positions: dict[str, float] = {}
    for row in rows:
        symbol = str(row.get("symbol") or row.get("Symbol") or row.get("contract.symbol") or "").upper().strip()
        qty_value = row.get("quantity") or row.get("Quantity") or row.get("position") or row.get("Position") or row.get("qty")
        if not symbol or qty_value in (None, ""):
            continue
        positions[symbol] = positions.get(symbol, 0.0) + float(qty_value)
    return {symbol: qty for symbol, qty in positions.items() if abs(qty) > 1e-9}


def load_broker_positions_json(path: str | Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if "positions" in data:
            data = data["positions"]
        else:
            return {str(symbol).upper(): float(qty) for symbol, qty in data.items() if abs(float(qty)) > 1e-9}
    positions: dict[str, float] = {}
    for row in data if isinstance(data, list) else []:
        symbol = str(row.get("symbol") or row.get("Symbol") or "").upper().strip()
        qty_value = row.get("quantity") or row.get("Quantity") or row.get("position") or row.get("Position") or row.get("qty")
        if not symbol or qty_value in (None, ""):
            continue
        positions[symbol] = positions.get(symbol, 0.0) + float(qty_value)
    return {symbol: qty for symbol, qty in positions.items() if abs(qty) > 1e-9}


def fetch_broker_positions(*, host: str, port: int, client_id: int, timeout: float) -> dict[str, float]:
    from ib_insync import IB  # type: ignore

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        positions: dict[str, float] = {}
        for pos in ib.positions():
            symbol = str(getattr(pos.contract, "symbol", "") or "").upper().strip()
            qty = float(getattr(pos, "position", 0.0) or 0.0)
            if not symbol or abs(qty) <= 1e-9:
                continue
            positions[symbol] = positions.get(symbol, 0.0) + qty
        return positions
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair SQLite positions.active state from immutable executions ledger.")
    parser.add_argument("--sqlite-path", default=resolve_sqlite_path(None))
    parser.add_argument("--symbol", action="append", default=[], help="Limit repair to one symbol; repeatable.")
    parser.add_argument("--allow-historical-open-lots", action="store_true", help="Allow old unmatched BUY lots to become active.")
    parser.add_argument("--broker-positions-csv", help="CSV snapshot with symbol and quantity/position columns.")
    parser.add_argument("--broker-positions-json", help="JSON broker snapshot, either {SYMBOL: qty} or {'positions': [...]}.")
    parser.add_argument("--fetch-broker-positions", action="store_true", help="Fetch current IBKR portfolio and constrain active positions to broker quantities.")
    parser.add_argument("--broker-host", default="127.0.0.1")
    parser.add_argument("--broker-port", type=int, default=4002)
    parser.add_argument("--broker-client-id", type=int, default=178)
    parser.add_argument("--broker-timeout", type=float, default=5.0)
    parser.add_argument("--apply", action="store_true", help="Apply repair. Without this flag, only reports differences.")
    args = parser.parse_args()

    symbols = sorted({symbol.upper().strip() for symbol in args.symbol if symbol.strip()}) or None
    broker_positions: dict[str, float] | None = None
    broker_source = ""
    if args.broker_positions_csv:
        broker_positions = load_broker_positions_csv(args.broker_positions_csv)
        broker_source = f"csv:{args.broker_positions_csv}"
    if args.broker_positions_json:
        loaded = load_broker_positions_json(args.broker_positions_json)
        broker_positions = loaded if broker_positions is None else {**broker_positions, **loaded}
        broker_source = f"{broker_source + ',' if broker_source else ''}json:{args.broker_positions_json}"
    if args.fetch_broker_positions:
        fetched = fetch_broker_positions(
            host=args.broker_host,
            port=args.broker_port,
            client_id=args.broker_client_id,
            timeout=args.broker_timeout,
        )
        broker_positions = fetched if broker_positions is None else {**broker_positions, **fetched}
        broker_source = f"{broker_source + ',' if broker_source else ''}ibkr:{args.broker_host}:{args.broker_port}:{args.broker_client_id}"
    if symbols and broker_positions is not None:
        broker_positions = {symbol: qty for symbol, qty in broker_positions.items() if symbol in set(symbols)}

    store = SQLiteRuntimeStore(args.sqlite_path)
    try:
        before_ledger = ledger_net_positions(store, symbols)
        before_active = sqlite_active_positions(store, symbols)
        target_positions = broker_positions if broker_positions is not None else before_ledger
        before_diff = diff_positions(target_positions, before_active)
        result: dict[str, Any] = {
            "apply": args.apply,
            "sqlite_path": str(args.sqlite_path),
            "symbols_limited": symbols or [],
            "target": "broker_positions" if broker_positions is not None else "executions_ledger_net",
            "broker_source": broker_source,
            "broker_positions_count": len(broker_positions or {}),
            "broker_positions_qty_sum": sum((broker_positions or {}).values()),
            "mismatches_before": len(before_diff),
            "mismatch_rows_before": before_diff,
        }
        if args.apply:
            repair = store.rebuild_positions_from_executions(
                symbols,
                allow_historical_open_lots=args.allow_historical_open_lots,
                broker_net_positions=broker_positions,
            )
            after_ledger = ledger_net_positions(store, symbols)
            after_active = sqlite_active_positions(store, symbols)
            after_target = broker_positions if broker_positions is not None else after_ledger
            after_diff = diff_positions(after_target, after_active)
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
