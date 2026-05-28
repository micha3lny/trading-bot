#!/usr/bin/env python3
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


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def active_symbols_from_managed(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    payload = parse_jsonish(path.read_text(encoding="utf-8"))
    positions = payload.get("positions") or {}
    if isinstance(positions, dict):
        iterable = positions.items()
    elif isinstance(positions, list):
        iterable = [(str(item.get("symbol") or ""), item) for item in positions if isinstance(item, dict)]
    else:
        return set()
    out: set[str] = set()
    for symbol, row in iterable:
        if not isinstance(row, dict):
            continue
        if bool(row.get("active", True)) and abs(float(row.get("quantity") or row.get("ibkr_quantity") or 0.0)) > 0:
            out.add(str(symbol or row.get("symbol") or "").upper())
    return {s for s in out if s}


def active_symbols_from_portfolio(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    latest: dict[str, Any] | None = None
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            latest = row
    if not latest:
        return set()
    positions = latest.get("positions_json")
    try:
        parsed = json.loads(positions or "[]")
    except Exception:
        parsed = []
    out: set[str] = set()
    if isinstance(parsed, dict):
        parsed = list(parsed.values())
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        contract = item.get("contract") if isinstance(item.get("contract"), dict) else {}
        symbol = str(item.get("symbol") or contract.get("symbol") or "").upper()
        qty = float(item.get("position") or item.get("quantity") or 0.0)
        if symbol and abs(qty) > 0:
            out.add(symbol)
    return out


def find_stale_rows(store: SQLiteRuntimeStore, keep_symbols: set[str]) -> list[dict[str, Any]]:
    rows = store.query(
        """
        SELECT position_key, strategy_name, session_date, symbol, status, quantity, ibkr_quantity, updated_at, raw_json
        FROM positions
        WHERE COALESCE(active, 0) = 1
        ORDER BY COALESCE(updated_at, '') DESC, symbol
        """
    )
    stale: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        raw = parse_jsonish(row.get("raw_json"))
        if symbol in keep_symbols:
            continue
        if bool(raw.get("ibkr_position_flat_confirmed")):
            stale.append(row)
            continue
        stale.append(row)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run/apply cleanup for stale active SQLite positions.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--managed-positions-json", default=None)
    parser.add_argument("--portfolio-snapshots-csv", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reason", default="cleanup_stale_positions")
    args = parser.parse_args()

    managed_path = Path(args.managed_positions_json) if args.managed_positions_json else None
    portfolio_path = Path(args.portfolio_snapshots_csv) if args.portfolio_snapshots_csv else None
    keep_symbols = active_symbols_from_managed(managed_path) | active_symbols_from_portfolio(portfolio_path)

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    try:
        stale = find_stale_rows(store, keep_symbols)
        print(
            f"STALE_POSITIONS_CLEANUP dry_run={int(not args.apply)} "
            f"active_reference_symbols={len(keep_symbols)} stale_rows={len(stale)}"
        )
        for row in stale[:200]:
            print(
                "STALE_POSITION_ROW "
                f"position_key={row.get('position_key')} symbol={row.get('symbol')} "
                f"session_date={row.get('session_date')} status={row.get('status')} "
                f"quantity={row.get('quantity')} ibkr_quantity={row.get('ibkr_quantity')} "
                f"updated_at={row.get('updated_at')}"
            )
        if len(stale) > 200:
            print(f"STALE_POSITION_ROW_SUPPRESSED count={len(stale) - 200}")
        if args.apply:
            for row in stale:
                store.mark_position_flat(
                    symbol=str(row.get("symbol") or ""),
                    strategy_name=row.get("strategy_name"),
                    session_date=row.get("session_date"),
                    reason=args.reason,
                    status="FLAT_CONFIRMED",
                )
            print(f"STALE_POSITIONS_CLEANUP_APPLIED rows={len(stale)}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
