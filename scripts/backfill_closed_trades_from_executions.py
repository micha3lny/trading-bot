#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore, parse_jsonish, resolve_sqlite_path


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def date_part(value: Any) -> str:
    dt = parse_dt(value)
    if dt is not None:
        return dt.strftime("%F")
    text = str(value or "")
    return text[:10] if len(text) >= 10 else ""


def as_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def normalized_side(value: Any) -> str:
    text = str(value or "").upper()
    if text in {"BOT", "BUY"}:
        return "BUY"
    if text in {"SLD", "SELL"}:
        return "SELL"
    return text


def execution_time(row: dict[str, Any]) -> str:
    raw = parse_jsonish(row.get("raw_json"))
    execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
    raw_time = execution.get("time") or execution.get("executionTime") or raw.get("executed_at") or raw.get("execution_time") or raw.get("time")
    return str(row.get("executed_at") or raw_time or row.get("recorded_at") or "")


def commission_value(row: dict[str, Any], fraction: float) -> tuple[float, bool]:
    if str(row.get("commission_source") or "").lower() != "ibkr":
        return 0.0, False
    value = row.get("commission")
    if value in (None, ""):
        return 0.0, False
    return abs(as_float(value)) * fraction, True


def trade_id_for(entry_date: str, exit_date: str, symbol: str, buy_exec_id: str, sell_exec_id: str) -> str:
    return f"reconstructed:{entry_date}:{exit_date}:{symbol}:{buy_exec_id}:{sell_exec_id}"


def existing_trade_fingerprints(store: SQLiteRuntimeStore, *, include_reconstructed: bool = False) -> tuple[set[str], set[tuple[str, str, str, str, str, str]]]:
    reconstructed_filter = "" if include_reconstructed else """
          AND trade_id NOT LIKE 'reconstructed:%'
          AND COALESCE(raw_json, '') NOT LIKE '%executions_pair_repair%'
          AND COALESCE(raw_json, '') NOT LIKE '%sqlite_execution_reducer%'
    """
    rows = store.query(
        f"""
        SELECT trade_id, symbol, entry_fill_time, exit_fill_time, entry_price, exit_price, quantity, raw_json
        FROM trades
        WHERE UPPER(COALESCE(status, '')) IN ('CLOSED', 'DONE', 'EXIT_FILLED', 'FLAT')
        {reconstructed_filter}
        """
    )
    trade_ids: set[str] = set()
    fingerprints: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        trade_id = str(row.get("trade_id") or "")
        if trade_id:
            trade_ids.add(trade_id)
        raw = parse_jsonish(row.get("raw_json"))
        buy_exec = str(raw.get("buy_execution_id") or "")
        sell_exec = str(raw.get("sell_execution_id") or "")
        if buy_exec and sell_exec:
            trade_ids.add(trade_id_for(date_part(row.get("entry_fill_time")), date_part(row.get("exit_fill_time")), str(row.get("symbol") or "").upper(), buy_exec, sell_exec))
        fingerprints.add(
            (
                str(row.get("symbol") or "").upper(),
                str(row.get("entry_fill_time") or ""),
                str(row.get("exit_fill_time") or ""),
                f"{as_float(row.get('quantity')):.8f}",
                f"{as_float(row.get('entry_price')):.8f}",
                f"{as_float(row.get('exit_price')):.8f}",
            )
        )
    return trade_ids, fingerprints


def delete_reconstructed_for_exit_date(store: SQLiteRuntimeStore, target_date: str) -> int:
    rows = store.query(
        """
        SELECT trade_id
        FROM trades
        WHERE (
            trade_id LIKE 'reconstructed:%'
            OR COALESCE(raw_json, '') LIKE '%executions_pair_repair%'
            OR COALESCE(raw_json, '') LIKE '%sqlite_execution_reducer%'
        )
          AND (
            substr(exit_fill_time, 1, 10) = ?
            OR substr(closed_at, 1, 10) = ?
            OR COALESCE(raw_json, '') LIKE ?
          )
        """,
        [target_date, target_date, f'%"exit_date": "{target_date}"%'],
    )
    trade_ids = [str(row.get("trade_id") or "") for row in rows if row.get("trade_id")]
    if not trade_ids:
        return 0
    placeholders = ",".join("?" for _ in trade_ids)
    store.execute(f"UPDATE executions SET trade_id = NULL WHERE trade_id IN ({placeholders})", trade_ids)
    store.execute(f"DELETE FROM trades WHERE trade_id IN ({placeholders})", trade_ids)
    return len(trade_ids)


def load_execution_rows(store: SQLiteRuntimeStore, target_date: str, lookback_days: int) -> list[dict[str, Any]]:
    start_date = (datetime.fromisoformat(target_date).date() - timedelta(days=max(0, lookback_days))).isoformat()
    rows = store.query(
        """
        SELECT execution_id, trade_id, COALESCE(strategy_name, 'unknown') AS strategy_name,
               session_date, symbol, side, quantity, price, executed_at, recorded_at,
               commission, commission_source, commission_currency, raw_json
        FROM executions
        WHERE COALESCE(session_date, substr(executed_at, 1, 10), substr(recorded_at, 1, 10)) BETWEEN ? AND ?
        ORDER BY COALESCE(executed_at, recorded_at), execution_id
        """,
        [start_date, target_date],
    )
    return [dict(row) for row in rows]


def reconstruct_closed_trades_from_executions(
    store: SQLiteRuntimeStore,
    target_date: str,
    *,
    lookback_days: int = 45,
    apply: bool = False,
) -> dict[str, Any]:
    rows = load_execution_rows(store, target_date, lookback_days)
    existing_ids, existing_fingerprints = existing_trade_fingerprints(store, include_reconstructed=False)
    open_lots: dict[tuple[str, str], list[dict[str, Any]]] = {}
    planned: list[dict[str, Any]] = []
    skipped_sell_only = 0
    skipped_existing = 0
    deleted_reconstructed = 0

    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        strategy = str(row.get("strategy_name") or "unknown")
        side = normalized_side(row.get("side"))
        qty = as_float(row.get("quantity"))
        price = as_float(row.get("price"))
        if not symbol or qty <= 0 or price <= 0:
            continue
        key = (strategy, symbol)
        row_time = execution_time(row)
        if side == "BUY":
            lot = dict(row)
            lot["execution_time"] = row_time
            lot["remaining_qty"] = qty
            lot["original_qty"] = qty
            open_lots.setdefault(key, []).append(lot)
            continue
        if side != "SELL":
            continue

        remaining = qty
        if not open_lots.get(key):
            if date_part(row_time or row.get("session_date")) == target_date:
                skipped_sell_only += 1
            continue
        while remaining > 1e-9 and open_lots.get(key):
            lot = open_lots[key][0]
            lot_remaining = as_float(lot.get("remaining_qty"))
            matched_qty = min(remaining, lot_remaining)
            buy_time = str(lot.get("execution_time") or execution_time(lot))
            sell_time = row_time
            exit_date = date_part(sell_time or row.get("session_date"))
            if exit_date != target_date:
                lot["remaining_qty"] = lot_remaining - matched_qty
                remaining -= matched_qty
                if as_float(lot.get("remaining_qty")) <= 1e-9:
                    open_lots[key].pop(0)
                continue
            entry_date = date_part(buy_time or lot.get("session_date"))
            if not entry_date:
                skipped_sell_only += 1
                break
            buy_exec = str(lot.get("execution_id") or "")
            sell_exec = str(row.get("execution_id") or "")
            trade_id = trade_id_for(entry_date, exit_date, symbol, buy_exec, sell_exec)
            buy_fraction = matched_qty / as_float(lot.get("original_qty")) if as_float(lot.get("original_qty")) else 0.0
            sell_fraction = matched_qty / qty if qty else 0.0
            buy_commission, buy_commission_ok = commission_value(lot, buy_fraction)
            sell_commission, sell_commission_ok = commission_value(row, sell_fraction)
            commission = buy_commission + sell_commission
            gross = (price - as_float(lot.get("price"))) * matched_qty
            fingerprint = (
                symbol,
                buy_time,
                sell_time,
                f"{matched_qty:.8f}",
                f"{as_float(lot.get('price')):.8f}",
                f"{price:.8f}",
            )
            if trade_id in existing_ids or fingerprint in existing_fingerprints:
                skipped_existing += 1
            else:
                commission_status = "OK" if buy_commission_ok and sell_commission_ok else ("PARTIAL" if buy_commission_ok or sell_commission_ok else "MISSING")
                planned.append(
                    {
                        "trade_id": trade_id,
                        "strategy_name": strategy,
                        "session_date": entry_date,
                        "symbol": symbol,
                        "status": "CLOSED",
                        "entry_fill_time": buy_time,
                        "exit_fill_time": sell_time,
                        "closed_at": sell_time,
                        "entry_price": as_float(lot.get("price")),
                        "exit_price": price,
                        "quantity": matched_qty,
                        "gross_pnl": gross,
                        "commission": commission,
                        "net_pnl": gross - commission,
                        "ibkr_entry_confirmed": True,
                        "ibkr_exit_confirmed": True,
                        "ibkr_position_flat_confirmed": True,
                        "ibkr_position_flat_confirmed_at": sell_time,
                        "raw_json": {
                            "reconstruction_source": "executions_pair_repair",
                            "buy_execution_id": buy_exec,
                            "sell_execution_id": sell_exec,
                            "matched_quantity": matched_qty,
                            "buy_original_quantity": as_float(lot.get("original_qty")),
                            "sell_original_quantity": qty,
                            "entry_executed_at": buy_time,
                            "exit_executed_at": sell_time,
                            "exit_date": exit_date,
                            "commission_status": commission_status,
                            "buy_commission_confirmed": buy_commission_ok,
                            "sell_commission_confirmed": sell_commission_ok,
                        },
                    }
                )
            lot["remaining_qty"] = lot_remaining - matched_qty
            remaining -= matched_qty
            if as_float(lot.get("remaining_qty")) <= 1e-9:
                open_lots[key].pop(0)

    if apply:
        with store.transaction():
            deleted_reconstructed = delete_reconstructed_for_exit_date(store, target_date)
            for trade in planned:
                store.upsert_trade(trade)

    return {
        "date": target_date,
        "apply": bool(apply),
        "planned": len(planned),
        "created": len(planned) if apply else 0,
        "deleted_reconstructed": deleted_reconstructed,
        "skipped_existing": skipped_existing,
        "skipped_sell_only": skipped_sell_only,
        "trades": planned,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill persisted CLOSED trades from SQLite executions.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--date", required=True)
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--show-trades", action="store_true")
    args = parser.parse_args()

    store = SQLiteRuntimeStore(resolve_sqlite_path(args.sqlite_path))
    try:
        result = reconstruct_closed_trades_from_executions(
            store,
            args.date,
            lookback_days=args.lookback_days,
            apply=bool(args.apply),
        )
        printable = dict(result)
        if not args.show_trades:
            printable.pop("trades", None)
        print(json.dumps(printable, indent=2, ensure_ascii=False, default=str))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
