#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, deque
from pathlib import Path
from typing import Any


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def jget(raw: Any, key: str) -> Any:
    try:
        data = json.loads(raw or "{}") if isinstance(raw, str) else (raw or {})
    except Exception:
        data = {}
    return data.get(key) if isinstance(data, dict) else None


def execution_time(row: dict[str, Any]) -> str:
    return str(row.get("executed_at") or row.get("recorded_at") or "")


def side(row: dict[str, Any]) -> str:
    value = str(row.get("side") or "").upper()
    if value in {"BOT", "BUY"}:
        return "BUY"
    if value in {"SLD", "SELL"}:
        return "SELL"
    return value


def print_table(title: str, data: list[dict[str, Any]], cols: list[str]) -> None:
    print(f"\n=== {title} ({len(data)}) ===")
    if not data:
        return
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in data)) for col in cols}
    print(" | ".join(col.ljust(widths[col]) for col in cols))
    print("-+-".join("-" * widths[col] for col in cols))
    for row in data:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols))


def proposed_same_session_pairs(executions: list[dict[str, Any]], selected_date: str) -> list[dict[str, Any]]:
    buys = deque()
    pairs: list[dict[str, Any]] = []
    for row in sorted(executions, key=lambda item: (execution_time(item), str(item.get("execution_id") or ""))):
        row_side = side(row)
        qty = abs(float(row.get("quantity") or 0.0))
        if qty <= 0:
            continue
        if row_side == "BUY":
            if str(row.get("session_date") or execution_time(row)[:10]) == selected_date:
                item = dict(row)
                item["remaining_qty"] = qty
                buys.append(item)
            continue
        if row_side != "SELL":
            continue
        if str(row.get("session_date") or execution_time(row)[:10]) != selected_date:
            continue
        remaining = qty
        while remaining > 1e-9 and buys:
            buy = buys[0]
            buy_remaining = float(buy.get("remaining_qty") or 0.0)
            matched = min(remaining, buy_remaining)
            pairs.append(
                {
                    "buy_execution_id": buy.get("execution_id"),
                    "sell_execution_id": row.get("execution_id"),
                    "matched_qty": matched,
                    "buy_time": execution_time(buy),
                    "sell_time": execution_time(row),
                    "buy_price": buy.get("price"),
                    "sell_price": row.get("price"),
                    "buy_order_id": buy.get("order_id"),
                    "sell_order_id": row.get("order_id"),
                }
            )
            buy["remaining_qty"] = buy_remaining - matched
            remaining -= matched
            if float(buy.get("remaining_qty") or 0.0) <= 1e-9:
                buys.popleft()
    return pairs


def diagnose(symbol: str, selected_date: str, sqlite_path: Path) -> int:
    symbol = symbol.upper()
    with connect(sqlite_path) as conn:
        executions = rows(
            conn,
            """
            SELECT execution_id, trade_id, order_id, perm_id, strategy_name, session_date, symbol, side,
                   quantity, price, executed_at, recorded_at, commission, commission_source, realized_pnl,
                   exit_reason, raw_json
            FROM executions
            WHERE upper(symbol) = ?
              AND (
                session_date = ?
                OR substr(COALESCE(executed_at, recorded_at), 1, 10) = ?
              )
            ORDER BY COALESCE(executed_at, recorded_at), execution_id
            """,
            (symbol, selected_date, selected_date),
        )
        historical_executions = rows(
            conn,
            """
            SELECT execution_id, trade_id, order_id, perm_id, strategy_name, session_date, symbol, side,
                   quantity, price, executed_at, recorded_at, commission, commission_source, realized_pnl,
                   exit_reason, raw_json
            FROM executions
            WHERE upper(symbol) = ?
            ORDER BY COALESCE(executed_at, recorded_at), execution_id
            """,
            (symbol,),
        )
        trades = rows(
            conn,
            """
            SELECT trade_id, strategy_name, session_date, symbol, status, entry_fill_time, exit_fill_time,
                   closed_at, entry_price, exit_price, quantity, gross_pnl, commission, net_pnl, raw_json
            FROM trades
            WHERE upper(symbol) = ?
              AND (
                substr(COALESCE(exit_fill_time, closed_at), 1, 10) = ?
                OR session_date = ?
              )
            ORDER BY COALESCE(exit_fill_time, closed_at, entry_fill_time), trade_id
            """,
            (symbol, selected_date, selected_date),
        )

    buy_execs = [row for row in executions if side(row) == "BUY"]
    sell_execs = [row for row in executions if side(row) == "SELL"]
    reconstructed = [
        {
            **row,
            "buy_execution_id": jget(row.get("raw_json"), "buy_execution_id"),
            "sell_execution_id": jget(row.get("raw_json"), "sell_execution_id"),
            "reconstruction_source": jget(row.get("raw_json"), "reconstruction_source"),
        }
        for row in trades
        if str(row.get("trade_id") or "").startswith("reconstructed:")
        or str(jget(row.get("raw_json"), "reconstruction_source") or "").strip()
    ]
    sell_counts = Counter(str(row.get("sell_execution_id") or "") for row in reconstructed if row.get("sell_execution_id"))
    buy_counts = Counter(str(row.get("buy_execution_id") or "") for row in reconstructed if row.get("buy_execution_id"))
    reused_sell = {key: count for key, count in sell_counts.items() if count > 1}
    reused_buy = {key: count for key, count in buy_counts.items() if count > 1}
    cross_session = [
        row
        for row in reconstructed
        if str(row.get("entry_fill_time") or "")[:10]
        and str(row.get("exit_fill_time") or row.get("closed_at") or "")[:10]
        and str(row.get("entry_fill_time") or "")[:10] < str(row.get("exit_fill_time") or row.get("closed_at") or "")[:10]
    ]
    proposed_pairs = proposed_same_session_pairs(executions, selected_date)

    print(f"symbol={symbol} date={selected_date} sqlite_path={sqlite_path}")
    print(
        "summary "
        f"buy_execs={len(buy_execs)} sell_execs={len(sell_execs)} "
        f"reconstructed_trades={len(reconstructed)} "
        f"reused_sell_execution_id_count={len(reused_sell)} "
        f"reused_buy_execution_id_count={len(reused_buy)} "
        f"cross_session_fifo_match_count={len(cross_session)} "
        f"duplicate_reconstructed_sell_rows={sum(count - 1 for count in reused_sell.values())}"
    )

    print_table(
        "BUY executions selected date",
        buy_execs,
        ["execution_id", "trade_id", "order_id", "quantity", "price", "executed_at", "commission", "commission_source"],
    )
    print_table(
        "SELL executions selected date",
        sell_execs,
        ["execution_id", "trade_id", "order_id", "quantity", "price", "executed_at", "commission", "commission_source", "realized_pnl"],
    )
    print_table(
        "Reconstructed trades for selected date",
        reconstructed,
        [
            "trade_id",
            "status",
            "quantity",
            "entry_fill_time",
            "exit_fill_time",
            "buy_execution_id",
            "sell_execution_id",
            "entry_price",
            "exit_price",
            "net_pnl",
        ],
    )
    print_table(
        "Proposed same-session FIFO pairs",
        proposed_pairs,
        ["buy_execution_id", "sell_execution_id", "matched_qty", "buy_time", "sell_time", "buy_price", "sell_price", "buy_order_id", "sell_order_id"],
    )
    if reused_sell:
        print("\nreused_sell_execution_ids=" + json.dumps(reused_sell, sort_keys=True))
    if reused_buy:
        print("reused_buy_execution_ids=" + json.dumps(reused_buy, sort_keys=True))
    if cross_session:
        print("cross_session_trade_ids=" + ",".join(str(row.get("trade_id") or "") for row in cross_session))

    historical_buys = [row for row in historical_executions if side(row) == "BUY" and str(row.get("session_date") or execution_time(row)[:10]) < selected_date]
    if historical_buys:
        print_table(
            "Historical BUY executions before selected date",
            historical_buys[-20:],
            ["execution_id", "trade_id", "order_id", "session_date", "quantity", "price", "executed_at", "commission_source"],
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose reconstructed FIFO rows for a symbol/date without mutating SQLite.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--sqlite-path", default=str(DEFAULT_SQLITE_PATH))
    args = parser.parse_args()
    return diagnose(args.symbol, args.date, Path(args.sqlite_path))


if __name__ == "__main__":
    raise SystemExit(main())
