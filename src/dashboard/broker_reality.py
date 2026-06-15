from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.dashboard.runtime_queries import parse_raw_json, to_float
from src.live_trading.storage.sqlite_store import resolve_sqlite_path


BROKER_EXECUTION_COLUMNS = [
    "execution_time",
    "symbol",
    "side",
    "quantity",
    "price",
    "commission",
    "execution_id",
    "order_id",
    "perm_id",
    "account",
    "exchange",
    "currency",
    "source",
]

BROKER_CLOSED_TRADE_COLUMNS = [
    "symbol",
    "entry_time",
    "exit_time",
    "quantity",
    "entry_price",
    "exit_price",
    "realized_pnl",
    "commission",
    "net_pnl",
    "source",
    "entry_execution_id",
    "exit_execution_id",
]


@dataclass(frozen=True)
class ReconciliationResult:
    summary: dict[str, Any]
    matched: pd.DataFrame
    missing_in_sqlite: pd.DataFrame
    extra_in_sqlite: pd.DataFrame
    execution_mismatches: pd.DataFrame
    position_mismatches: pd.DataFrame
    broker_closed_trades: pd.DataFrame
    sqlite_closed_trades: pd.DataFrame
    matched_trades: pd.DataFrame
    missing_trades: pd.DataFrame
    extra_trades: pd.DataFrame
    trade_mismatches: pd.DataFrame
    pnl_comparison: pd.DataFrame


def parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def iso_time(value: Any) -> str:
    dt = parse_dt(value)
    if dt is None:
        return str(value or "")
    return dt.isoformat()


def date_part(value: Any) -> str:
    dt = parse_dt(value)
    if dt is not None:
        return dt.strftime("%F")
    text = str(value or "")
    return text[:10] if len(text) >= 10 else ""


def normalize_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"BOT", "BUY", "BOUGHT"}:
        return "BUY"
    if text in {"SLD", "SELL", "SOLD"}:
        return "SELL"
    if text.startswith("B"):
        return "BUY"
    if text.startswith("S"):
        return "SELL"
    return text


def norm_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def pick(row: dict[str, Any], *names: str) -> Any:
    normalized = {norm_key(k): v for k, v in row.items()}
    for name in names:
        value = normalized.get(norm_key(name))
        if value not in (None, ""):
            return value
    return None


def empty_broker_executions() -> pd.DataFrame:
    return pd.DataFrame(columns=BROKER_EXECUTION_COLUMNS)


def empty_closed_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=BROKER_CLOSED_TRADE_COLUMNS)


def ensure_asyncio_event_loop() -> str:
    thread_name = threading.current_thread().name
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return f"created_replacement_loop thread={thread_name}"
        return f"existing_loop thread={thread_name} running={loop.is_running()}"
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return f"created_loop thread={thread_name}"


def describe_asyncio_event_loop() -> str:
    thread_name = threading.current_thread().name
    try:
        loop = asyncio.get_event_loop()
        return f"thread={thread_name} loop_present=true running={loop.is_running()} closed={loop.is_closed()}"
    except RuntimeError as exc:
        return f"thread={thread_name} loop_present=false error={exc}"


def normalize_execution_record(row: dict[str, Any], *, source: str = "csv") -> dict[str, Any]:
    symbol = pick(row, "symbol", "underlying", "contractsymbol", "description")
    if symbol:
        symbol = str(symbol).strip().split()[0].upper()
    side = normalize_side(pick(row, "side", "buysell", "action", "transactiontype"))
    quantity = abs(to_float(pick(row, "quantity", "qty", "shares"), 0.0) or 0.0)
    price = to_float(pick(row, "price", "tprice", "tradeprice", "fillprice", "transactionprice"), None)
    commission = to_float(pick(row, "commission", "commfee", "commissionfee", "ibkrcomm", "commissionamount"), None)
    execution_time = pick(row, "executiontime", "datetime", "date/time", "time", "tradedatetime", "date")
    return {
        "execution_time": iso_time(execution_time),
        "symbol": symbol or "",
        "side": side,
        "quantity": quantity,
        "price": price,
        "commission": abs(float(commission)) if commission is not None else None,
        "execution_id": str(pick(row, "executionid", "execid", "execution", "ibexecid") or "").strip(),
        "order_id": str(pick(row, "orderid", "order", "iborderid") or "").strip(),
        "perm_id": str(pick(row, "permid", "perm id") or "").strip(),
        "account": str(pick(row, "account", "accountid", "acctid") or "").strip(),
        "exchange": str(pick(row, "exchange", "listingexchange") or "").strip(),
        "currency": str(pick(row, "currency", "currencyprimary", "fxcurrency") or "").strip(),
        "source": source,
    }


def parse_ibkr_activity_csv(content: str | bytes | io.BytesIO | io.StringIO) -> pd.DataFrame:
    if hasattr(content, "read"):
        raw = content.read()
    else:
        raw = content
    if isinstance(raw, bytes):
        raw_text = raw.decode("utf-8-sig", errors="replace")
    else:
        raw_text = str(raw or "")
    if not raw_text.strip():
        return empty_broker_executions()

    lines = [line for line in raw_text.splitlines() if line.strip()]
    flex_records: list[dict[str, Any]] = []
    flex_header: list[str] | None = None
    for line in lines:
        parts = next(pd.read_csv(io.StringIO(line), header=None, dtype=str, keep_default_na=False).itertuples(index=False, name=None))
        parts = [str(x) for x in parts]
        if len(parts) < 2 or parts[0].strip().lower() != "trades":
            continue
        row_type = parts[1].strip().lower()
        if row_type == "header":
            flex_header = parts[2:]
            continue
        if row_type == "data" and flex_header:
            values = parts[2:]
            flex_records.append(dict(zip(flex_header, values)))
    if flex_records:
        rows = [normalize_execution_record(row, source="ibkr_activity_csv") for row in flex_records]
        rows = [row for row in rows if row["symbol"] and row["side"] in {"BUY", "SELL"}]
        return pd.DataFrame(rows, columns=BROKER_EXECUTION_COLUMNS)

    try:
        df = pd.read_csv(io.StringIO(raw_text), dtype=str, keep_default_na=False)
    except Exception:
        return empty_broker_executions()
    rows = [normalize_execution_record(row, source="ibkr_activity_csv") for row in df.to_dict("records")]
    rows = [row for row in rows if row["symbol"] and row["side"] in {"BUY", "SELL"}]
    return pd.DataFrame(rows, columns=BROKER_EXECUTION_COLUMNS)


def load_sqlite_executions(sqlite_path: str | Path, selected_date: str) -> pd.DataFrame:
    path = Path(resolve_sqlite_path(sqlite_path))
    if not path.exists():
        return empty_broker_executions()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = pd.read_sql_query(
            """
            SELECT execution_id, symbol, side, quantity, price, commission, order_id, perm_id,
                   exchange, executed_at, recorded_at, commission_currency, realized_pnl,
                   commission_source, raw_json
            FROM executions
            WHERE COALESCE(substr(executed_at, 1, 10), substr(recorded_at, 1, 10), session_date) = ?
            ORDER BY COALESCE(executed_at, recorded_at), execution_id
            """,
            conn,
            params=[selected_date],
        )
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        raw = parse_raw_json(row.get("raw_json"))
        out.append(
            {
                "execution_time": iso_time(row.get("executed_at") or row.get("recorded_at")),
                "symbol": str(row.get("symbol") or "").upper(),
                "side": normalize_side(row.get("side")),
                "quantity": abs(to_float(row.get("quantity"), 0.0) or 0.0),
                "price": to_float(row.get("price"), None),
                "commission": abs(to_float(row.get("commission"), 0.0) or 0.0) if str(row.get("commission_source") or "").lower() == "ibkr" else None,
                "execution_id": str(row.get("execution_id") or ""),
                "order_id": str(row.get("order_id") or ""),
                "perm_id": str(row.get("perm_id") or ""),
                "account": str(raw.get("account") or raw.get("acctNumber") or ""),
                "exchange": str(row.get("exchange") or ""),
                "currency": str(row.get("commission_currency") or raw.get("currency") or ""),
                "source": "sqlite",
            }
        )
    return pd.DataFrame(out, columns=BROKER_EXECUTION_COLUMNS)


def load_sqlite_closed_trades(sqlite_path: str | Path, selected_date: str) -> pd.DataFrame:
    path = Path(resolve_sqlite_path(sqlite_path))
    if not path.exists():
        return empty_closed_trades()
    conn = sqlite3.connect(str(path))
    try:
        rows = pd.read_sql_query(
            """
            SELECT trade_id, symbol, entry_fill_time, exit_fill_time, closed_at,
                   quantity, entry_price, exit_price, gross_pnl, commission, net_pnl,
                   raw_json
            FROM trades
            WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
              AND COALESCE(substr(exit_fill_time, 1, 10), substr(closed_at, 1, 10), session_date) = ?
            ORDER BY COALESCE(exit_fill_time, closed_at), symbol, trade_id
            """,
            conn,
            params=[selected_date],
        )
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        out.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "entry_time": iso_time(row.get("entry_fill_time")),
                "exit_time": iso_time(row.get("exit_fill_time") or row.get("closed_at")),
                "quantity": abs(to_float(row.get("quantity"), 0.0) or 0.0),
                "entry_price": to_float(row.get("entry_price"), None),
                "exit_price": to_float(row.get("exit_price"), None),
                "realized_pnl": to_float(row.get("gross_pnl"), 0.0) or 0.0,
                "commission": abs(to_float(row.get("commission"), 0.0) or 0.0),
                "net_pnl": to_float(row.get("net_pnl"), 0.0) or 0.0,
                "source": "sqlite_trades",
                "entry_execution_id": "",
                "exit_execution_id": str(row.get("trade_id") or ""),
            }
        )
    return pd.DataFrame(out, columns=BROKER_CLOSED_TRADE_COLUMNS)


def fetch_ibkr_live_portfolio(host: str, port: int, client_id: int, timeout: float = 4.0) -> tuple[pd.DataFrame, str]:
    loop_info = ensure_asyncio_event_loop()
    try:
        from ib_insync import IB  # type: ignore
    except Exception as exc:
        return pd.DataFrame(), f"ib_insync_unavailable: {exc}; {loop_info}"
    ib = IB()
    try:
        ib.connect(host, int(port), clientId=int(client_id), timeout=timeout)
        rows: list[dict[str, Any]] = []
        refreshed_at = datetime.now(timezone.utc).isoformat()
        for item in ib.portfolio():
            contract = getattr(item, "contract", None)
            symbol = getattr(contract, "symbol", "") or getattr(contract, "localSymbol", "")
            rows.append(
                {
                    "symbol": str(symbol).upper(),
                    "quantity": to_float(getattr(item, "position", None), 0.0),
                    "average_cost": to_float(getattr(item, "averageCost", None), None),
                    "market_price": to_float(getattr(item, "marketPrice", None), None),
                    "market_value": to_float(getattr(item, "marketValue", None), None),
                    "unrealized_pnl": to_float(getattr(item, "unrealizedPNL", None), None),
                    "account": getattr(item, "account", ""),
                    "last_refresh_time": refreshed_at,
                }
            )
        return pd.DataFrame(rows), f"OK; {loop_info}"
    except Exception as exc:
        return pd.DataFrame(), f"ibkr_portfolio_error: {exc}; {loop_info}"
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def fetch_ibkr_executions_for_date(host: str, port: int, client_id: int, selected_date: str, timeout: float = 4.0) -> tuple[pd.DataFrame, str]:
    loop_info = ensure_asyncio_event_loop()
    try:
        from ib_insync import ExecutionFilter, IB  # type: ignore
    except Exception as exc:
        return empty_broker_executions(), f"ib_insync_unavailable: {exc}; {loop_info}"
    ib = IB()
    try:
        ib.connect(host, int(port), clientId=int(client_id), timeout=timeout)
        execution_filter = ExecutionFilter()
        execution_filter.time = f"{selected_date.replace('-', '')} 00:00:00"
        fills = ib.reqExecutions(execution_filter)
        rows: list[dict[str, Any]] = []
        for fill in fills:
            contract = getattr(fill, "contract", None)
            execution = getattr(fill, "execution", None)
            commission_report = getattr(fill, "commissionReport", None)
            rows.append(
                normalize_execution_record(
                    {
                        "execution_time": getattr(execution, "time", None),
                        "symbol": getattr(contract, "symbol", None),
                        "side": getattr(execution, "side", None),
                        "quantity": getattr(execution, "shares", None),
                        "price": getattr(execution, "price", None),
                        "commission": getattr(commission_report, "commission", None),
                        "execution_id": getattr(execution, "execId", None),
                        "order_id": getattr(execution, "orderId", None),
                        "perm_id": getattr(execution, "permId", None),
                        "account": getattr(execution, "acctNumber", None),
                        "exchange": getattr(execution, "exchange", None),
                        "currency": getattr(commission_report, "currency", None),
                    },
                    source="ibkr_api",
                )
            )
        df = pd.DataFrame(rows, columns=BROKER_EXECUTION_COLUMNS)
        if not df.empty:
            df = df[df["execution_time"].map(date_part) == selected_date].reset_index(drop=True)
        return df, f"OK; {loop_info}"
    except Exception as exc:
        return empty_broker_executions(), f"ibkr_executions_error: {exc}; {loop_info}"
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def execution_signature(row: dict[str, Any]) -> tuple[str, str, float, float]:
    return (
        str(row.get("symbol") or "").upper(),
        normalize_side(row.get("side")),
        round(float(to_float(row.get("quantity"), 0.0) or 0.0), 6),
        round(float(to_float(row.get("price"), 0.0) or 0.0), 4),
    )


def times_close(left: Any, right: Any, tolerance_seconds: int) -> bool:
    left_dt = parse_dt(left)
    right_dt = parse_dt(right)
    if left_dt is None or right_dt is None:
        return True
    return abs((left_dt - right_dt).total_seconds()) <= tolerance_seconds


def match_executions(
    broker: pd.DataFrame,
    sqlite_execs: pd.DataFrame,
    *,
    time_tolerance_seconds: int = 5,
    price_tolerance: float = 0.01,
    quantity_tolerance: float = 1e-6,
    commission_tolerance: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    broker_rows = broker.to_dict("records") if not broker.empty else []
    sqlite_rows = sqlite_execs.to_dict("records") if not sqlite_execs.empty else []
    used_sqlite: set[int] = set()
    matched: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    sqlite_by_exec_id = {
        str(row.get("execution_id")): idx
        for idx, row in enumerate(sqlite_rows)
        if str(row.get("execution_id") or "")
    }

    for broker_row in broker_rows:
        match_idx: int | None = None
        exec_id = str(broker_row.get("execution_id") or "")
        if exec_id and exec_id in sqlite_by_exec_id and sqlite_by_exec_id[exec_id] not in used_sqlite:
            match_idx = sqlite_by_exec_id[exec_id]
            matched_by = "execution_id"
        else:
            matched_by = "symbol_side_qty_price_time"
            broker_sig = execution_signature(broker_row)
            for idx, sqlite_row in enumerate(sqlite_rows):
                if idx in used_sqlite:
                    continue
                if execution_signature(sqlite_row) != broker_sig:
                    continue
                if times_close(broker_row.get("execution_time"), sqlite_row.get("execution_time"), time_tolerance_seconds):
                    match_idx = idx
                    break
        if match_idx is None:
            missing.append({**broker_row, "mismatch_type": "MISSING_IN_SQLITE"})
            continue
        used_sqlite.add(match_idx)
        sqlite_row = sqlite_rows[match_idx]
        flags: list[str] = []
        qty_delta = (to_float(broker_row.get("quantity"), 0.0) or 0.0) - (to_float(sqlite_row.get("quantity"), 0.0) or 0.0)
        price_delta = (to_float(broker_row.get("price"), 0.0) or 0.0) - (to_float(sqlite_row.get("price"), 0.0) or 0.0)
        broker_comm = to_float(broker_row.get("commission"), None)
        sqlite_comm = to_float(sqlite_row.get("commission"), None)
        comm_delta = None
        if broker_comm is not None and sqlite_comm is not None:
            comm_delta = broker_comm - sqlite_comm
        if abs(qty_delta) > quantity_tolerance:
            flags.append("QUANTITY_MISMATCH")
        if abs(price_delta) > price_tolerance:
            flags.append("PRICE_MISMATCH")
        if comm_delta is not None and abs(comm_delta) > commission_tolerance:
            flags.append("COMMISSION_MISMATCH")
        row = {
            "symbol": broker_row.get("symbol"),
            "side": broker_row.get("side"),
            "execution_id": broker_row.get("execution_id") or sqlite_row.get("execution_id"),
            "matched_by": matched_by,
            "broker_quantity": broker_row.get("quantity"),
            "sqlite_quantity": sqlite_row.get("quantity"),
            "broker_price": broker_row.get("price"),
            "sqlite_price": sqlite_row.get("price"),
            "broker_commission": broker_comm,
            "sqlite_commission": sqlite_comm,
            "quantity_delta": qty_delta,
            "price_delta": price_delta,
            "commission_delta": comm_delta,
            "status": "MATCHED" if not flags else ";".join(flags),
        }
        matched.append(row)
        if flags:
            mismatches.append(row)

    extra = [{**row, "mismatch_type": "EXTRA_IN_SQLITE"} for idx, row in enumerate(sqlite_rows) if idx not in used_sqlite]
    return pd.DataFrame(matched), pd.DataFrame(missing), pd.DataFrame(extra), pd.DataFrame(mismatches)


def reconstruct_closed_trades_fifo(executions: pd.DataFrame, selected_date: str) -> pd.DataFrame:
    if executions.empty:
        return empty_closed_trades()
    rows = executions.copy()
    rows["_dt"] = pd.to_datetime(rows["execution_time"], errors="coerce", utc=True)
    rows = rows.sort_values(["symbol", "_dt", "execution_id"], na_position="last")
    lots_by_symbol: dict[str, list[dict[str, Any]]] = {}
    closed: list[dict[str, Any]] = []

    for row in rows.to_dict("records"):
        symbol = str(row.get("symbol") or "").upper()
        side = normalize_side(row.get("side"))
        qty = abs(to_float(row.get("quantity"), 0.0) or 0.0)
        price = to_float(row.get("price"), None)
        if not symbol or qty <= 0 or price is None:
            continue
        commission = abs(to_float(row.get("commission"), 0.0) or 0.0)
        if side == "BUY":
            lots_by_symbol.setdefault(symbol, []).append(
                {
                    "remaining_qty": qty,
                    "original_qty": qty,
                    "price": float(price),
                    "commission": commission,
                    "execution_time": row.get("execution_time"),
                    "execution_id": row.get("execution_id"),
                }
            )
            continue
        if side != "SELL":
            continue
        remaining_sell = qty
        sell_original_qty = qty
        lots = lots_by_symbol.setdefault(symbol, [])
        while remaining_sell > 1e-9 and lots:
            lot = lots[0]
            match_qty = min(float(lot["remaining_qty"]), remaining_sell)
            buy_comm = (float(lot.get("commission") or 0.0) * match_qty / float(lot.get("original_qty") or match_qty)) if lot.get("original_qty") else 0.0
            sell_comm = (commission * match_qty / sell_original_qty) if sell_original_qty else 0.0
            gross = (float(price) - float(lot["price"])) * match_qty
            total_comm = buy_comm + sell_comm
            exit_date = date_part(row.get("execution_time"))
            if exit_date == selected_date:
                closed.append(
                    {
                        "symbol": symbol,
                        "entry_time": iso_time(lot.get("execution_time")),
                        "exit_time": iso_time(row.get("execution_time")),
                        "quantity": match_qty,
                        "entry_price": float(lot["price"]),
                        "exit_price": float(price),
                        "realized_pnl": gross,
                        "commission": total_comm,
                        "net_pnl": gross - total_comm,
                        "source": "BROKER_FIFO_RECONSTRUCTED",
                        "entry_execution_id": str(lot.get("execution_id") or ""),
                        "exit_execution_id": str(row.get("execution_id") or ""),
                    }
                )
            lot["remaining_qty"] = float(lot["remaining_qty"]) - match_qty
            remaining_sell -= match_qty
            if float(lot["remaining_qty"]) <= 1e-9:
                lots.pop(0)
    return pd.DataFrame(closed, columns=BROKER_CLOSED_TRADE_COLUMNS)


def trade_signature(row: dict[str, Any]) -> tuple[str, float, float, float]:
    return (
        str(row.get("symbol") or "").upper(),
        round(float(to_float(row.get("quantity"), 0.0) or 0.0), 6),
        round(float(to_float(row.get("entry_price"), 0.0) or 0.0), 4),
        round(float(to_float(row.get("exit_price"), 0.0) or 0.0), 4),
    )


def compare_closed_trades(
    broker_trades: pd.DataFrame,
    sqlite_trades: pd.DataFrame,
    *,
    pnl_tolerance: float = 0.02,
    commission_tolerance: float = 0.02,
    quantity_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    broker_rows = broker_trades.to_dict("records") if not broker_trades.empty else []
    sqlite_rows = sqlite_trades.to_dict("records") if not sqlite_trades.empty else []
    used_sqlite: set[int] = set()
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for broker_row in broker_rows:
        match_idx: int | None = None
        broker_sig = trade_signature(broker_row)
        for idx, sqlite_row in enumerate(sqlite_rows):
            if idx in used_sqlite:
                continue
            if trade_signature(sqlite_row) == broker_sig:
                match_idx = idx
                break
        if match_idx is None:
            missing.append({**broker_row, "mismatch_type": "MISSING_TRADE_IN_SQLITE"})
            continue
        used_sqlite.add(match_idx)
        sqlite_row = sqlite_rows[match_idx]
        qty_delta = (to_float(broker_row.get("quantity"), 0.0) or 0.0) - (to_float(sqlite_row.get("quantity"), 0.0) or 0.0)
        pnl_delta = (to_float(broker_row.get("realized_pnl"), 0.0) or 0.0) - (to_float(sqlite_row.get("realized_pnl"), 0.0) or 0.0)
        commission_delta = (to_float(broker_row.get("commission"), 0.0) or 0.0) - (to_float(sqlite_row.get("commission"), 0.0) or 0.0)
        flags: list[str] = []
        if abs(qty_delta) > quantity_tolerance:
            flags.append("QUANTITY_MISMATCH")
        if abs(pnl_delta) > pnl_tolerance:
            flags.append("PNL_MISMATCH")
        if abs(commission_delta) > commission_tolerance:
            flags.append("COMMISSION_MISMATCH")
        row = {
            "symbol": broker_row.get("symbol"),
            "broker_quantity": broker_row.get("quantity"),
            "sqlite_quantity": sqlite_row.get("quantity"),
            "broker_realized_pnl": broker_row.get("realized_pnl"),
            "sqlite_realized_pnl": sqlite_row.get("realized_pnl"),
            "broker_commission": broker_row.get("commission"),
            "sqlite_commission": sqlite_row.get("commission"),
            "broker_net_pnl": broker_row.get("net_pnl"),
            "sqlite_net_pnl": sqlite_row.get("net_pnl"),
            "quantity_delta": qty_delta,
            "pnl_delta": pnl_delta,
            "commission_delta": commission_delta,
            "status": "MATCHED" if not flags else ";".join(flags),
        }
        matched.append(row)
        if flags:
            mismatches.append(row)
    extra = [{**row, "mismatch_type": "EXTRA_TRADE_IN_SQLITE"} for idx, row in enumerate(sqlite_rows) if idx not in used_sqlite]
    return pd.DataFrame(matched), pd.DataFrame(missing), pd.DataFrame(extra), pd.DataFrame(mismatches)


def load_sqlite_active_positions(sqlite_path: str | Path, selected_date: str) -> pd.DataFrame:
    path = Path(resolve_sqlite_path(sqlite_path))
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = pd.read_sql_query(
            """
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                         PARTITION BY UPPER(symbol)
                         ORDER BY COALESCE(updated_at, '') DESC, COALESCE(session_date, '') DESC, rowid DESC
                       ) AS rn
                FROM positions
                WHERE COALESCE(session_date, '') <= ?
            )
            SELECT symbol, quantity, avg_price, ibkr_quantity, ibkr_avg_cost, status, active, source, updated_at, raw_json
            FROM ranked
            WHERE rn = 1
              AND COALESCE(active, 0) = 1
              AND UPPER(COALESCE(status, '')) IN ('OPEN', 'EXIT_ORDER')
            """,
            conn,
            params=[selected_date],
        )
    finally:
        conn.close()
    return rows


def compare_positions(ibkr_portfolio: pd.DataFrame, sqlite_positions: pd.DataFrame) -> pd.DataFrame:
    ibkr_map = {str(row.get("symbol") or "").upper(): row for row in ibkr_portfolio.to_dict("records")} if not ibkr_portfolio.empty else {}
    sqlite_map = {str(row.get("symbol") or "").upper(): row for row in sqlite_positions.to_dict("records")} if not sqlite_positions.empty else {}
    rows: list[dict[str, Any]] = []
    for symbol in sorted(set(ibkr_map) | set(sqlite_map)):
        broker_row = ibkr_map.get(symbol)
        sqlite_row = sqlite_map.get(symbol)
        ibkr_qty = to_float(broker_row.get("quantity") if broker_row else None, 0.0) or 0.0
        sqlite_qty = to_float(sqlite_row.get("ibkr_quantity") if sqlite_row else None, None)
        if sqlite_qty is None and sqlite_row:
            sqlite_qty = to_float(sqlite_row.get("quantity"), 0.0) or 0.0
        statuses: list[str] = []
        if sqlite_row and abs(ibkr_qty) <= 1e-9:
            statuses.append("LOCAL_STALE_OPEN")
        elif broker_row and not sqlite_row:
            statuses.append("IBKR_ORPHAN_POSITION")
        elif sqlite_row and broker_row and abs((sqlite_qty or 0.0) - ibkr_qty) > 1e-6:
            statuses.append("QUANTITY_MISMATCH")
        if sqlite_row and broker_row:
            ibkr_avg = to_float(broker_row.get("average_cost"), None)
            sqlite_avg = to_float(sqlite_row.get("avg_price"), None)
            if ibkr_avg is not None and sqlite_avg is not None and abs(ibkr_avg - sqlite_avg) > 0.01:
                statuses.append("AVG_PRICE_MISMATCH")
        status = ";".join(statuses) if statuses else "MATCHED"
        rows.append(
            {
                "symbol": symbol,
                "status": status,
                "ibkr_quantity": ibkr_qty if broker_row else None,
                "sqlite_quantity": sqlite_qty if sqlite_row else None,
                "ibkr_avg_cost": broker_row.get("average_cost") if broker_row else None,
                "sqlite_avg_price": sqlite_row.get("avg_price") if sqlite_row else None,
                "ibkr_market_price": broker_row.get("market_price") if broker_row else None,
                "ibkr_unrealized_pnl": broker_row.get("unrealized_pnl") if broker_row else None,
                "sqlite_status": sqlite_row.get("status") if sqlite_row else None,
                "sqlite_source": sqlite_row.get("source") if sqlite_row else None,
            }
        )
    return pd.DataFrame(rows)


def load_sqlite_trade_pnl(sqlite_path: str | Path, selected_date: str) -> pd.DataFrame:
    path = Path(resolve_sqlite_path(sqlite_path))
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(path))
    try:
        return pd.read_sql_query(
            """
            SELECT
                COUNT(*) AS trades,
                COALESCE(SUM(gross_pnl), 0) AS sqlite_gross,
                COALESCE(SUM(commission), 0) AS sqlite_commission,
                COALESCE(SUM(net_pnl), 0) AS sqlite_net
            FROM trades
            WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
              AND COALESCE(substr(exit_fill_time, 1, 10), substr(closed_at, 1, 10), session_date) = ?
            """,
            conn,
            params=[selected_date],
        )
    finally:
        conn.close()


def reconcile_broker_vs_sqlite(
    broker_executions: pd.DataFrame,
    sqlite_executions: pd.DataFrame,
    ibkr_portfolio: pd.DataFrame,
    sqlite_positions: pd.DataFrame,
    broker_closed_trades: pd.DataFrame,
    sqlite_closed_trades: pd.DataFrame,
    sqlite_trade_pnl: pd.DataFrame,
    *,
    selected_date: str,
    broker_status: str = "",
) -> ReconciliationResult:
    matched, missing, extra, execution_mismatches = match_executions(broker_executions, sqlite_executions)
    position_mismatches = compare_positions(ibkr_portfolio, sqlite_positions)
    if not position_mismatches.empty:
        position_mismatches = position_mismatches[position_mismatches["status"] != "MATCHED"].reset_index(drop=True)
    matched_trades, missing_trades, extra_trades, trade_mismatches = compare_closed_trades(broker_closed_trades, sqlite_closed_trades)
    broker_gross = float(pd.to_numeric(broker_closed_trades.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not broker_closed_trades.empty else 0.0
    broker_commission = float(pd.to_numeric(broker_closed_trades.get("commission", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not broker_closed_trades.empty else 0.0
    broker_net = float(pd.to_numeric(broker_closed_trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not broker_closed_trades.empty else 0.0
    sqlite_pnl_row = sqlite_trade_pnl.iloc[0].to_dict() if not sqlite_trade_pnl.empty else {}
    sqlite_net = float(sqlite_pnl_row.get("sqlite_net", 0.0) or 0.0)
    pnl_comparison = pd.DataFrame(
        [
            {
                "selected_date": selected_date,
                "broker_gross_pnl": broker_gross,
                "broker_commission": broker_commission,
                "broker_net_pnl": broker_net,
                "sqlite_gross": sqlite_pnl_row.get("sqlite_gross", 0.0),
                "sqlite_commission": sqlite_pnl_row.get("sqlite_commission", 0.0),
                "sqlite_net": sqlite_net,
                "sqlite_closed_trades": sqlite_pnl_row.get("trades", 0),
                "net_delta": broker_net - sqlite_net,
                "commission_delta": broker_commission - float(sqlite_pnl_row.get("sqlite_commission", 0.0) or 0.0),
            }
        ]
    )
    if broker_status and not str(broker_status).startswith("OK") and broker_executions.empty:
        status = "CSV_REQUIRED_FOR_HISTORICAL_DATE"
    elif broker_executions.empty and not sqlite_executions.empty:
        status = "NOT_RECONCILED"
    elif (
        missing.empty
        and extra.empty
        and execution_mismatches.empty
        and position_mismatches.empty
        and missing_trades.empty
        and extra_trades.empty
        and trade_mismatches.empty
    ):
        status = "IBKR_RECONCILED"
    else:
        status = "IBKR_MISMATCH"
    summary = {
        "status": status,
        "broker_status": broker_status,
        "broker_execution_count": int(len(broker_executions)),
        "sqlite_execution_count": int(len(sqlite_executions)),
        "matched_executions": int(len(matched)),
        "missing_in_sqlite": int(len(missing)),
        "extra_in_sqlite": int(len(extra)),
        "execution_mismatches": int(len(execution_mismatches)),
        "position_mismatches": int(len(position_mismatches)),
        "broker_closed_trades": int(len(broker_closed_trades)),
        "sqlite_closed_trades": int(len(sqlite_closed_trades)),
        "matched_trades": int(len(matched_trades)),
        "missing_trades": int(len(missing_trades)),
        "extra_trades": int(len(extra_trades)),
        "trade_mismatches": int(len(trade_mismatches)),
        "broker_gross_pnl": broker_gross,
        "broker_commission": broker_commission,
        "broker_net_pnl": broker_net,
        "sqlite_gross_pnl": float(sqlite_pnl_row.get("sqlite_gross", 0.0) or 0.0),
        "sqlite_commission": float(sqlite_pnl_row.get("sqlite_commission", 0.0) or 0.0),
        "sqlite_net_pnl": sqlite_net,
        "net_pnl_difference": broker_net - sqlite_net,
    }
    return ReconciliationResult(
        summary,
        matched,
        missing,
        extra,
        execution_mismatches,
        position_mismatches,
        broker_closed_trades,
        sqlite_closed_trades,
        matched_trades,
        missing_trades,
        extra_trades,
        trade_mismatches,
        pnl_comparison,
    )


def reconciliation_export_frames(result: ReconciliationResult) -> dict[str, pd.DataFrame]:
    return {
        "summary": pd.DataFrame([result.summary]),
        "matched_executions": result.matched,
        "missing_in_sqlite": result.missing_in_sqlite,
        "extra_in_sqlite": result.extra_in_sqlite,
        "execution_mismatches": result.execution_mismatches,
        "position_mismatches": result.position_mismatches,
        "broker_closed_trades": result.broker_closed_trades,
        "sqlite_closed_trades": result.sqlite_closed_trades,
        "matched_trades": result.matched_trades,
        "missing_trades": result.missing_trades,
        "extra_trades": result.extra_trades,
        "trade_mismatches": result.trade_mismatches,
        "pnl_comparison": result.pnl_comparison,
    }
