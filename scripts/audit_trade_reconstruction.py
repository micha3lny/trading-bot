#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.runtime_queries import DateWindow, load_dashboard_snapshot  # noqa: E402


DEFAULT_SQLITE_PATH = Path("data/runtime/trading_runtime.sqlite")
DEFAULT_OUTPUT_DIR = Path("data/analysis")
STATUS_ORDER = [
    "COMPONENT_BUY_OVERUSE",
    "COMPONENT_SELL_OVERUSE",
    "COMPONENT_QUANTITY_MISMATCH",
    "MISSING_COMPONENTS",
    "BROKER_SQLITE_MISMATCH",
    "SQLITE_FIFO_MISMATCH",
    "STALE_EXECUTIONS_TRADE_ID",
    "FIFO_DASHBOARD_MISMATCH",
    "DASHBOARD_HOLDING_TIME_MISMATCH",
    "ENTRY_PRICE_MISMATCH",
    "EXIT_PRICE_MISMATCH",
    "PNL_MISMATCH",
    "UNKNOWN",
    "OK_COMPONENT_AGGREGATE",
    "OK_COMPONENT_EXACT",
    "OK",
]
PRICE_TOLERANCE = 0.02
PNL_TOLERANCE = 0.05
QTY_TOLERANCE = 1e-6
HOLDING_TIME_TOLERANCE_MINUTES = 30.0


@dataclass
class AuditTrade:
    source: str
    symbol: str
    buy_time: str = ""
    sell_time: str = ""
    qty: float | None = None
    avg_buy: float | None = None
    avg_sell: float | None = None
    realized_pnl: float | None = None
    exit_reason: str = ""
    trade_id: str = ""
    buy_execution_ids: str = ""
    sell_execution_ids: str = ""


def parse_dt(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def iso(value: Any) -> str:
    parsed = parse_dt(value)
    return "" if parsed is None else parsed.isoformat()


def date_part(value: Any) -> str:
    parsed = parse_dt(value)
    if parsed is not None:
        return parsed.date().isoformat()
    text = str(value or "")
    return text[:10] if len(text) >= 10 else ""


def fnum(value: Any) -> float | None:
    try:
        if value in (None, "") or pd.isna(value):
            return None
        parsed = float(value)
        if math.isnan(parsed):
            return None
        return parsed
    except Exception:
        return None


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def norm_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def norm_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"BOT", "BUY", "BOUGHT", "B"}:
        return "BUY"
    if text in {"SLD", "SELL", "SOLD", "S"}:
        return "SELL"
    return text


def first_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_lower = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in by_lower:
            return by_lower[key]
    return None


def first_nonempty_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    by_lower = {str(col).strip().lower(): col for col in df.columns}
    fallback: str | None = None
    for candidate in candidates:
        key = candidate.strip().lower()
        if key not in by_lower:
            continue
        col = by_lower[key]
        if fallback is None:
            fallback = col
        values = df[col].dropna().astype(str).str.strip()
        if bool((values != "").any()):
            return col
    return fallback


def holding_minutes(start: Any, end: Any) -> float | None:
    start_dt = parse_dt(start)
    end_dt = parse_dt(end)
    if start_dt is None or end_dt is None:
        return None
    return (end_dt - start_dt).total_seconds() / 60.0


def read_csv_flexible(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def looks_like_xml(path: Path) -> bool:
    if path.suffix.lower() == ".xml":
        return True
    try:
        with path.open("rb") as handle:
            prefix = handle.read(256).lstrip()
        return prefix.startswith(b"<")
    except Exception:
        return False


def read_flex_xml(path: Path) -> pd.DataFrame:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return pd.DataFrame()
    statements = [elem for elem in root.iter() if xml_local_name(elem.tag) == "FlexStatement"]
    trades = [elem.attrib for elem in root.iter() if xml_local_name(elem.tag) == "Trade"]
    rows: list[dict[str, Any]] = []
    for item in trades:
        rows.append(
            {
                "symbol": item.get("symbol") or item.get("underlyingSymbol"),
                "buySell": item.get("buySell") or item.get("side"),
                "quantity": item.get("quantity"),
                "tradePrice": item.get("tradePrice") or item.get("price"),
                "tradeDate": item.get("tradeDate") or item.get("tradeTime") or item.get("reportDate"),
                "tradeTime": item.get("tradeTime") or item.get("dateTime"),
                "tradeID": item.get("tradeID") or item.get("ibExecID") or item.get("execID"),
                "fifoPnlRealized": item.get("fifoPnlRealized"),
                "ibCommission": item.get("ibCommission"),
                "reportDate": item.get("reportDate"),
                "raw_json": json.dumps(item, sort_keys=True),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["flex_meta"] = flex_metadata(df, source="xml", statements_count=len(statements), raw_trade_rows=len(trades))
    return df


def read_flex_file(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        df = pd.DataFrame()
        df.attrs["flex_meta"] = {"source": "missing", "statements_count": 0, "trade_rows": 0, "symbols": 0, "date_min": "", "date_max": ""}
        return df
    if looks_like_xml(path):
        return read_flex_xml(path)
    df = read_csv_flexible(path)
    df.attrs["flex_meta"] = flex_metadata(df, source="csv", statements_count=0, raw_trade_rows=len(df))
    return df


def flex_metadata(df: pd.DataFrame, *, source: str, statements_count: int, raw_trade_rows: int) -> dict[str, Any]:
    if df.empty:
        return {
            "source": source,
            "statements_count": statements_count,
            "trade_rows": 0,
            "symbols": 0,
            "date_min": "",
            "date_max": "",
        }
    cols = list(df.columns)
    symbol_col = first_col(cols, ["symbol", "underlying", "contract", "ticker", "underlyingSymbol"])
    time_col = first_nonempty_col(df, ["executed_at", "execution_time", "datetime", "date/time", "date_time", "time", "trade_time", "tradeTime", "tradeDate", "reportDate"])
    symbols = sorted({norm_symbol(x) for x in df[symbol_col].tolist() if norm_symbol(x)}) if symbol_col else []
    dates = []
    if time_col:
        dates = [date_part(x) for x in df[time_col].tolist()]
        dates = [x for x in dates if x]
    return {
        "source": source,
        "statements_count": statements_count,
        "trade_rows": raw_trade_rows,
        "symbols": len(symbols),
        "date_min": min(dates) if dates else "",
        "date_max": max(dates) if dates else "",
    }


def normalize_execution_frame(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "side", "quantity", "price", "time", "execution_id", "order_id", "realized_pnl", "commission", "raw_source"])
    cols = list(df.columns)
    symbol_col = first_col(cols, ["symbol", "underlying", "contract", "ticker"])
    side_col = first_col(cols, ["side", "action", "buySell", "buy/sell", "transactiontype", "transaction_type"])
    qty_col = first_col(cols, ["quantity", "qty", "shares", "totalquantity"])
    price_col = first_col(cols, ["price", "fill_price", "tradeprice", "tradePrice", "trade_price", "avgprice"])
    time_col = first_nonempty_col(df, ["executed_at", "execution_time", "datetime", "date/time", "date_time", "time", "trade_time", "tradeTime", "tradeDate", "reportDate"])
    exec_col = first_col(cols, ["execution_id", "execid", "exec_id", "ibexecid", "tradeID", "trade_id"])
    order_col = first_col(cols, ["order_id", "orderid", "order"])
    pnl_col = first_col(cols, ["realized_pnl", "realizedpnl", "fifoPnlRealized", "realized p/l", "realized p&l", "realizedpl"])
    comm_col = first_col(cols, ["commission", "commissions", "ibCommission"])
    out = pd.DataFrame()
    out["symbol"] = df[symbol_col].map(norm_symbol) if symbol_col else ""
    out["side"] = df[side_col].map(norm_side) if side_col else ""
    out["quantity"] = pd.to_numeric(df[qty_col], errors="coerce").abs() if qty_col else pd.NA
    out["price"] = pd.to_numeric(df[price_col], errors="coerce") if price_col else pd.NA
    out["time"] = df[time_col].map(iso) if time_col else ""
    out["execution_id"] = df[exec_col].fillna("").astype(str) if exec_col else ""
    out["order_id"] = df[order_col].fillna("").astype(str) if order_col else ""
    out["realized_pnl"] = pd.to_numeric(df[pnl_col], errors="coerce") if pnl_col else pd.NA
    out["commission"] = pd.to_numeric(df[comm_col], errors="coerce") if comm_col else pd.NA
    out["raw_source"] = source
    out = out[(out["symbol"] != "") & out["side"].isin(["BUY", "SELL"])]
    return out.sort_values(["symbol", "time", "execution_id"]).reset_index(drop=True)


def fifo_from_executions(executions: pd.DataFrame, *, selected_start: str, selected_end: str, source: str) -> list[AuditTrade]:
    if executions.empty:
        return []
    out: list[AuditTrade] = []
    for symbol, group_df in executions.sort_values(["symbol", "time", "execution_id"]).groupby("symbol", dropna=False):
        open_lots: list[dict[str, Any]] = []
        for row in group_df.to_dict("records"):
            side = norm_side(row.get("side"))
            qty = abs(fnum(row.get("quantity")) or 0.0)
            price = fnum(row.get("price"))
            if qty <= 0 or price is None:
                continue
            if side == "BUY":
                lot = dict(row)
                lot["remaining_qty"] = qty
                lot["original_qty"] = qty
                open_lots.append(lot)
                continue
            if side != "SELL":
                continue
            sell_date = date_part(row.get("time"))
            if sell_date < selected_start or sell_date > selected_end:
                continue
            remaining = qty
            while remaining > QTY_TOLERANCE and open_lots:
                lot = open_lots[0]
                lot_remaining = fnum(lot.get("remaining_qty")) or 0.0
                if lot_remaining <= QTY_TOLERANCE:
                    open_lots.pop(0)
                    continue
                matched = min(remaining, lot_remaining)
                fraction = matched / qty if qty else 1.0
                buy_price = fnum(lot.get("price"))
                sell_price = fnum(row.get("price"))
                realized = fnum(row.get("realized_pnl"))
                if realized is not None:
                    pnl = realized * fraction
                elif buy_price is not None and sell_price is not None:
                    pnl = (sell_price - buy_price) * matched
                else:
                    pnl = None
                out.append(
                    AuditTrade(
                        source=source,
                        symbol=norm_symbol(symbol),
                        buy_time=iso(lot.get("time")),
                        sell_time=iso(row.get("time")),
                        qty=matched,
                        avg_buy=buy_price,
                        avg_sell=sell_price,
                        realized_pnl=pnl,
                        trade_id=f"{source}:{norm_symbol(symbol)}:{lot.get('execution_id')}:{row.get('execution_id')}:{matched:g}",
                        buy_execution_ids=str(lot.get("execution_id") or ""),
                        sell_execution_ids=str(row.get("execution_id") or ""),
                    )
                )
                lot["remaining_qty"] = lot_remaining - matched
                remaining -= matched
                if (fnum(lot.get("remaining_qty")) or 0.0) <= QTY_TOLERANCE:
                    open_lots.pop(0)
    return out


def connect_sqlite(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", [table]).fetchone()
    return row is not None


def read_sql_df(conn: sqlite3.Connection, sql: str, params: list[Any]) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def qmarks(values: Iterable[Any]) -> str:
    return ",".join("?" for _ in values)


def execution_time(row: dict[str, Any]) -> str:
    return iso(row.get("executed_at") or row.get("time") or row.get("recorded_at"))


def execution_date(row: dict[str, Any]) -> str:
    return str(row.get("session_date") or "")[:10] or date_part(execution_time(row))


def weighted_avg(rows: list[dict[str, Any]], qty_key: str = "quantity", price_key: str = "price") -> float | None:
    total_qty = 0.0
    total_value = 0.0
    for row in rows:
        qty = abs(fnum(row.get(qty_key)) or 0.0)
        price = fnum(row.get(price_key))
        if qty <= 0 or price is None:
            continue
        total_qty += qty
        total_value += qty * price
    if total_qty <= QTY_TOLERANCE:
        return None
    return total_value / total_qty


def sum_qty(rows: list[dict[str, Any]]) -> float | None:
    total = 0.0
    seen = False
    for row in rows:
        qty = fnum(row.get("quantity"))
        if qty is None:
            continue
        total += abs(qty)
        seen = True
    return total if seen else None


def join_ids(rows: list[dict[str, Any]], key: str = "execution_id") -> str:
    seen: list[str] = []
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value and value not in seen:
            seen.append(value)
    return ",".join(seen)


def sqlite_selected_sell_executions(conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    if not table_exists(conn, "executions"):
        return pd.DataFrame()
    return read_sql_df(
        conn,
        """
        SELECT execution_id, trade_id, order_key, order_id, perm_id, strategy_name,
               session_date, symbol, side, quantity, price, executed_at, recorded_at,
               commission, realized_pnl, commission_source, raw_json
        FROM executions
        WHERE upper(side) IN ('SLD', 'SELL', 'SOLD', 'S')
          AND (
            session_date BETWEEN ? AND ?
            OR substr(COALESCE(executed_at, recorded_at), 1, 10) BETWEEN ? AND ?
          )
        ORDER BY symbol, COALESCE(executed_at, recorded_at), execution_id
        """,
        [start_date, end_date, start_date, end_date],
    )


def sqlite_executions_for_symbols(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    if not symbols or not table_exists(conn, "executions"):
        return pd.DataFrame()
    placeholders = qmarks(symbols)
    return read_sql_df(
        conn,
        f"""
        SELECT execution_id, trade_id, order_key, order_id, perm_id, strategy_name,
               session_date, symbol, side, quantity, price, executed_at, recorded_at,
               commission, realized_pnl, commission_source, raw_json
        FROM executions
        WHERE upper(symbol) IN ({placeholders})
        ORDER BY symbol, COALESCE(executed_at, recorded_at), execution_id
        """,
        symbols,
    )


def sqlite_all_executions_range(conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    if not table_exists(conn, "executions"):
        return pd.DataFrame()
    return read_sql_df(
        conn,
        """
        SELECT execution_id, trade_id, order_key, order_id, perm_id, strategy_name,
               session_date, symbol, side, quantity, price, executed_at, recorded_at,
               commission, realized_pnl, commission_source, raw_json
        FROM executions
        WHERE session_date BETWEEN ? AND ?
           OR substr(COALESCE(executed_at, recorded_at), 1, 10) BETWEEN ? AND ?
        ORDER BY symbol, COALESCE(executed_at, recorded_at), execution_id
        """,
        [start_date, end_date, start_date, end_date],
    )


def sqlite_trades_for_symbols(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    if not symbols or not table_exists(conn, "trades"):
        return pd.DataFrame()
    placeholders = qmarks(symbols)
    return read_sql_df(
        conn,
        f"""
        SELECT *
        FROM trades
        WHERE upper(symbol) IN ({placeholders})
        ORDER BY symbol, COALESCE(exit_fill_time, closed_at, entry_fill_time), trade_id
        """,
        symbols,
    )


def sqlite_positions_for_symbols(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    if not symbols or not table_exists(conn, "positions"):
        return pd.DataFrame()
    placeholders = qmarks(symbols)
    return read_sql_df(
        conn,
        f"""
        SELECT position_key, strategy_name, session_date, symbol, status, quantity,
               avg_price, updated_at, raw_json
        FROM positions
        WHERE upper(symbol) IN ({placeholders})
        ORDER BY symbol, session_date, updated_at, position_key
        """,
        symbols,
    )


def sqlite_trade_components_for_sell_ids(conn: sqlite3.Connection, sell_ids: set[str]) -> pd.DataFrame:
    if not sell_ids or not table_exists(conn, "trade_components"):
        return pd.DataFrame()
    ids = sorted(str(value) for value in sell_ids if str(value or "").strip())
    if not ids:
        return pd.DataFrame()
    placeholders = qmarks(ids)
    return read_sql_df(
        conn,
        f"""
        SELECT component_id, trade_id, symbol, session_date,
               buy_execution_id, sell_execution_id, matched_qty,
               buy_price, sell_price, entry_time, exit_time,
               buy_commission_alloc, sell_commission_alloc,
               realized_pnl_alloc, gross_pnl, net_pnl, raw_json, updated_at
        FROM trade_components
        WHERE sell_execution_id IN ({placeholders})
        ORDER BY sell_execution_id, entry_time, buy_execution_id, component_id
        """,
        ids,
    )


def component_summaries_by_sell(components: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if components.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for sell_id, group in components.groupby("sell_execution_id", dropna=False):
        records = group.to_dict("records")
        matched_qty = sum(abs(fnum(row.get("matched_qty")) or 0.0) for row in records)
        trade_ids = [str(row.get("trade_id") or "") for row in records if str(row.get("trade_id") or "")]
        buy_ids = [str(row.get("buy_execution_id") or "") for row in records if str(row.get("buy_execution_id") or "")]
        buy_times = [iso(row.get("entry_time")) for row in records if iso(row.get("entry_time"))]
        buy_qty_values = [str(fnum(row.get("matched_qty")) or 0.0) for row in records]
        entry_price = weighted_avg(
            [{"quantity": row.get("matched_qty"), "price": row.get("buy_price")} for row in records]
        )
        exit_price = weighted_avg(
            [{"quantity": row.get("matched_qty"), "price": row.get("sell_price")} for row in records]
        )
        out[str(sell_id or "")] = {
            "canonical_trade_id": ",".join(dict.fromkeys(trade_ids)),
            "component_count": len(records),
            "component_buy_execution_ids": ",".join(dict.fromkeys(buy_ids)),
            "component_buy_times": ",".join(dict.fromkeys(buy_times)),
            "component_matched_qty": ",".join(buy_qty_values),
            "component_total_matched_qty": matched_qty,
            "canonical_entry_quantity": matched_qty,
            "canonical_exit_quantity": matched_qty,
            "canonical_entry_price": entry_price,
            "canonical_exit_price": exit_price,
        }
    return out


def same_session_fifo_candidates(executions: pd.DataFrame, selected_sell_ids: set[str]) -> dict[str, dict[str, Any]]:
    if executions.empty:
        return {}
    rows = executions.to_dict("records")
    for row in rows:
        row["_symbol_norm"] = norm_symbol(row.get("symbol"))
        row["_side_norm"] = norm_side(row.get("side"))
        row["_time_norm"] = execution_time(row)
        row["_session_norm"] = execution_date(row)
    out: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["_symbol_norm"], row["_session_norm"]), []).append(row)
    for group_rows in grouped.values():
        open_lots: list[dict[str, Any]] = []
        for row in sorted(group_rows, key=lambda item: (item.get("_time_norm") or "", str(item.get("execution_id") or ""))):
            side = row.get("_side_norm")
            qty = abs(fnum(row.get("quantity")) or 0.0)
            price = fnum(row.get("price"))
            if qty <= 0 or price is None:
                continue
            if side == "BUY":
                lot = dict(row)
                lot["remaining_qty"] = qty
                open_lots.append(lot)
                continue
            if side != "SELL":
                continue
            remaining = qty
            portions: list[dict[str, Any]] = []
            lots_before = len([lot for lot in open_lots if (fnum(lot.get("remaining_qty")) or 0.0) > QTY_TOLERANCE])
            while remaining > QTY_TOLERANCE and open_lots:
                lot = open_lots[0]
                lot_remaining = fnum(lot.get("remaining_qty")) or 0.0
                if lot_remaining <= QTY_TOLERANCE:
                    open_lots.pop(0)
                    continue
                matched = min(remaining, lot_remaining)
                portion = dict(lot)
                portion["quantity"] = matched
                portions.append(portion)
                lot["remaining_qty"] = lot_remaining - matched
                remaining -= matched
                if (fnum(lot.get("remaining_qty")) or 0.0) <= QTY_TOLERANCE:
                    open_lots.pop(0)
            sell_id = str(row.get("execution_id") or "")
            if sell_id in selected_sell_ids:
                out[sell_id] = {
                    "candidate_entry_execution_id": join_ids(portions),
                    "candidate_entry_time": ",".join(x for x in [execution_time(p) for p in portions] if x),
                    "candidate_entry_quantity": sum_qty(portions),
                    "candidate_entry_price": weighted_avg(portions),
                    "candidate_entry_count": len(portions),
                    "candidate_ambiguous": int(len(portions) > 1 or lots_before > 1),
                    "candidate_unmatched_qty": max(remaining, 0.0),
                    "candidate_holding_minutes": holding_minutes(portions[0].get("_time_norm") if portions else None, row.get("_time_norm")),
                }
    return out


def trade_rows_by_id(trades: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades.empty or "trade_id" not in trades.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in trades.to_dict("records"):
        trade_id = str(row.get("trade_id") or "")
        if trade_id and trade_id not in out:
            out[trade_id] = row
    return out


def parse_trade_id_entry_ids(trade_id: str) -> list[str]:
    if not trade_id:
        return []
    parts = trade_id.split(":")
    if len(parts) >= 6 and parts[0] == "reconstructed":
        return [parts[-2]]
    return []


def current_trade_entry(
    sell: dict[str, Any],
    executions: pd.DataFrame,
    trades_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trade_id = str(sell.get("trade_id") or "")
    if not trade_id:
        return {}
    all_rows = executions.to_dict("records") if not executions.empty else []
    sell_time = execution_time(sell)
    buy_rows = [
        row for row in all_rows
        if str(row.get("trade_id") or "") == trade_id
        and norm_side(row.get("side")) == "BUY"
        and norm_symbol(row.get("symbol")) == norm_symbol(sell.get("symbol"))
        and (not sell_time or execution_time(row) <= sell_time)
    ]
    if not buy_rows:
        entry_ids = parse_trade_id_entry_ids(trade_id)
        trade_row = trades_by_id.get(trade_id) or {}
        raw = parse_json(trade_row.get("raw_json"))
        raw_ids = raw.get("buy_execution_ids") or raw.get("entry_execution_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        entry_ids.extend(str(x) for x in raw_ids if x)
        entry_ids = [x for x in dict.fromkeys(entry_ids) if x]
        if entry_ids:
            buy_rows = [row for row in all_rows if str(row.get("execution_id") or "") in entry_ids]
    if not buy_rows:
        trade_row = trades_by_id.get(trade_id) or {}
        if trade_row:
            return {
                "entry_execution_id": "",
                "entry_time": iso(trade_row.get("entry_fill_time")),
                "entry_quantity": fnum(trade_row.get("quantity")),
                "entry_price": fnum(trade_row.get("entry_price")),
            }
        return {}
    buy_rows = sorted(buy_rows, key=lambda row: (execution_time(row), str(row.get("execution_id") or "")))
    return {
        "entry_execution_id": join_ids(buy_rows),
        "entry_time": ",".join(x for x in [execution_time(row) for row in buy_rows] if x),
        "entry_quantity": sum_qty(buy_rows),
        "entry_price": weighted_avg(buy_rows),
    }


def copy_trade_for_sqlite_sell(trade: AuditTrade, sqlite_sell: dict[str, Any], flex_sell_qty: float | None) -> AuditTrade:
    sqlite_qty = abs(fnum(sqlite_sell.get("quantity")) or 0.0)
    fraction = sqlite_qty / flex_sell_qty if flex_sell_qty and flex_sell_qty > QTY_TOLERANCE else 1.0
    return AuditTrade(
        source=trade.source,
        symbol=trade.symbol,
        buy_time=trade.buy_time,
        sell_time=execution_time(sqlite_sell) or trade.sell_time,
        qty=sqlite_qty or trade.qty,
        avg_buy=trade.avg_buy,
        avg_sell=fnum(sqlite_sell.get("price")) or trade.avg_sell,
        realized_pnl=(trade.realized_pnl * fraction) if trade.realized_pnl is not None else None,
        exit_reason=trade.exit_reason,
        trade_id=trade.trade_id,
        buy_execution_ids=trade.buy_execution_ids,
        sell_execution_ids=trade.sell_execution_ids,
    )


def combine_audit_trades(trades: list[AuditTrade], sqlite_sell: dict[str, Any]) -> AuditTrade | None:
    if not trades:
        return None
    qty_total = sum(abs(fnum(t.qty) or 0.0) for t in trades)
    if qty_total <= QTY_TOLERANCE:
        qty_total = abs(fnum(sqlite_sell.get("quantity")) or 0.0)
    avg_buy = sum((abs(fnum(t.qty) or 0.0) * (fnum(t.avg_buy) or 0.0)) for t in trades) / qty_total if qty_total > QTY_TOLERANCE else None
    avg_sell = sum((abs(fnum(t.qty) or 0.0) * (fnum(t.avg_sell) or 0.0)) for t in trades) / qty_total if qty_total > QTY_TOLERANCE else None
    pnl_values = [fnum(t.realized_pnl) for t in trades if fnum(t.realized_pnl) is not None]
    return AuditTrade(
        source="ibkr_flex",
        symbol=trades[0].symbol,
        buy_time=trades[0].buy_time,
        sell_time=execution_time(sqlite_sell) or trades[-1].sell_time,
        qty=qty_total,
        avg_buy=avg_buy,
        avg_sell=avg_sell,
        realized_pnl=sum(pnl_values) if pnl_values else None,
        trade_id=";".join(t.trade_id for t in trades if t.trade_id),
        buy_execution_ids=",".join(x for t in trades for x in str(t.buy_execution_ids or "").split(",") if x),
        sell_execution_ids=",".join(x for t in trades for x in str(t.sell_execution_ids or "").split(",") if x),
    )


def broker_trade_by_sell_execution(
    flex_file: Path | None,
    start_date: str,
    end_date: str,
    sqlite_matches: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, AuditTrade]:
    broker_df = normalize_execution_frame(read_flex_file(flex_file), source="ibkr_flex") if flex_file else pd.DataFrame()
    if broker_df.empty:
        return {}
    flex_to_sqlite: dict[str, list[dict[str, Any]]] = {}
    if sqlite_matches:
        for _sqlite_id, flex_rows in sqlite_matches.items():
            for flex_row in flex_rows:
                flex_id = str(flex_row.get("flex_trade_id") or flex_row.get("execution_id") or "")
                sqlite_row = flex_row.get("_sqlite_row")
                if flex_id and isinstance(sqlite_row, dict):
                    flex_to_sqlite.setdefault(flex_id, []).append(sqlite_row)
    broker_df["_date"] = broker_df["time"].map(date_part)
    broker_df = broker_df[(broker_df["_date"] >= start_date) & (broker_df["_date"] <= end_date)].drop(columns=["_date"])
    out: dict[str, AuditTrade] = {}
    for trade in fifo_from_executions(broker_df, selected_start=start_date, selected_end=end_date, source="ibkr_flex"):
        for sell_id in str(trade.sell_execution_ids or "").split(","):
            sell_id = sell_id.strip()
            matched_sqlite = flex_to_sqlite.get(sell_id, [])
            if matched_sqlite:
                flex_sell_qty = trade.qty
                for sqlite_sell in matched_sqlite:
                    sqlite_id = str(sqlite_sell.get("execution_id") or "")
                    if sqlite_id:
                        out.setdefault(sqlite_id, copy_trade_for_sqlite_sell(trade, sqlite_sell, flex_sell_qty))
            elif sell_id and sell_id not in out:
                out[sell_id] = trade
    if sqlite_matches:
        for sqlite_id, flex_rows in sqlite_matches.items():
            if sqlite_id in out:
                continue
            flex_ids = {str(row.get("flex_trade_id") or row.get("execution_id") or "") for row in flex_rows}
            portions = []
            sqlite_sell = flex_rows[0].get("_sqlite_row") if flex_rows and isinstance(flex_rows[0].get("_sqlite_row"), dict) else {}
            for trade in fifo_from_executions(broker_df, selected_start=start_date, selected_end=end_date, source="ibkr_flex"):
                if any(x in flex_ids for x in str(trade.sell_execution_ids or "").split(",")):
                    portions.append(copy_trade_for_sqlite_sell(trade, sqlite_sell, trade.qty))
            combined = combine_audit_trades(portions, sqlite_sell)
            if combined is not None:
                out[sqlite_id] = combined
    return out


def sqlite_executions(conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    if not table_exists(conn, "executions"):
        return pd.DataFrame()
    rows = read_sql_df(
        conn,
        """
        SELECT execution_id, order_id, session_date, symbol, side, quantity, price,
               executed_at AS time, recorded_at, realized_pnl, commission, raw_json
        FROM executions
        WHERE session_date BETWEEN ? AND ?
           OR substr(COALESCE(executed_at, recorded_at), 1, 10) BETWEEN ? AND ?
        ORDER BY symbol, COALESCE(executed_at, recorded_at), execution_id
        """,
        [start_date, end_date, start_date, end_date],
    )
    return normalize_execution_frame(rows, source="sqlite_executions")


def sqlite_fills(conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    for table in ("fills", "execution_fills"):
        if not table_exists(conn, table):
            continue
        rows = read_sql_df(conn, f"SELECT * FROM {table}", [])
        normalized = normalize_execution_frame(rows, source=f"sqlite_{table}")
        if normalized.empty:
            return normalized
        normalized["_date"] = normalized["time"].map(date_part)
        return normalized[(normalized["_date"] >= start_date) & (normalized["_date"] <= end_date)].drop(columns=["_date"]).reset_index(drop=True)
    return pd.DataFrame()


def reconstructed_trades(conn: sqlite3.Connection, start_date: str, end_date: str) -> list[AuditTrade]:
    if not table_exists(conn, "trades"):
        return []
    rows = read_sql_df(
        conn,
        """
        SELECT trade_id, session_date, symbol, status, entry_fill_time, exit_fill_time,
               closed_at, entry_price, exit_price, quantity, gross_pnl, commission, net_pnl,
               exit_reason, raw_json
        FROM trades
        WHERE (
            trade_id LIKE 'reconstructed:%'
            OR raw_json LIKE '%sqlite_execution_reducer%'
            OR raw_json LIKE '%executions_pair%'
            OR raw_json LIKE '%reconstructed%'
        )
          AND (
            substr(COALESCE(exit_fill_time, closed_at), 1, 10) BETWEEN ? AND ?
            OR session_date BETWEEN ? AND ?
          )
        ORDER BY symbol, COALESCE(exit_fill_time, closed_at, entry_fill_time), trade_id
        """,
        [start_date, end_date, start_date, end_date],
    )
    out: list[AuditTrade] = []
    for row in rows.to_dict("records"):
        raw = parse_json(row.get("raw_json"))
        buy_ids = raw.get("buy_execution_ids") or [raw.get("buy_execution_id")]
        sell_ids = raw.get("sell_execution_ids") or [raw.get("sell_execution_id")]
        out.append(
            AuditTrade(
                source="sqlite_reconstructed_trades",
                symbol=norm_symbol(row.get("symbol")),
                buy_time=iso(row.get("entry_fill_time") or raw.get("entry_executed_at")),
                sell_time=iso(row.get("exit_fill_time") or row.get("closed_at") or raw.get("exit_executed_at")),
                qty=fnum(row.get("quantity")),
                avg_buy=fnum(row.get("entry_price") or raw.get("weighted_entry_price")),
                avg_sell=fnum(row.get("exit_price") or raw.get("weighted_exit_price")),
                realized_pnl=fnum(row.get("net_pnl")),
                exit_reason=str(row.get("exit_reason") or raw.get("exit_reason") or ""),
                trade_id=str(row.get("trade_id") or ""),
                buy_execution_ids=",".join(str(x) for x in buy_ids if x),
                sell_execution_ids=",".join(str(x) for x in sell_ids if x),
            )
        )
    return out


def dashboard_closed_positions(sqlite_path: Path, start_date: str, end_date: str, include_reconstructed: bool) -> list[AuditTrade]:
    try:
        snapshot = load_dashboard_snapshot(sqlite_path, DateWindow(start_date, end_date), "All", include_reconstructed=include_reconstructed)
    except Exception:
        return []
    closed = snapshot.get("closed_positions", pd.DataFrame())
    if not isinstance(closed, pd.DataFrame) or closed.empty:
        return []
    out: list[AuditTrade] = []
    for row in closed.to_dict("records"):
        out.append(
            AuditTrade(
                source="dashboard_closed_positions",
                symbol=norm_symbol(row.get("symbol")),
                buy_time=iso(row.get("entry_time")),
                sell_time=iso(row.get("exit_time")),
                qty=fnum(row.get("qty")),
                avg_buy=fnum(row.get("buy")),
                avg_sell=fnum(row.get("sell")),
                realized_pnl=fnum(row.get("net_actual") or row.get("gross")),
                exit_reason=str(row.get("exit_reason") or ""),
                trade_id=str(row.get("trade_id") or ""),
            )
        )
    return out


def dashboard_closed_rows(sqlite_path: Path, start_date: str, end_date: str, include_reconstructed: bool) -> list[dict[str, Any]]:
    try:
        snapshot = load_dashboard_snapshot(sqlite_path, DateWindow(start_date, end_date), "All", include_reconstructed=include_reconstructed)
    except Exception:
        return []
    closed = snapshot.get("closed_positions", pd.DataFrame())
    if not isinstance(closed, pd.DataFrame) or closed.empty:
        return []
    return closed.to_dict("records")


def dashboard_closed_rows_from_trades(trades: pd.DataFrame, start_date: str, end_date: str) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    out: list[dict[str, Any]] = []
    for row in trades.to_dict("records"):
        exit_time = row.get("exit_fill_time") or row.get("closed_at")
        exit_date = date_part(exit_time)
        if exit_date < start_date or exit_date > end_date:
            continue
        raw = parse_json(row.get("raw_json"))
        data_quality = raw.get("data_quality") or raw.get("quality") or ""
        if str(row.get("trade_id") or "").startswith("reconstructed:") and "RECONSTRUCTED" not in str(data_quality).upper():
            data_quality = f"{data_quality}; RECONSTRUCTED_CLOSED_METADATA".strip("; ")
        out.append(
            {
                "symbol": row.get("symbol"),
                "entry_time": row.get("entry_fill_time"),
                "exit_time": exit_time,
                "qty": row.get("quantity"),
                "buy": row.get("entry_price"),
                "sell": row.get("exit_price"),
                "net_actual": row.get("net_pnl"),
                "gross": row.get("gross_pnl"),
                "exit_reason": row.get("exit_reason") or raw.get("exit_reason"),
                "trade_id": row.get("trade_id"),
                "data_quality": data_quality,
                "source": "trades",
            }
        )
    return out


def infer_dashboard_closed_source(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    haystack = " ".join(str(row.get(key) or "") for key in ("data_quality", "source", "trade_id", "raw_json")).upper()
    if "UNATTRIBUTED_EXECUTION_CLOSED" in haystack or "EXECUTION" in haystack and "RECONSTRUCTED" not in haystack:
        return "executions.realized_pnl"
    if "RECONSTRUCTED" in haystack or str(row.get("trade_id") or "").startswith("reconstructed:"):
        return "reconstructed FIFO"
    if row.get("trade_id"):
        return "table trades"
    if "POSITION" in haystack or "RAW_JSON" in haystack or row.get("position_key"):
        return "positions/raw_json"
    return "unknown"


def match_dashboard_row(sell: dict[str, Any], dashboard_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    symbol = norm_symbol(sell.get("symbol"))
    sell_time = parse_dt(execution_time(sell))
    sell_qty = abs(fnum(sell.get("quantity")) or 0.0)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in dashboard_rows:
        if norm_symbol(row.get("symbol")) != symbol:
            continue
        row_exit = parse_dt(row.get("exit_time") or row.get("closed_at"))
        if sell_time is not None and row_exit is not None and sell_time.date() != row_exit.date():
            continue
        row_qty = abs(fnum(row.get("qty") or row.get("quantity")) or 0.0)
        qty_penalty = abs(row_qty - sell_qty) if row_qty and sell_qty else 0.0
        time_penalty = abs((row_exit - sell_time).total_seconds()) / 60.0 if row_exit is not None and sell_time is not None else 999999.0
        candidates.append((time_penalty + qty_penalty, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def load_symbol_candles(history_dir: Path | None, symbol: str, session_date: str) -> pd.DataFrame:
    if history_dir is None:
        return pd.DataFrame()
    day = parse_dt(session_date)
    if day is None:
        return pd.DataFrame()
    path = (
        history_dir
        / "session_type=RTH"
        / f"symbol={norm_symbol(symbol)}"
        / f"year={day.year:04d}"
        / f"month={day.month:02d}"
        / f"day={day.day:02d}.parquet"
    )
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=["timestamp", "high"])
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    time_col = first_col(df.columns, ["timestamp", "bar_time", "time", "datetime"])
    high_col = first_col(df.columns, ["high", "High"])
    if not time_col or not high_col:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    out["high"] = pd.to_numeric(df[high_col], errors="coerce")
    return out.dropna(subset=["timestamp", "high"]).sort_values("timestamp")


def candle_peak(
    history_dir: Path | None,
    symbol: str,
    entry_time: Any,
    sell_time: Any,
    entry_price: float | None,
) -> dict[str, Any]:
    if history_dir is None or entry_price is None or entry_price <= 0:
        return {}
    entry_dt = parse_dt(entry_time)
    sell_dt = parse_dt(sell_time)
    if entry_dt is None or sell_dt is None:
        return {}
    candles = load_symbol_candles(history_dir, symbol, entry_dt.date().isoformat())
    if candles.empty:
        return {}
    window = candles[(candles["timestamp"] >= entry_dt) & (candles["timestamp"] <= sell_dt)]
    if window.empty:
        return {}
    idx = window["high"].idxmax()
    peak_price = fnum(window.loc[idx, "high"])
    if peak_price is None:
        return {}
    return {
        "peak_price": peak_price,
        "peak_pct": (peak_price / entry_price - 1.0) * 100.0,
        "peak_source": "reconstructed_from_candles",
        "peak_position_key": "",
        "peak_match_status": "symbol_session_match",
    }


def position_peak(
    positions: pd.DataFrame,
    *,
    symbol: str,
    session_date: str,
    entry_execution_ids: str,
) -> dict[str, Any]:
    if positions.empty:
        return {}
    rows = [
        row for row in positions.to_dict("records")
        if norm_symbol(row.get("symbol")) == norm_symbol(symbol)
        and (str(row.get("session_date") or "")[:10] == session_date or not row.get("session_date"))
    ]
    if not rows:
        return {}
    entry_ids = [x.strip() for x in str(entry_execution_ids or "").split(",") if x.strip()]
    exact: list[dict[str, Any]] = []
    for row in rows:
        raw_text = str(row.get("raw_json") or "")
        if entry_ids and any(entry_id in raw_text for entry_id in entry_ids):
            exact.append(row)
    if len(exact) == 1:
        raw = parse_json(exact[0].get("raw_json"))
        peak_price = fnum(raw.get("peak_price"))
        peak_pct = fnum(raw.get("peak_pct") or raw.get("peak_unrealized_pct"))
        return {
            "peak_price": peak_price,
            "peak_pct": peak_pct,
            "peak_source": "live_position_json" if peak_price is not None or peak_pct is not None else "unavailable",
            "peak_position_key": str(exact[0].get("position_key") or ""),
            "peak_match_status": "exact_trade_match" if peak_price is not None or peak_pct is not None else "unavailable",
        }
    peak_rows: list[dict[str, Any]] = []
    for row in rows:
        raw = parse_json(row.get("raw_json"))
        if fnum(raw.get("peak_price")) is not None or fnum(raw.get("peak_pct") or raw.get("peak_unrealized_pct")) is not None:
            peak_rows.append(row)
    if len(peak_rows) == 1 and not entry_ids:
        raw = parse_json(peak_rows[0].get("raw_json"))
        return {
            "peak_price": fnum(raw.get("peak_price")),
            "peak_pct": fnum(raw.get("peak_pct") or raw.get("peak_unrealized_pct")),
            "peak_source": "live_position_json",
            "peak_position_key": str(peak_rows[0].get("position_key") or ""),
            "peak_match_status": "symbol_session_match",
        }
    if peak_rows:
        return {
            "peak_price": None,
            "peak_pct": None,
            "peak_source": "unavailable",
            "peak_position_key": ",".join(str(row.get("position_key") or "") for row in peak_rows[:5]),
            "peak_match_status": "ambiguous",
        }
    return {}


def classify_sell_problem(
    sell: dict[str, Any],
    current_entry: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    problems: list[str] = []
    trade_id = str(sell.get("trade_id") or "")
    sell_qty = abs(fnum(sell.get("quantity")) or 0.0)
    sell_date = date_part(execution_time(sell))
    current_entry_date = date_part(current_entry.get("entry_time"))
    current_qty = fnum(current_entry.get("entry_quantity"))
    candidate_ids = str(candidate.get("candidate_entry_execution_id") or "")
    if not trade_id:
        problems.append("missing_trade_id")
    if not candidate_ids:
        problems.append("unmatched_sell")
    if current_entry_date and sell_date and current_entry_date != sell_date:
        problems.append("cross_session_trade_id")
    if current_qty is not None and sell_qty and not close_enough(current_qty, sell_qty, QTY_TOLERANCE):
        problems.append("wrong_entry_quantity")
    if candidate_ids:
        current_ids = str(current_entry.get("entry_execution_id") or "")
        if not current_ids or current_ids != candidate_ids or "cross_session_trade_id" in problems:
            problems.append("same_session_candidate_exists")
    if int(candidate.get("candidate_ambiguous") or 0):
        problems.append("ambiguous_match")
    return ";".join(dict.fromkeys(problems)) or "OK"


def build_sell_level_audit(
    *,
    conn: sqlite3.Connection,
    sqlite_path: Path,
    start_date: str,
    end_date: str,
    flex_file: Path | None,
    sqlite_flex_matches: dict[str, list[dict[str, Any]]] | None,
    history_dir: Path | None,
    include_dashboard_reconstructed: bool,
) -> pd.DataFrame:
    selected_sells = sqlite_selected_sell_executions(conn, start_date, end_date)
    if selected_sells.empty:
        return pd.DataFrame()
    selected_sells["symbol"] = selected_sells["symbol"].map(norm_symbol)
    symbols = sorted({norm_symbol(x) for x in selected_sells["symbol"].tolist() if norm_symbol(x)})
    all_execs = sqlite_executions_for_symbols(conn, symbols)
    trades = sqlite_trades_for_symbols(conn, symbols)
    positions = sqlite_positions_for_symbols(conn, symbols)
    trades_by_id = trade_rows_by_id(trades)
    selected_sell_ids = {str(x) for x in selected_sells["execution_id"].fillna("").astype(str).tolist() if str(x)}
    candidates = same_session_fifo_candidates(all_execs, selected_sell_ids)
    component_summaries = component_summaries_by_sell(sqlite_trade_components_for_sell_ids(conn, selected_sell_ids))
    broker_by_sell = broker_trade_by_sell_execution(flex_file, start_date, end_date, sqlite_flex_matches)
    dashboard_rows = dashboard_closed_rows_from_trades(trades, start_date, end_date)
    rows: list[dict[str, Any]] = []
    for sell in selected_sells.to_dict("records"):
        sell_id = str(sell.get("execution_id") or "")
        symbol = norm_symbol(sell.get("symbol"))
        sell_time = execution_time(sell)
        sell_qty = abs(fnum(sell.get("quantity")) or 0.0)
        sell_price = fnum(sell.get("price"))
        current_entry = current_trade_entry(sell, all_execs, trades_by_id)
        candidate = candidates.get(sell_id, {})
        component = component_summaries.get(sell_id, {})
        broker = broker_by_sell.get(sell_id)
        dash = match_dashboard_row(sell, dashboard_rows)
        problem = classify_sell_problem(sell, current_entry, candidate)
        peak_entry_ids = str(current_entry.get("entry_execution_id") or "")
        if any(flag in problem for flag in ("cross_session_trade_id", "wrong_entry_quantity", "same_session_candidate_exists")):
            peak_entry_ids = str(candidate.get("candidate_entry_execution_id") or "")
        peak = position_peak(
            positions,
            symbol=symbol,
            session_date=date_part(sell_time),
            entry_execution_ids=peak_entry_ids,
        )
        if not peak or peak.get("peak_source") == "unavailable":
            candle = candle_peak(
                history_dir,
                symbol,
                current_entry.get("entry_time") or candidate.get("candidate_entry_time"),
                sell_time,
                fnum(current_entry.get("entry_price")) or fnum(candidate.get("candidate_entry_price")),
            )
            if candle:
                peak = candle
        if not peak:
            peak = {
                "peak_price": None,
                "peak_pct": None,
                "peak_source": "unavailable",
                "peak_position_key": "",
                "peak_match_status": "unavailable",
            }
        broker_qty = broker.qty if broker else None
        broker_entry = broker.avg_buy if broker else None
        broker_exit = broker.avg_sell if broker else None
        broker_pnl = broker.realized_pnl if broker else None
        sqlite_pnl = fnum(sell.get("realized_pnl"))
        row = {
            "sell_execution_id": sell_id,
            "symbol": symbol,
            "sell_time": sell_time,
            "sell_quantity": sell_qty,
            "sell_price": sell_price,
            "current_executions_trade_id": str(sell.get("trade_id") or ""),
            "entry_execution_id": current_entry.get("entry_execution_id", ""),
            "entry_time": current_entry.get("entry_time", ""),
            "entry_quantity": current_entry.get("entry_quantity"),
            "entry_price": current_entry.get("entry_price"),
            "candidate_entry_execution_id": candidate.get("candidate_entry_execution_id", ""),
            "candidate_entry_time": candidate.get("candidate_entry_time", ""),
            "candidate_entry_quantity": candidate.get("candidate_entry_quantity"),
            "candidate_entry_price": candidate.get("candidate_entry_price"),
            "candidate_entry_count": candidate.get("candidate_entry_count"),
            "candidate_unmatched_qty": candidate.get("candidate_unmatched_qty"),
            "problem_classification": problem,
            "canonical_trade_id": component.get("canonical_trade_id", ""),
            "component_count": component.get("component_count", 0),
            "component_buy_execution_ids": component.get("component_buy_execution_ids", ""),
            "component_buy_times": component.get("component_buy_times", ""),
            "component_matched_qty": component.get("component_matched_qty", ""),
            "component_total_matched_qty": component.get("component_total_matched_qty"),
            "sell_quantity_conservation_diff": (
                sell_qty - fnum(component.get("component_total_matched_qty"))
                if fnum(component.get("component_total_matched_qty")) is not None
                else None
            ),
            "canonical_entry_quantity": component.get("canonical_entry_quantity"),
            "canonical_exit_quantity": component.get("canonical_exit_quantity"),
            "canonical_entry_price": component.get("canonical_entry_price"),
            "canonical_exit_price": component.get("canonical_exit_price"),
            "holding_minutes_current_assignment": holding_minutes(current_entry.get("entry_time"), sell_time),
            "holding_minutes_same_session_candidate": candidate.get("candidate_holding_minutes"),
            "ibkr_flex_quantity": broker_qty,
            "sqlite_ibkr_quantity_diff": (sell_qty - broker_qty) if broker_qty is not None else None,
            "ibkr_flex_entry_price": broker_entry,
            "sqlite_ibkr_entry_price_diff": (fnum(current_entry.get("entry_price")) - broker_entry) if broker_entry is not None and fnum(current_entry.get("entry_price")) is not None else None,
            "ibkr_flex_exit_price": broker_exit,
            "sqlite_ibkr_exit_price_diff": (sell_price - broker_exit) if broker_exit is not None and sell_price is not None else None,
            "sqlite_realized_pnl": sqlite_pnl,
            "ibkr_flex_realized_pnl": broker_pnl,
            "sqlite_ibkr_realized_pnl_diff": (sqlite_pnl - broker_pnl) if sqlite_pnl is not None and broker_pnl is not None else None,
            "dashboard_closed_source": infer_dashboard_closed_source(dash),
            "dashboard_entry_time": iso(dash.get("entry_time")) if dash else "",
            "dashboard_exit_time": iso(dash.get("exit_time")) if dash else "",
            "dashboard_entry_price": fnum(dash.get("buy")) if dash else None,
            "dashboard_exit_price": fnum(dash.get("sell")) if dash else None,
            "dashboard_realized_pnl": fnum(dash.get("net_actual") or dash.get("gross")) if dash else None,
            "dashboard_data_quality": str(dash.get("data_quality") or "") if dash else "",
            "peak_price": peak.get("peak_price"),
            "peak_pct": peak.get("peak_pct"),
            "peak_source": peak.get("peak_source", "unavailable"),
            "peak_position_key": peak.get("peak_position_key", ""),
            "peak_match_status": peak.get("peak_match_status", "unavailable"),
        }
        row["status"] = sell_level_status(row)
        row["mismatch_score"] = mismatch_score(
            {
                "broker_avg_buy": broker_entry,
                "dashboard_entry_price": row["dashboard_entry_price"],
                "broker_avg_sell": broker_exit,
                "dashboard_exit_price": row["dashboard_exit_price"],
                "broker_realized_pnl": broker_pnl,
                "dashboard_realized_pnl": row["dashboard_realized_pnl"],
                "holding_minutes_broker": holding_minutes(broker.buy_time, broker.sell_time) if broker else None,
                "holding_minutes_dashboard": holding_minutes(row["dashboard_entry_time"], row["dashboard_exit_time"]),
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["status", "mismatch_score", "symbol", "sell_time"], ascending=[True, False, True, True]).reset_index(drop=True)
    return df


def sell_level_status(row: dict[str, Any]) -> str:
    component_count = int(fnum(row.get("component_count")) or 0)
    component_diff = fnum(row.get("sell_quantity_conservation_diff"))
    if component_count > 0:
        if component_diff is not None and abs(component_diff) > QTY_TOLERANCE:
            return "COMPONENT_QUANTITY_MISMATCH"
        if component_count == 1:
            return "OK_COMPONENT_EXACT"
        return "OK_COMPONENT_AGGREGATE"
    problem = str(row.get("problem_classification") or "")
    if "cross_session_trade_id" in problem or "wrong_entry_quantity" in problem or "same_session_candidate_exists" in problem:
        return "SQLITE_FIFO_MISMATCH"
    if row.get("dashboard_entry_time") and row.get("entry_time") and date_part(row["dashboard_entry_time"]) != date_part(row["entry_time"]):
        return "FIFO_DASHBOARD_MISMATCH"
    current_hold = fnum(row.get("holding_minutes_current_assignment"))
    dashboard_hold = holding_minutes(row.get("dashboard_entry_time"), row.get("dashboard_exit_time"))
    if current_hold is not None and dashboard_hold is not None and abs(current_hold - dashboard_hold) > HOLDING_TIME_TOLERANCE_MINUTES:
        return "DASHBOARD_HOLDING_TIME_MISMATCH"
    if row.get("sqlite_ibkr_quantity_diff") not in (None, "") and abs(fnum(row.get("sqlite_ibkr_quantity_diff")) or 0.0) > QTY_TOLERANCE:
        return "BROKER_SQLITE_MISMATCH"
    if row.get("sqlite_ibkr_entry_price_diff") not in (None, "") and abs(fnum(row.get("sqlite_ibkr_entry_price_diff")) or 0.0) > PRICE_TOLERANCE:
        return "ENTRY_PRICE_MISMATCH"
    if row.get("sqlite_ibkr_exit_price_diff") not in (None, "") and abs(fnum(row.get("sqlite_ibkr_exit_price_diff")) or 0.0) > PRICE_TOLERANCE:
        return "EXIT_PRICE_MISMATCH"
    if row.get("sqlite_ibkr_realized_pnl_diff") not in (None, "") and abs(fnum(row.get("sqlite_ibkr_realized_pnl_diff")) or 0.0) > PNL_TOLERANCE:
        return "PNL_MISMATCH"
    if "missing_trade_id" in problem or "unmatched_sell" in problem or "ambiguous_match" in problem:
        return "UNKNOWN"
    return "OK"


def nearest_sqlite_execution(flex_row: dict[str, Any], sqlite_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    flex_symbol = norm_symbol(flex_row.get("symbol"))
    flex_side = norm_side(flex_row.get("side"))
    flex_time = parse_dt(flex_row.get("time"))
    flex_qty = fnum(flex_row.get("quantity"))
    flex_price = fnum(flex_row.get("price"))
    flex_id = str(flex_row.get("execution_id") or "")
    by_id = [row for row in sqlite_rows if str(row.get("execution_id") or "") == flex_id]
    if by_id:
        row = by_id[0]
        reasons = []
        if norm_symbol(row.get("symbol")) != flex_symbol:
            reasons.append("symbol mismatch")
        if norm_side(row.get("side")) != flex_side:
            reasons.append("buySell mismatch")
        if flex_qty is not None and not close_enough(abs(fnum(row.get("quantity")) or 0.0), abs(flex_qty), QTY_TOLERANCE):
            reasons.append("quantity mismatch")
        if flex_price is not None and not close_enough(fnum(row.get("price")), flex_price, PRICE_TOLERANCE):
            reasons.append("price mismatch")
        return row, "; ".join(reasons) or "exact execution_id matched"
    candidates = sqlite_rows
    if flex_symbol:
        same_symbol = [row for row in candidates if norm_symbol(row.get("symbol")) == flex_symbol]
        if same_symbol:
            candidates = same_symbol
    if flex_side:
        same_side = [row for row in candidates if norm_side(row.get("side")) == flex_side]
        if same_side:
            candidates = same_side
    def score(row: dict[str, Any]) -> float:
        value = 0.0
        if norm_symbol(row.get("symbol")) != flex_symbol:
            value += 1_000_000.0
        if norm_side(row.get("side")) != flex_side:
            value += 100_000.0
        if flex_time is not None:
            row_time = parse_dt(execution_time(row))
            value += abs((row_time - flex_time).total_seconds()) if row_time is not None else 50_000.0
        if flex_qty is not None:
            value += abs(abs(fnum(row.get("quantity")) or 0.0) - abs(flex_qty)) * 100.0
        if flex_price is not None and fnum(row.get("price")) is not None:
            value += abs((fnum(row.get("price")) or 0.0) - flex_price) * 10.0
        return value
    if not candidates:
        return None, "no sqlite executions loaded"
    nearest = min(candidates, key=score)
    reasons = ["execution_id mismatch", "tradeID mismatch"]
    if norm_symbol(nearest.get("symbol")) != flex_symbol:
        reasons.append("symbol mismatch")
    if norm_side(nearest.get("side")) != flex_side:
        reasons.append("buySell mismatch")
    nearest_time = parse_dt(execution_time(nearest))
    if flex_time is None or nearest_time is None:
        reasons.append("timestamp mismatch")
    else:
        delta = abs((nearest_time - flex_time).total_seconds())
        if delta > 300:
            reasons.append("timestamp mismatch")
        elif delta > 0:
            reasons.append("timezone mismatch" if delta in (3600, 7200, 14400, 18000) else "timestamp mismatch")
    if flex_qty is not None and not close_enough(abs(fnum(nearest.get("quantity")) or 0.0), abs(flex_qty), QTY_TOLERANCE):
        reasons.append("quantity mismatch")
    if flex_price is not None and not close_enough(fnum(nearest.get("price")), flex_price, PRICE_TOLERANCE):
        reasons.append("price mismatch")
    return nearest, "; ".join(dict.fromkeys(reasons))


def sqlite_row_identity(row: dict[str, Any]) -> str:
    return str(row.get("execution_id") or "")


def flex_row_identity(row: dict[str, Any], fallback: int) -> str:
    value = str(row.get("execution_id") or "").strip()
    return value or f"flex_row_{fallback}"


def flex_sqlite_candidates(flex_row: dict[str, Any], sqlite_rows: list[dict[str, Any]], used_sqlite: set[str]) -> list[dict[str, Any]]:
    symbol = norm_symbol(flex_row.get("symbol"))
    side = norm_side(flex_row.get("side"))
    date = date_part(flex_row.get("time"))
    price = fnum(flex_row.get("price"))
    out = []
    for row in sqlite_rows:
        sqlite_id = sqlite_row_identity(row)
        if sqlite_id in used_sqlite:
            continue
        if norm_symbol(row.get("symbol")) != symbol:
            continue
        if norm_side(row.get("side")) != side:
            continue
        if execution_date(row) != date:
            continue
        if price is not None and not close_enough(fnum(row.get("price")), price, PRICE_TOLERANCE):
            continue
        out.append(row)
    return out


def assign_match(
    *,
    assignments: dict[int, list[dict[str, Any]]],
    sqlite_to_flex: dict[str, list[dict[str, Any]]],
    statuses: dict[int, str],
    flex_index: int,
    flex_row: dict[str, Any],
    sqlite_rows: list[dict[str, Any]],
    status: str,
    used_sqlite: set[str],
) -> None:
    enriched_rows = []
    for sqlite_row in sqlite_rows:
        enriched = dict(flex_row)
        enriched["flex_trade_id"] = flex_row_identity(flex_row, flex_index)
        enriched["_sqlite_row"] = sqlite_row
        enriched_rows.append(enriched)
        sqlite_id = sqlite_row_identity(sqlite_row)
        if sqlite_id:
            sqlite_to_flex.setdefault(sqlite_id, []).append(enriched)
            used_sqlite.add(sqlite_id)
    assignments[flex_index] = sqlite_rows
    statuses[flex_index] = status


def match_flex_rows_to_sqlite(
    flex: pd.DataFrame,
    sqlite_executions_df: pd.DataFrame,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[int, str]]:
    sqlite_rows = sqlite_executions_df.to_dict("records") if not sqlite_executions_df.empty else []
    sqlite_by_id = {sqlite_row_identity(row): row for row in sqlite_rows if sqlite_row_identity(row)}
    flex_rows = flex.to_dict("records")
    assignments: dict[int, list[dict[str, Any]]] = {}
    sqlite_to_flex: dict[str, list[dict[str, Any]]] = {}
    statuses: dict[int, str] = {}
    used_sqlite: set[str] = set()
    for idx, flex_row in enumerate(flex_rows):
        flex_id = str(flex_row.get("execution_id") or "")
        sqlite_row = sqlite_by_id.get(flex_id)
        if not sqlite_row or sqlite_row_identity(sqlite_row) in used_sqlite:
            continue
        if norm_symbol(sqlite_row.get("symbol")) != norm_symbol(flex_row.get("symbol")):
            continue
        if norm_side(sqlite_row.get("side")) != norm_side(flex_row.get("side")):
            continue
        if execution_date(sqlite_row) != date_part(flex_row.get("time")):
            continue
        if not close_enough(abs(fnum(sqlite_row.get("quantity")) or 0.0), abs(fnum(flex_row.get("quantity")) or 0.0), QTY_TOLERANCE):
            continue
        if not close_enough(fnum(sqlite_row.get("price")), fnum(flex_row.get("price")), PRICE_TOLERANCE):
            continue
        assign_match(
            assignments=assignments,
            sqlite_to_flex=sqlite_to_flex,
            statuses=statuses,
            flex_index=idx,
            flex_row=flex_row,
            sqlite_rows=[sqlite_row],
            status="exact_match",
            used_sqlite=used_sqlite,
        )
    for idx, flex_row in enumerate(flex_rows):
        if idx in assignments:
            continue
        candidates = flex_sqlite_candidates(flex_row, sqlite_rows, used_sqlite)
        flex_qty = abs(fnum(flex_row.get("quantity")) or 0.0)
        exact_qty = [row for row in candidates if close_enough(abs(fnum(row.get("quantity")) or 0.0), flex_qty, QTY_TOLERANCE)]
        if len(exact_qty) == 1:
            assign_match(
                assignments=assignments,
                sqlite_to_flex=sqlite_to_flex,
                statuses=statuses,
                flex_index=idx,
                flex_row=flex_row,
                sqlite_rows=[exact_qty[0]],
                status="exact_match",
                used_sqlite=used_sqlite,
            )
        elif len(exact_qty) > 1:
            statuses[idx] = "ambiguous_match"
    for idx, flex_row in enumerate(flex_rows):
        if idx in assignments or statuses.get(idx) == "ambiguous_match":
            continue
        candidates = flex_sqlite_candidates(flex_row, sqlite_rows, used_sqlite)
        flex_qty = abs(fnum(flex_row.get("quantity")) or 0.0)
        candidate_qty = sum(abs(fnum(row.get("quantity")) or 0.0) for row in candidates)
        if candidates and close_enough(candidate_qty, flex_qty, QTY_TOLERANCE):
            assign_match(
                assignments=assignments,
                sqlite_to_flex=sqlite_to_flex,
                statuses=statuses,
                flex_index=idx,
                flex_row=flex_row,
                sqlite_rows=candidates,
                status="aggregate_match",
                used_sqlite=used_sqlite,
            )
    grouped: dict[tuple[str, str, str, str], list[int]] = {}
    for idx, flex_row in enumerate(flex_rows):
        if idx in assignments or statuses.get(idx) == "ambiguous_match":
            continue
        key = (
            norm_symbol(flex_row.get("symbol")),
            date_part(flex_row.get("time")),
            norm_side(flex_row.get("side")),
            f"{fnum(flex_row.get('price')) or 0.0:.4f}",
        )
        grouped.setdefault(key, []).append(idx)
    for indexes in grouped.values():
        if len(indexes) < 2:
            continue
        sample = flex_rows[indexes[0]]
        candidates = flex_sqlite_candidates(sample, sqlite_rows, used_sqlite)
        if len(candidates) != 1:
            continue
        total_flex_qty = sum(abs(fnum(flex_rows[idx].get("quantity")) or 0.0) for idx in indexes)
        sqlite_qty = abs(fnum(candidates[0].get("quantity")) or 0.0)
        if not close_enough(total_flex_qty, sqlite_qty, QTY_TOLERANCE):
            continue
        for idx in indexes:
            assign_match(
                assignments=assignments,
                sqlite_to_flex=sqlite_to_flex,
                statuses=statuses,
                flex_index=idx,
                flex_row=flex_rows[idx],
                sqlite_rows=candidates,
                status="aggregate_match",
                used_sqlite=used_sqlite,
            )
    for idx in range(len(flex_rows)):
        statuses.setdefault(idx, "unmatched")
    return assignments, sqlite_to_flex, statuses


def build_flex_match_diagnostics(
    *,
    flex_file: Path | None,
    sqlite_executions_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    flex_raw = read_flex_file(flex_file) if flex_file else pd.DataFrame()
    flex_meta = getattr(flex_raw, "attrs", {}).get("flex_meta", {})
    flex = normalize_execution_frame(flex_raw, source="ibkr_flex") if flex_file else pd.DataFrame()
    raw_by_trade_id: dict[str, dict[str, Any]] = {}
    if not flex_raw.empty:
        raw_id_col = first_col(flex_raw.columns, ["tradeID", "execution_id", "execid", "exec_id", "ibexecid"])
        if raw_id_col:
            for raw_row in flex_raw.to_dict("records"):
                raw_id = str(raw_row.get(raw_id_col) or "")
                if raw_id:
                    raw_by_trade_id[raw_id] = raw_row
    if not flex.empty:
        flex["_date"] = flex["time"].map(date_part)
        flex = flex[(flex["_date"] >= start_date) & (flex["_date"] <= end_date)].drop(columns=["_date"])
    sqlite_rows = sqlite_executions_df.to_dict("records") if not sqlite_executions_df.empty else []
    assignments, sqlite_to_flex, statuses = match_flex_rows_to_sqlite(flex, sqlite_executions_df)
    rows: list[dict[str, Any]] = []
    matched_exact = 0
    matched_aggregate = 0
    ambiguous = 0
    unmatched = 0
    matched_buy = 0
    matched_sell = 0
    for idx, flex_row in enumerate(flex.to_dict("records")):
        flex_id = str(flex_row.get("execution_id") or "")
        side = norm_side(flex_row.get("side"))
        status = statuses.get(idx, "unmatched")
        matched = status in {"exact_match", "aggregate_match"}
        if status == "exact_match":
            matched_exact += 1
        elif status == "aggregate_match":
            matched_aggregate += 1
        elif status == "ambiguous_match":
            ambiguous += 1
        elif status == "unmatched":
            unmatched += 1
        if matched and side == "BUY":
            matched_buy += 1
        if matched and side == "SELL":
            matched_sell += 1
        nearest, reason = nearest_sqlite_execution(flex_row, sqlite_rows)
        if matched:
            continue
        raw = raw_by_trade_id.get(flex_id, {})
        rows.append(
            {
                "symbol": norm_symbol(flex_row.get("symbol")),
                "buySell": side,
                "tradeDate": raw.get("tradeDate") or raw.get("reportDate") or date_part(flex_row.get("time")),
                "tradeTime": raw.get("tradeTime") or raw.get("dateTime") or "",
                "quantity": flex_row.get("quantity"),
                "price": flex_row.get("price"),
                "tradeID": flex_id,
                "match_status": status,
                "match_failure_reason": reason,
                "nearest_sqlite_execution_id": str(nearest.get("execution_id") or "") if nearest else "",
                "nearest_sqlite_symbol": norm_symbol(nearest.get("symbol")) if nearest else "",
                "nearest_sqlite_side": norm_side(nearest.get("side")) if nearest else "",
                "nearest_sqlite_time": execution_time(nearest) if nearest else "",
                "nearest_sqlite_quantity": nearest.get("quantity") if nearest else None,
                "nearest_sqlite_price": nearest.get("price") if nearest else None,
            }
        )
    summary = {
        "flex_source": flex_meta.get("source", ""),
        "flex_trades_loaded": int(len(flex)),
        "sqlite_executions_loaded": int(len(sqlite_rows)),
        "flex_buy_matched": int(matched_buy),
        "flex_sell_matched": int(matched_sell),
        "flex_rows_matched_exact": int(matched_exact),
        "flex_rows_matched_aggregate": int(matched_aggregate),
        "flex_rows_ambiguous": int(ambiguous),
        "flex_rows_unmatched": int(unmatched),
        "flex_unmatched": int(unmatched),
    }
    return pd.DataFrame(rows), summary, sqlite_to_flex


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.to_dict("records"):
        values = []
        for col in cols:
            value = row.get(col)
            text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def flex_file_metadata(path: Path | None) -> dict[str, Any]:
    return getattr(read_flex_file(path), "attrs", {}).get(
        "flex_meta",
        {"source": "missing", "statements_count": 0, "trade_rows": 0, "symbols": 0, "date_min": "", "date_max": ""},
    )


def flex_covers_requested_range(meta: dict[str, Any], start_date: str, end_date: str) -> bool:
    flex_start = str(meta.get("date_min") or "")
    flex_end = str(meta.get("date_max") or "")
    if not flex_start or not flex_end:
        return False
    return flex_start <= start_date and flex_end >= end_date


def print_flex_date_range_mismatch(start_date: str, end_date: str, meta: dict[str, Any]) -> None:
    print("FLEX_DATE_RANGE_MISMATCH", flush=True)
    print(f"requested_start={start_date}", flush=True)
    print(f"requested_end={end_date}", flush=True)
    print(f"flex_start={meta.get('date_min') or ''}", flush=True)
    print(f"flex_end={meta.get('date_max') or ''}", flush=True)


def group_by_symbol_sell_date(trades: list[AuditTrade]) -> dict[tuple[str, str], list[AuditTrade]]:
    out: dict[tuple[str, str], list[AuditTrade]] = {}
    for trade in trades:
        key = (trade.symbol, date_part(trade.sell_time))
        out.setdefault(key, []).append(trade)
    for key in list(out):
        out[key].sort(key=lambda item: (item.sell_time, item.buy_time, item.trade_id))
    return out


def pick(source: dict[tuple[str, str], list[AuditTrade]], key: tuple[str, str], index: int) -> AuditTrade | None:
    values = source.get(key) or []
    return values[index] if index < len(values) else None


def close_enough(a: float | None, b: float | None, tolerance: float) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance


def classify(row: dict[str, Any]) -> str:
    if not row.get("broker_buy_time") and not row.get("sqlite_buy_time") and not row.get("dashboard_entry_time"):
        return "UNKNOWN"
    if row.get("broker_buy_time") and row.get("sqlite_buy_time"):
        if date_part(row["broker_buy_time"]) != date_part(row["sqlite_buy_time"]) or date_part(row["broker_sell_time"]) != date_part(row["sqlite_sell_time"]):
            return "BROKER_SQLITE_MISMATCH"
        if not close_enough(fnum(row.get("broker_qty")), fnum(row.get("sqlite_qty")), QTY_TOLERANCE):
            return "BROKER_SQLITE_MISMATCH"
    if row.get("sqlite_buy_time") and row.get("reconstructed_buy_time"):
        if date_part(row["sqlite_buy_time"]) != date_part(row["reconstructed_buy_time"]) or date_part(row["sqlite_sell_time"]) != date_part(row["reconstructed_sell_time"]):
            return "SQLITE_FIFO_MISMATCH"
        if not close_enough(fnum(row.get("sqlite_qty")), fnum(row.get("reconstructed_qty")), QTY_TOLERANCE):
            return "SQLITE_FIFO_MISMATCH"
    if row.get("reconstructed_buy_time") and row.get("dashboard_entry_time"):
        if date_part(row["reconstructed_buy_time"]) != date_part(row["dashboard_entry_time"]) or date_part(row["reconstructed_sell_time"]) != date_part(row["dashboard_exit_time"]):
            return "FIFO_DASHBOARD_MISMATCH"
    broker_hold = fnum(row.get("holding_minutes_broker"))
    dashboard_hold = fnum(row.get("holding_minutes_dashboard"))
    if broker_hold is not None and dashboard_hold is not None and abs(broker_hold - dashboard_hold) > HOLDING_TIME_TOLERANCE_MINUTES:
        return "DASHBOARD_HOLDING_TIME_MISMATCH"
    if row.get("broker_avg_buy") not in (None, "") and row.get("dashboard_entry_price") not in (None, ""):
        if not close_enough(fnum(row.get("broker_avg_buy")), fnum(row.get("dashboard_entry_price")), PRICE_TOLERANCE):
            return "ENTRY_PRICE_MISMATCH"
    if row.get("broker_avg_sell") not in (None, "") and row.get("dashboard_exit_price") not in (None, ""):
        if not close_enough(fnum(row.get("broker_avg_sell")), fnum(row.get("dashboard_exit_price")), PRICE_TOLERANCE):
            return "EXIT_PRICE_MISMATCH"
    if row.get("broker_realized_pnl") not in (None, "") and row.get("dashboard_realized_pnl") not in (None, ""):
        if not close_enough(fnum(row.get("broker_realized_pnl")), fnum(row.get("dashboard_realized_pnl")), PNL_TOLERANCE):
            return "PNL_MISMATCH"
    return "OK"


def mismatch_score(row: dict[str, Any]) -> float:
    score = 0.0
    broker_hold = fnum(row.get("holding_minutes_broker"))
    dashboard_hold = fnum(row.get("holding_minutes_dashboard"))
    if broker_hold is not None and dashboard_hold is not None:
        score += abs(dashboard_hold - broker_hold)
    for left, right in [
        ("broker_avg_buy", "dashboard_entry_price"),
        ("broker_avg_sell", "dashboard_exit_price"),
        ("broker_realized_pnl", "dashboard_realized_pnl"),
        ("sqlite_avg_buy", "dashboard_entry_price"),
        ("sqlite_avg_sell", "dashboard_exit_price"),
    ]:
        a = fnum(row.get(left))
        b = fnum(row.get(right))
        if a is not None and b is not None:
            score += abs(a - b)
    return score


def audit(
    *,
    start_date: str,
    end_date: str,
    sqlite_path: Path,
    flex_file: Path | None,
    history_dir: Path | None,
    output_dir: Path,
    include_dashboard_reconstructed: bool,
) -> tuple[pd.DataFrame, Path, Path, Path | None]:
    flex_raw = read_flex_file(flex_file) if flex_file else pd.DataFrame()
    flex_meta = getattr(flex_raw, "attrs", {}).get("flex_meta", {"source": "not_provided", "statements_count": 0, "trade_rows": 0, "symbols": 0, "date_min": "", "date_max": ""})
    broker_df = normalize_execution_frame(flex_raw, source="ibkr_flex") if flex_file else pd.DataFrame()
    if not broker_df.empty:
        broker_df["_date"] = broker_df["time"].map(date_part)
        broker_df = broker_df[(broker_df["_date"] >= start_date) & (broker_df["_date"] <= end_date)].drop(columns=["_date"])
    broker_trades = fifo_from_executions(broker_df, selected_start=start_date, selected_end=end_date, source="ibkr_flex")
    sqlite_trades: list[AuditTrade] = []
    fill_trades: list[AuditTrade] = []
    recon_trades: list[AuditTrade] = []
    dashboard_trades: list[AuditTrade] = []
    detailed_df = pd.DataFrame()
    flex_diag_df = pd.DataFrame()
    flex_diag_summary = {
        "flex_trades_loaded": 0,
        "sqlite_executions_loaded": 0,
        "flex_buy_matched": 0,
        "flex_sell_matched": 0,
        "flex_rows_matched_exact": 0,
        "flex_rows_matched_aggregate": 0,
        "flex_rows_ambiguous": 0,
        "flex_rows_unmatched": 0,
        "flex_unmatched": 0,
    }
    if sqlite_path.exists():
        with connect_sqlite(sqlite_path) as conn:
            sqlite_range = sqlite_all_executions_range(conn, start_date, end_date)
            flex_diag_df, flex_diag_summary, sqlite_flex_matches = build_flex_match_diagnostics(
                flex_file=flex_file,
                sqlite_executions_df=sqlite_range,
                start_date=start_date,
                end_date=end_date,
            )
            detailed_df = build_sell_level_audit(
                conn=conn,
                sqlite_path=sqlite_path,
                start_date=start_date,
                end_date=end_date,
                flex_file=flex_file,
                sqlite_flex_matches=sqlite_flex_matches,
                history_dir=history_dir,
                include_dashboard_reconstructed=include_dashboard_reconstructed,
            )
            sqlite_trades = fifo_from_executions(sqlite_executions(conn, start_date, end_date), selected_start=start_date, selected_end=end_date, source="sqlite_executions")
            fill_trades = fifo_from_executions(sqlite_fills(conn, start_date, end_date), selected_start=start_date, selected_end=end_date, source="sqlite_fills")
            recon_trades = reconstructed_trades(conn, start_date, end_date)
        if detailed_df.empty:
            dashboard_trades = dashboard_closed_positions(sqlite_path, start_date, end_date, include_dashboard_reconstructed)
    sources = {
        "broker": group_by_symbol_sell_date(broker_trades),
        "sqlite": group_by_symbol_sell_date(sqlite_trades),
        "fills": group_by_symbol_sell_date(fill_trades),
        "reconstructed": group_by_symbol_sell_date(recon_trades),
        "dashboard": group_by_symbol_sell_date(dashboard_trades),
    }
    keys = sorted(set().union(*(source.keys() for source in sources.values())))
    rows: list[dict[str, Any]] = []
    for key in keys:
        max_len = max(len(source.get(key) or []) for source in sources.values())
        for idx in range(max_len):
            broker = pick(sources["broker"], key, idx)
            sqlite = pick(sources["sqlite"], key, idx)
            fills = pick(sources["fills"], key, idx)
            recon = pick(sources["reconstructed"], key, idx)
            dash = pick(sources["dashboard"], key, idx)
            row = {
                "symbol": key[0],
                "broker_buy_time": broker.buy_time if broker else "",
                "broker_sell_time": broker.sell_time if broker else "",
                "sqlite_buy_time": sqlite.buy_time if sqlite else "",
                "sqlite_sell_time": sqlite.sell_time if sqlite else "",
                "sqlite_fill_buy_time": fills.buy_time if fills else "",
                "sqlite_fill_sell_time": fills.sell_time if fills else "",
                "reconstructed_buy_time": recon.buy_time if recon else "",
                "reconstructed_sell_time": recon.sell_time if recon else "",
                "dashboard_entry_time": dash.buy_time if dash else "",
                "dashboard_exit_time": dash.sell_time if dash else "",
                "broker_qty": broker.qty if broker else None,
                "sqlite_qty": sqlite.qty if sqlite else None,
                "sqlite_fill_qty": fills.qty if fills else None,
                "reconstructed_qty": recon.qty if recon else None,
                "broker_avg_buy": broker.avg_buy if broker else None,
                "broker_avg_sell": broker.avg_sell if broker else None,
                "sqlite_avg_buy": sqlite.avg_buy if sqlite else None,
                "sqlite_avg_sell": sqlite.avg_sell if sqlite else None,
                "sqlite_fill_avg_buy": fills.avg_buy if fills else None,
                "sqlite_fill_avg_sell": fills.avg_sell if fills else None,
                "dashboard_entry_price": dash.avg_buy if dash else None,
                "dashboard_exit_price": dash.avg_sell if dash else None,
                "broker_realized_pnl": broker.realized_pnl if broker else None,
                "sqlite_realized_pnl": sqlite.realized_pnl if sqlite else None,
                "sqlite_fill_realized_pnl": fills.realized_pnl if fills else None,
                "dashboard_realized_pnl": dash.realized_pnl if dash else None,
                "holding_minutes_broker": holding_minutes(broker.buy_time, broker.sell_time) if broker else None,
                "holding_minutes_dashboard": holding_minutes(dash.buy_time, dash.sell_time) if dash else None,
                "exit_reason_dashboard": dash.exit_reason if dash else "",
                "broker_trade_id": broker.trade_id if broker else "",
                "sqlite_trade_id": sqlite.trade_id if sqlite else "",
                "sqlite_fill_trade_id": fills.trade_id if fills else "",
                "reconstructed_trade_id": recon.trade_id if recon else "",
                "dashboard_trade_id": dash.trade_id if dash else "",
                "broker_buy_execution_ids": broker.buy_execution_ids if broker else "",
                "broker_sell_execution_ids": broker.sell_execution_ids if broker else "",
                "sqlite_buy_execution_ids": sqlite.buy_execution_ids if sqlite else "",
                "sqlite_sell_execution_ids": sqlite.sell_execution_ids if sqlite else "",
                "sqlite_fill_buy_execution_ids": fills.buy_execution_ids if fills else "",
                "sqlite_fill_sell_execution_ids": fills.sell_execution_ids if fills else "",
                "reconstructed_buy_execution_ids": recon.buy_execution_ids if recon else "",
                "reconstructed_sell_execution_ids": recon.sell_execution_ids if recon else "",
            }
            row["status"] = classify(row)
            row["mismatch_score"] = mismatch_score(row)
            rows.append(row)
    df = detailed_df if not detailed_df.empty else pd.DataFrame(rows)
    if not df.empty:
        status_rank = {status: idx for idx, status in enumerate(STATUS_ORDER)}
        df["_status_rank"] = df["status"].map(lambda value: status_rank.get(str(value), 99))
        df = df.sort_values(["_status_rank", "mismatch_score", "symbol"], ascending=[True, False, True]).drop(columns=["_status_rank"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = start_date if start_date == end_date else f"{start_date}_{end_date}"
    csv_path = output_dir / f"trade_reconstruction_audit_{suffix}.csv"
    summary_path = output_dir / f"trade_reconstruction_summary_{suffix}.md"
    flex_diag_path = output_dir / f"trade_reconstruction_flex_match_diagnostics_{suffix}.csv"
    df.to_csv(csv_path, index=False)
    if not flex_diag_df.empty:
        flex_diag_df.to_csv(flex_diag_path, index=False)
    else:
        flex_diag_path = None
    write_summary(summary_path, df, start_date=start_date, end_date=end_date, flex_file=flex_file, sqlite_path=sqlite_path, flex_meta=flex_meta, flex_diag_summary=flex_diag_summary, flex_diag_df=flex_diag_df)
    generic_csv_path = output_dir / "trade_reconstruction_audit.csv"
    generic_summary_path = output_dir / "trade_reconstruction_summary.md"
    if generic_csv_path != csv_path:
        df.to_csv(generic_csv_path, index=False)
    if generic_summary_path != summary_path:
        write_summary(generic_summary_path, df, start_date=start_date, end_date=end_date, flex_file=flex_file, sqlite_path=sqlite_path, flex_meta=flex_meta, flex_diag_summary=flex_diag_summary, flex_diag_df=flex_diag_df)
    if not flex_diag_df.empty:
        flex_diag_df.to_csv(output_dir / "trade_reconstruction_flex_match_diagnostics.csv", index=False)
    return df, csv_path, summary_path, flex_diag_path


def write_summary(
    path: Path,
    df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    flex_file: Path | None,
    sqlite_path: Path,
    flex_meta: dict[str, Any] | None = None,
    flex_diag_summary: dict[str, Any] | None = None,
    flex_diag_df: pd.DataFrame | None = None,
) -> None:
    status_counts = df["status"].value_counts().to_dict() if not df.empty and "status" in df.columns else {}
    top = df.sort_values("mismatch_score", ascending=False).head(100) if not df.empty else pd.DataFrame()
    code_locations = [
        "src/live_trading/storage/sqlite_store.py: reconstructed_trade_id(), SQLiteRuntimeStore.rebuild_symbol_trade_state(), _clear_reconstructed_trades_for_symbol()",
        "src/dashboard/runtime_queries.py: closed_from_executions(), closed_from_execution_realized_pnl(), load_closed_positions(), aggregate_closed_positions()",
        "src/dashboard/runtime_dashboard.py: render_closed_positions(), Runtime tab snapshot rendering",
        "scripts/diagnose_reconstructed_fifo.py: proposed_same_session_pairs() diagnostic reference",
        "scripts/backfill_closed_trades_from_executions.py: reconstruct_closed_trades_from_executions() historical repair path",
    ]
    lines = [
        "# Trade Reconstruction Audit",
        "",
        f"- start_date: {start_date}",
        f"- end_date: {end_date}",
        f"- sqlite_path: `{sqlite_path}`",
        f"- ibkr_flex_file: `{flex_file}`" if flex_file else "- ibkr_flex_file: not provided",
        f"- rows: {len(df)}",
        f"- flex_source: {(flex_meta or {}).get('source', '')}",
        f"- flex_statements: {(flex_meta or {}).get('statements_count', 0)}",
        f"- flex_trade_rows: {(flex_meta or {}).get('trade_rows', 0)}",
        f"- flex_date_range: {(flex_meta or {}).get('date_min', '')}..{(flex_meta or {}).get('date_max', '')}",
        f"- flex_symbols: {(flex_meta or {}).get('symbols', 0)}",
        "",
        "## Flex / SQLite Matching Diagnostics",
        "",
        f"- Flex trades loaded: {(flex_diag_summary or {}).get('flex_trades_loaded', 0)}",
        f"- SQLite executions loaded: {(flex_diag_summary or {}).get('sqlite_executions_loaded', 0)}",
        f"- Flex BUY matched: {(flex_diag_summary or {}).get('flex_buy_matched', 0)}",
        f"- Flex SELL matched: {(flex_diag_summary or {}).get('flex_sell_matched', 0)}",
        f"- Flex rows matched exact: {(flex_diag_summary or {}).get('flex_rows_matched_exact', 0)}",
        f"- Flex rows matched aggregate: {(flex_diag_summary or {}).get('flex_rows_matched_aggregate', 0)}",
        f"- Flex rows ambiguous: {(flex_diag_summary or {}).get('flex_rows_ambiguous', 0)}",
        f"- Flex rows unmatched: {(flex_diag_summary or {}).get('flex_rows_unmatched', 0)}",
        f"- Flex unmatched: {(flex_diag_summary or {}).get('flex_unmatched', 0)}",
        "",
        "### First 20 Flex Unmatched",
        "",
    ]
    if flex_diag_df is not None and not flex_diag_df.empty:
        first_unmatched_cols = [
            "symbol",
            "buySell",
            "tradeDate",
            "tradeTime",
            "quantity",
            "price",
            "tradeID",
            "match_failure_reason",
            "nearest_sqlite_execution_id",
            "nearest_sqlite_symbol",
            "nearest_sqlite_side",
            "nearest_sqlite_time",
            "nearest_sqlite_quantity",
            "nearest_sqlite_price",
        ]
        lines.append(markdown_table(flex_diag_df.head(20)[[col for col in first_unmatched_cols if col in flex_diag_df.columns]]))
    else:
        lines.append("_No unmatched Flex rows._")
    lines.extend([
        "",
        "## Status Counts",
        "",
    ])
    if status_counts:
        for status, count in sorted(status_counts.items(), key=lambda item: STATUS_ORDER.index(item[0]) if item[0] in STATUS_ORDER else 99):
            lines.append(f"- {status}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Top 100 Largest Divergences", ""])
    if top.empty:
        lines.append("_No rows._")
    else:
        display_cols = [
            "sell_execution_id",
            "symbol",
            "status",
            "problem_classification",
            "mismatch_score",
            "sell_time",
            "sell_quantity",
            "canonical_trade_id",
            "component_count",
            "component_buy_execution_ids",
            "component_buy_times",
            "component_matched_qty",
            "component_total_matched_qty",
            "sell_quantity_conservation_diff",
            "current_executions_trade_id",
            "entry_execution_id",
            "entry_time",
            "candidate_entry_execution_id",
            "candidate_entry_time",
            "holding_minutes_current_assignment",
            "holding_minutes_same_session_candidate",
            "dashboard_closed_source",
            "peak_source",
            "peak_match_status",
            "broker_buy_time",
            "broker_sell_time",
            "dashboard_entry_time",
            "dashboard_exit_time",
            "holding_minutes_broker",
            "holding_minutes_dashboard",
            "broker_realized_pnl",
            "dashboard_realized_pnl",
        ]
        lines.append(markdown_table(top[[col for col in display_cols if col in top.columns]]))
    cross_symbols = {"MXL", "NBIS", "WDAY", "AMPG", "CARL", "PENG", "POET"}
    lines.extend(["", "## Cross-Session Component Details", ""])
    if df.empty or "symbol" not in df.columns:
        lines.append("_No rows._")
    else:
        component_cols = [
            "symbol",
            "sell_execution_id",
            "sell_time",
            "sell_quantity",
            "canonical_trade_id",
            "component_count",
            "component_buy_execution_ids",
            "component_buy_times",
            "component_matched_qty",
            "component_total_matched_qty",
            "status",
        ]
        cross = df[df["symbol"].isin(cross_symbols)][[col for col in component_cols if col in df.columns]]
        lines.append(markdown_table(cross) if not cross.empty else "_No cross-session target rows._")
    lines.extend(["", "## Code Locations Responsible For Reconstruction / Closed Positions", ""])
    for item in code_locations:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit broker/SQLite/FIFO/dashboard trade reconstruction without mutating data.")
    parser.add_argument("--date", help="Single selected session date.")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--ibkr-flex-file", help="Optional IBKR Flex Query XML or Activity/Flex CSV for the same period.")
    parser.add_argument("--ibkr-flex-csv", help="Backward-compatible alias for --ibkr-flex-file.")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_SQLITE_PATH))
    parser.add_argument("--history-dir", default="data/history/universe_1m", help="Optional parquet history dir for candle-reconstructed peak diagnostics.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include-dashboard-reconstructed", action="store_true", help="Ask dashboard snapshot to include execution-reconstructed rows.")
    args = parser.parse_args()
    start_date = args.start_date or args.date
    end_date = args.end_date or args.date
    if not start_date or not end_date:
        parser.error("provide --date or --start-date/--end-date")
    flex_file = args.ibkr_flex_file or args.ibkr_flex_csv
    if flex_file:
        flex_meta = flex_file_metadata(Path(flex_file))
        if not flex_covers_requested_range(flex_meta, start_date, end_date):
            print_flex_date_range_mismatch(start_date, end_date, flex_meta)
            return 2
    df, csv_path, summary_path, flex_diag_path = audit(
        start_date=start_date,
        end_date=end_date,
        sqlite_path=Path(args.sqlite_path),
        flex_file=Path(flex_file) if flex_file else None,
        history_dir=Path(args.history_dir) if args.history_dir else None,
        output_dir=Path(args.output_dir),
        include_dashboard_reconstructed=bool(args.include_dashboard_reconstructed),
    )
    print(
        f"TRADE_RECONSTRUCTION_AUDIT_DONE rows={len(df)} output={csv_path} summary={summary_path}",
        flush=True,
    )
    if flex_diag_path:
        print(f"FLEX_MATCH_DIAGNOSTICS output={flex_diag_path}", flush=True)
    if not df.empty and "status" in df.columns:
        print("status_counts=" + json.dumps(df["status"].value_counts().to_dict(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
