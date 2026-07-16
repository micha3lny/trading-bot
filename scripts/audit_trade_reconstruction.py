#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
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
    "BROKER_SQLITE_MISMATCH",
    "SQLITE_FIFO_MISMATCH",
    "FIFO_DASHBOARD_MISMATCH",
    "DASHBOARD_HOLDING_TIME_MISMATCH",
    "ENTRY_PRICE_MISMATCH",
    "EXIT_PRICE_MISMATCH",
    "PNL_MISMATCH",
    "UNKNOWN",
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


def normalize_execution_frame(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "side", "quantity", "price", "time", "execution_id", "order_id", "realized_pnl", "commission", "raw_source"])
    cols = list(df.columns)
    symbol_col = first_col(cols, ["symbol", "underlying", "contract", "ticker"])
    side_col = first_col(cols, ["side", "action", "buy/sell", "transactiontype", "transaction_type"])
    qty_col = first_col(cols, ["quantity", "qty", "shares", "totalquantity"])
    price_col = first_col(cols, ["price", "fill_price", "tradeprice", "trade_price", "avgprice"])
    time_col = first_col(cols, ["executed_at", "execution_time", "datetime", "date/time", "date_time", "time", "trade_time"])
    exec_col = first_col(cols, ["execution_id", "execid", "exec_id", "ibexecid"])
    order_col = first_col(cols, ["order_id", "orderid", "order"])
    pnl_col = first_col(cols, ["realized_pnl", "realizedpnl", "realized p/l", "realized p&l", "realizedpl"])
    comm_col = first_col(cols, ["commission", "commissions"])
    out = pd.DataFrame()
    out["symbol"] = df[symbol_col].map(norm_symbol) if symbol_col else ""
    out["side"] = df[side_col].map(norm_side) if side_col else ""
    out["quantity"] = pd.to_numeric(df[qty_col], errors="coerce") if qty_col else pd.NA
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
    snapshot = load_dashboard_snapshot(sqlite_path, DateWindow(start_date, end_date), "All", include_reconstructed=include_reconstructed)
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
    flex_csv: Path | None,
    output_dir: Path,
    include_dashboard_reconstructed: bool,
) -> tuple[pd.DataFrame, Path, Path]:
    broker_df = normalize_execution_frame(read_csv_flexible(flex_csv), source="ibkr_flex") if flex_csv else pd.DataFrame()
    if not broker_df.empty:
        broker_df["_date"] = broker_df["time"].map(date_part)
        broker_df = broker_df[(broker_df["_date"] >= start_date) & (broker_df["_date"] <= end_date)].drop(columns=["_date"])
    broker_trades = fifo_from_executions(broker_df, selected_start=start_date, selected_end=end_date, source="ibkr_flex")
    sqlite_trades: list[AuditTrade] = []
    fill_trades: list[AuditTrade] = []
    recon_trades: list[AuditTrade] = []
    dashboard_trades: list[AuditTrade] = []
    if sqlite_path.exists():
        with connect_sqlite(sqlite_path) as conn:
            sqlite_trades = fifo_from_executions(sqlite_executions(conn, start_date, end_date), selected_start=start_date, selected_end=end_date, source="sqlite_executions")
            fill_trades = fifo_from_executions(sqlite_fills(conn, start_date, end_date), selected_start=start_date, selected_end=end_date, source="sqlite_fills")
            recon_trades = reconstructed_trades(conn, start_date, end_date)
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
    df = pd.DataFrame(rows)
    if not df.empty:
        status_rank = {status: idx for idx, status in enumerate(STATUS_ORDER)}
        df["_status_rank"] = df["status"].map(lambda value: status_rank.get(str(value), 99))
        df = df.sort_values(["_status_rank", "mismatch_score", "symbol"], ascending=[True, False, True]).drop(columns=["_status_rank"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = start_date if start_date == end_date else f"{start_date}_{end_date}"
    csv_path = output_dir / f"trade_reconstruction_audit_{suffix}.csv"
    summary_path = output_dir / f"trade_reconstruction_summary_{suffix}.md"
    df.to_csv(csv_path, index=False)
    write_summary(summary_path, df, start_date=start_date, end_date=end_date, flex_csv=flex_csv, sqlite_path=sqlite_path)
    generic_csv_path = output_dir / "trade_reconstruction_audit.csv"
    generic_summary_path = output_dir / "trade_reconstruction_summary.md"
    if generic_csv_path != csv_path:
        df.to_csv(generic_csv_path, index=False)
    if generic_summary_path != summary_path:
        write_summary(generic_summary_path, df, start_date=start_date, end_date=end_date, flex_csv=flex_csv, sqlite_path=sqlite_path)
    return df, csv_path, summary_path


def write_summary(path: Path, df: pd.DataFrame, *, start_date: str, end_date: str, flex_csv: Path | None, sqlite_path: Path) -> None:
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
        f"- ibkr_flex_csv: `{flex_csv}`" if flex_csv else "- ibkr_flex_csv: not provided",
        f"- rows: {len(df)}",
        "",
        "## Status Counts",
        "",
    ]
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
            "symbol",
            "status",
            "mismatch_score",
            "broker_buy_time",
            "broker_sell_time",
            "dashboard_entry_time",
            "dashboard_exit_time",
            "holding_minutes_broker",
            "holding_minutes_dashboard",
            "broker_realized_pnl",
            "dashboard_realized_pnl",
        ]
        lines.append(top[[col for col in display_cols if col in top.columns]].to_markdown(index=False))
    lines.extend(["", "## Code Locations Responsible For Reconstruction / Closed Positions", ""])
    for item in code_locations:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit broker/SQLite/FIFO/dashboard trade reconstruction without mutating data.")
    parser.add_argument("--date", help="Single selected session date.")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--ibkr-flex-csv", help="Optional IBKR Flex/Activity execution CSV for the same period.")
    parser.add_argument("--sqlite-path", default=str(DEFAULT_SQLITE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include-dashboard-reconstructed", action="store_true", help="Ask dashboard snapshot to include execution-reconstructed rows.")
    args = parser.parse_args()
    start_date = args.start_date or args.date
    end_date = args.end_date or args.date
    if not start_date or not end_date:
        parser.error("provide --date or --start-date/--end-date")
    df, csv_path, summary_path = audit(
        start_date=start_date,
        end_date=end_date,
        sqlite_path=Path(args.sqlite_path),
        flex_csv=Path(args.ibkr_flex_csv) if args.ibkr_flex_csv else None,
        output_dir=Path(args.output_dir),
        include_dashboard_reconstructed=bool(args.include_dashboard_reconstructed),
    )
    print(
        f"TRADE_RECONSTRUCTION_AUDIT_DONE rows={len(df)} output={csv_path} summary={summary_path}",
        flush=True,
    )
    if not df.empty and "status" in df.columns:
        print("status_counts=" + json.dumps(df["status"].value_counts().to_dict(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
