from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analytics.v67_daily_report import (
    reconstruct_closed_trades_from_fills,
    simulate_exit_strategies,
)
from src.live_trading.storage.sqlite_store import DEFAULT_SQLITE_PATH, connect_sqlite, migrate_runtime_schema, resolve_sqlite_path


CLOSED_STATUSES = {"CLOSED"}
PENDING_TRADE_STATUSES = {"COMMISSION_PENDING", "PNL_PENDING"}
TERMINAL_POSITION_STATUSES = {"CLOSED", "FLAT", "FLAT_CONFIRMED", "ENTRY_REJECTED", "ENTRY_NOT_FILLED", "STALE_DUPLICATE_SUPPRESSED", "ORPHAN_STALE_POSITION"}
OPEN_POSITION_STATUSES = {"OPEN", "EXIT_ORDER"}
DEFAULT_RECORDER_ROOT = Path("data/live/recorder")
DEFAULT_ORPHAN_STALE_DAYS = 7
_MIGRATED_SQLITE_PATHS: set[str] = set()
OPEN_POSITION_STATUS_SQL = """
(
    UPPER(COALESCE(p.status, '')) IN ('OPEN', 'EXIT_ORDER')
    OR UPPER(COALESCE(p.status, '')) LIKE 'OPEN|%'
    OR UPPER(COALESCE(p.status, '')) LIKE 'EXIT_ORDER|%'
)
AND UPPER(COALESCE(p.status, '')) NOT LIKE '%ORPHAN_STALE_POSITION%'
"""


@dataclass(frozen=True)
class DateWindow:
    start_date: str
    end_date: str


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%F")


def connect(sqlite_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(resolve_sqlite_path(sqlite_path or DEFAULT_SQLITE_PATH))
    key = str(path)
    if key not in _MIGRATED_SQLITE_PATHS:
        migrate_runtime_schema(path)
        _MIGRATED_SQLITE_PATHS.add(key)
    return connect_sqlite(path, read_only=True)


def read_sql(conn: sqlite3.Connection, sql: str, params: list[Any] | tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def parse_raw_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def raw_trade_execution_id(raw: dict[str, Any], side: str) -> str:
    if side == "entry":
        keys = ("buy_execution_id", "entry_execution_id", "bot_execution_id")
    else:
        keys = ("sell_execution_id", "exit_execution_id", "sld_execution_id")
    for key in keys:
        value = raw.get(key)
        if value:
            return str(value)
    return ""


def normalized_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null", "0.0"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def raw_bool(raw: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "y", "on"}:
            return True
    return False


def append_quality(existing: Any, *flags: str) -> str:
    parts = [x.strip() for x in str(existing or "OK").split(";") if x.strip() and x.strip() != "OK"]
    for flag in flags:
        if flag and flag not in parts:
            parts.append(flag)
    return ";".join(parts) if parts else "OK"


def raw_json_peak_value(raw: dict[str, Any]) -> float | None:
    for key in ("mfe_pct", "peak_pct", "peak_unrealized_pct", "peak_gain_pct", "max_gain_pct"):
        value = to_float(raw.get(key), None)
        if value is not None:
            return value
    entry_price = to_float(raw.get("entry_price") or raw.get("buy") or raw.get("buy_price"), None)
    peak_price = to_float(raw.get("peak_price") or raw.get("high_watermark") or raw.get("mfe_price"), None)
    if entry_price and peak_price is not None:
        return ((peak_price / entry_price) - 1.0) * 100.0
    return None


def raw_price_value(raw: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = to_float(raw.get(key), None)
        if value is not None:
            return value
    return None


def raw_execution_time_value(raw_value: Any) -> Any:
    raw = parse_raw_json(raw_value)
    execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
    if execution:
        return execution.get("time") or execution.get("executionTime") or execution.get("executed_at")
    return raw.get("executed_at") or raw.get("execution_time") or raw.get("time")


def pct_from_prices(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return ((numerator / denominator) - 1.0) * 100.0


def to_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value in (None, ""):
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def parse_dt(value: Any) -> datetime | None:
    try:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def date_part(value: Any) -> str:
    dt = parse_dt(value)
    if dt is not None:
        return dt.strftime("%F")
    text = str(value or "")
    return text[:10] if len(text) >= 10 else ""


def window_contains_date(window: DateWindow, value: Any) -> bool:
    day = date_part(value)
    return bool(day and window.start_date <= day <= window.end_date)


def expanded_lookup_window(window: DateWindow, days_back: int = 45) -> DateWindow:
    start = datetime.fromisoformat(window.start_date).date() - timedelta(days=days_back)
    return DateWindow(start.isoformat(), window.end_date)


def is_missing_value(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def hold_minutes(start: Any, end: Any = None) -> float:
    start_dt = parse_dt(start)
    end_dt = parse_dt(end) or datetime.now(timezone.utc)
    if start_dt is None:
        return 0.0
    return max(0.0, (end_dt - start_dt).total_seconds() / 60.0)


def age_days(start: Any, end: Any = None) -> float | None:
    start_dt = parse_dt(start)
    end_dt = parse_dt(end) or datetime.now(timezone.utc)
    if start_dt is None:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds() / 86400.0)


def adopted_timestamp_value(value: Any) -> str | None:
    text = str(value or "")
    if text.startswith("adopted_on_restart:"):
        return text.split(":", 1)[1]
    return None


def displayable_entry_time(value: Any) -> Any:
    adopted = adopted_timestamp_value(value)
    return adopted or value


def list_sessions(sqlite_path: str | Path | None = None) -> list[str]:
    if not Path(resolve_sqlite_path(sqlite_path)).exists():
        return []
    conn = connect(sqlite_path)
    try:
        rows = read_sql(
            conn,
            """
            SELECT DISTINCT session_date FROM (
                SELECT session_date FROM executions WHERE session_date IS NOT NULL
                UNION ALL SELECT session_date FROM trades WHERE session_date IS NOT NULL
                UNION ALL SELECT substr(exit_fill_time, 1, 10) AS session_date FROM trades WHERE exit_fill_time IS NOT NULL
                UNION ALL SELECT substr(closed_at, 1, 10) AS session_date FROM trades WHERE closed_at IS NOT NULL
                UNION ALL SELECT session_date FROM positions WHERE session_date IS NOT NULL
                UNION ALL SELECT session_date FROM risk_events WHERE session_date IS NOT NULL
                UNION ALL SELECT session_date FROM runtime_events WHERE session_date IS NOT NULL
            )
            ORDER BY session_date DESC
            """,
        )
        return [str(x) for x in rows["session_date"].dropna().tolist()]
    finally:
        conn.close()


def list_strategies(sqlite_path: str | Path | None, window: DateWindow) -> list[str]:
    if not Path(resolve_sqlite_path(sqlite_path)).exists():
        return []
    conn = connect(sqlite_path)
    try:
        rows = read_sql(
            conn,
            """
            SELECT DISTINCT strategy_name FROM (
                SELECT strategy_name, session_date FROM executions
                UNION ALL SELECT strategy_name, session_date FROM trades
                UNION ALL SELECT strategy_name, substr(exit_fill_time, 1, 10) AS session_date FROM trades WHERE exit_fill_time IS NOT NULL
                UNION ALL SELECT strategy_name, substr(closed_at, 1, 10) AS session_date FROM trades WHERE closed_at IS NOT NULL
                UNION ALL SELECT strategy_name, session_date FROM positions
                UNION ALL SELECT strategy_name, session_date FROM risk_events
                UNION ALL SELECT strategy_name, session_date FROM runtime_events
            )
            WHERE session_date BETWEEN ? AND ?
            ORDER BY strategy_name
            """,
            [window.start_date, window.end_date],
        )
        return [str(x) for x in rows["strategy_name"].fillna("unknown").replace("", "unknown").unique().tolist()]
    finally:
        conn.close()


def strategy_clause(alias: str, strategy: str | None) -> tuple[str, list[Any]]:
    if not strategy or strategy == "All":
        return "", []
    return f" AND COALESCE({alias}.strategy_name, 'unknown') = ?", [strategy]


def load_executions(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> pd.DataFrame:
    if not strategy or strategy == "All":
        clause, params = "", []
    else:
        clause, params = " AND (COALESCE(e.strategy_name, 'unknown') = ? OR COALESCE(e.strategy_name, 'unknown') = 'unknown')", [strategy]
    rows = read_sql(
        conn,
        f"""
        SELECT
            execution_id,
            trade_id,
            COALESCE(strategy_name, 'unknown') AS strategy,
            COALESCE(strategy_name, 'unknown') AS strategy_name,
            session_date,
            symbol,
            side,
            side AS action,
            quantity AS qty,
            quantity,
            price,
            price AS fill_price,
            quantity * price AS gross_value,
            order_id,
            perm_id,
            exchange,
            liquidity,
            executed_at AS time,
            executed_at,
            recorded_at,
            commission,
            commission_currency,
            realized_pnl,
            commission_source,
            exit_reason,
            exit_reason_source,
            raw_json
        FROM executions e
        WHERE e.session_date BETWEEN ? AND ? {clause}
        ORDER BY COALESCE(e.executed_at, e.recorded_at) DESC, e.execution_id DESC
        """,
        [window.start_date, window.end_date, *params],
    )
    if rows.empty:
        return rows
    data_quality: list[str] = []
    for row in rows.to_dict("records"):
        side = str(row.get("side") or row.get("action") or "").upper()
        commission = to_float(row.get("commission"), None)
        commission_source = str(row.get("commission_source") or "").strip()
        realized_pnl = to_float(row.get("realized_pnl"), None)
        if side in {"SLD", "SELL"} and realized_pnl is None:
            data_quality.append("PNL_PENDING")
        elif (commission is None or abs(commission) < 1e-12) and not commission_source:
            data_quality.append("COMMISSION_PENDING")
        else:
            data_quality.append("OK")
    rows["data_quality"] = data_quality
    return rows


def load_execution_pnl_summary(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[str, float]:
    """Closed daily PnL source of truth from immutable IBKR executions."""
    if not strategy or strategy == "All":
        clause, params = "", []
    else:
        clause, params = " AND (COALESCE(e.strategy_name, 'unknown') = ? OR COALESCE(e.strategy_name, 'unknown') = 'unknown')", [strategy]
    execution_day_expr = """
            COALESCE(
                NULLIF(substr(e.executed_at, 1, 10), ''),
                NULLIF(substr(e.recorded_at, 1, 10), ''),
                COALESCE(e.session_date, '')
            )
    """
    rows = read_sql(
        conn,
        f"""
        SELECT
            COUNT(*) AS execution_rows,
            COUNT(DISTINCT COALESCE(symbol, '')) AS symbols,
            SUM(COALESCE(realized_pnl, 0)) AS gross_realized,
            SUM(CASE
                WHEN UPPER(COALESCE(side, '')) IN ('SLD', 'SELL') THEN COALESCE(commission, 0)
                ELSE 0
            END) AS sell_commissions,
            SUM(COALESCE(commission, 0)) AS all_commissions
        FROM executions e
        WHERE {execution_day_expr} BETWEEN ? AND ?
        {clause}
        """,
        [window.start_date, window.end_date, *params],
    )
    row = rows.iloc[0].to_dict() if not rows.empty else {}
    closed_symbol_rows = read_sql(
        conn,
        f"""
        SELECT COUNT(*) AS closed_symbols
        FROM (
            SELECT
                UPPER(COALESCE(symbol, '')) AS symbol,
                SUM(CASE
                    WHEN UPPER(COALESCE(side, '')) IN ('BOT', 'BUY') THEN COALESCE(quantity, 0)
                    WHEN UPPER(COALESCE(side, '')) IN ('SLD', 'SELL') THEN -COALESCE(quantity, 0)
                    ELSE 0
                END) AS net_quantity,
                SUM(CASE
                    WHEN UPPER(COALESCE(side, '')) IN ('SLD', 'SELL') THEN COALESCE(quantity, 0)
                    ELSE 0
                END) AS sell_quantity
            FROM executions e
            WHERE {execution_day_expr} BETWEEN ? AND ?
            {clause}
            GROUP BY UPPER(COALESCE(symbol, ''))
            HAVING symbol != ''
               AND ABS(net_quantity) <= 0.000001
               AND sell_quantity > 0
        )
        """,
        [window.start_date, window.end_date, *params],
    )
    closed_symbols = float((closed_symbol_rows.iloc[0].to_dict() if not closed_symbol_rows.empty else {}).get("closed_symbols") or 0)
    gross = float(row.get("gross_realized") or 0.0)
    sell_commissions = float(row.get("sell_commissions") or 0.0)
    all_commissions = float(row.get("all_commissions") or 0.0)
    return {
        "execution_rows": float(row.get("execution_rows") or 0),
        "symbols": float(row.get("symbols") or 0),
        "closed_symbols": closed_symbols,
        "gross_pnl": gross,
        "commissions": sell_commissions,
        "sell_commissions": sell_commissions,
        "all_commissions": all_commissions,
        "net_actual_pnl": gross - sell_commissions,
        "main_pnl_source": "executions_realized_pnl_minus_sell_commission",
    }


def confirmed_commission_maps(executions: pd.DataFrame) -> tuple[dict[str, float], dict[tuple[str, str, str], float]]:
    if executions.empty or "commission" not in executions.columns:
        return {}, {}
    rows = executions.copy()
    rows["commission_source"] = rows.get("commission_source", "").fillna("").astype(str).str.lower()
    rows["commission"] = pd.to_numeric(rows["commission"], errors="coerce")
    rows = rows[(rows["commission_source"] == "ibkr") & rows["commission"].notna()]
    if rows.empty:
        return {}, {}
    rows["confirmed_commission"] = rows["commission"].abs()
    by_trade: dict[str, float] = {}
    if "trade_id" in rows.columns:
        trade_rows = rows[rows["trade_id"].fillna("").astype(str) != ""]
        by_trade = {
            str(trade_id): float(value)
            for trade_id, value in trade_rows.groupby("trade_id")["confirmed_commission"].sum().items()
        }
    by_symbol = {
        (str(session_date), str(strategy), str(symbol).upper()): float(value)
        for (session_date, strategy, symbol), value in rows.groupby(["session_date", "strategy", "symbol"])["confirmed_commission"].sum().items()
    }
    return by_trade, by_symbol


def execution_time_value(row: dict[str, Any]) -> Any:
    return row.get("executed_at") or raw_execution_time_value(row.get("raw_json")) or row.get("recorded_at")


def execution_time_sort_key(row: dict[str, Any]) -> datetime:
    return parse_dt(execution_time_value(row)) or parse_dt(row.get("recorded_at")) or datetime.min.replace(tzinfo=timezone.utc)


def side_rows(executions: pd.DataFrame, *, action_values: set[str]) -> list[dict[str, Any]]:
    if executions.empty:
        return []
    action_series = executions.get("action", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    rows = executions[action_series.isin(action_values)].to_dict("records")
    return sorted(rows, key=execution_time_sort_key)


def time_window_rows(rows: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    if rows.empty:
        return rows
    start_dt = parse_dt(start)
    end_dt = parse_dt(end)
    if start_dt is None and end_dt is None:
        return rows
    mask = []
    for row in rows.to_dict("records"):
        ts = execution_time_sort_key(row)
        keep = True
        if start_dt is not None:
            keep = keep and ts >= start_dt
        if end_dt is not None:
            keep = keep and ts <= end_dt
        mask.append(keep)
    return rows[pd.Series(mask, index=rows.index)]


def execution_matches_for_trade(row: dict[str, Any], executions: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if executions.empty:
        return [], [], "missing"
    trade_id = str(row.get("trade_id") or "")
    matched = pd.DataFrame()
    matched_by = "missing"
    if trade_id and "trade_id" in executions.columns:
        matched = executions[executions["trade_id"].fillna("").astype(str) == trade_id]
        if not matched.empty:
            matched_by = "trade_id"
    if matched.empty and "execution_id" in executions.columns:
        raw = parse_raw_json(row.get("raw_json"))
        execution_ids = {
            str(raw.get("buy_execution_id") or ""),
            str(raw.get("sell_execution_id") or ""),
        } - {""}
        if execution_ids:
            matched = executions[executions["execution_id"].fillna("").astype(str).isin(execution_ids)]
            if not matched.empty:
                matched_by = "reconstructed_pair"
    if matched.empty:
        entry_date = date_part(row.get("entry_time"))
        exit_date = date_part(row.get("exit_time") or row.get("closed_at"))
        allowed_dates = {str(row.get("session_date") or ""), entry_date, exit_date} - {""}
        same = executions[
            (executions["session_date"].fillna("").astype(str).isin(allowed_dates))
            & (executions["symbol"].fillna("").astype(str).str.upper() == str(row.get("symbol") or "").upper())
        ]
        has_time_window = parse_dt(row.get("entry_time")) is not None or parse_dt(row.get("exit_time") or row.get("closed_at")) is not None
        same = time_window_rows(same, row.get("entry_time"), row.get("exit_time") or row.get("closed_at"))
        buy_count = len(side_rows(same, action_values={"BOT", "BUY"}))
        sell_count = len(side_rows(same, action_values={"SLD", "SELL"}))
        if has_time_window and buy_count >= 1 and sell_count >= 1:
            matched = same
            matched_by = "symbol_session"
        elif buy_count == 1 and sell_count == 1:
            matched = same
            matched_by = "symbol_session"
    return side_rows(matched, action_values={"BOT", "BUY"}), side_rows(matched, action_values={"SLD", "SELL"}), matched_by


def infer_entry_exit_times(row: dict[str, Any], buy_rows: list[dict[str, Any]], sell_rows: list[dict[str, Any]]) -> tuple[Any, Any, set[str]]:
    flags: set[str] = set()
    entry_time = row.get("entry_time")
    if is_missing_value(entry_time) and buy_rows:
        entry_time = execution_time_value(buy_rows[0])
    if is_missing_value(entry_time):
        entry_time = None
        flags.add("MISSING_ENTRY")

    exit_time = row.get("exit_time")
    if is_missing_value(exit_time):
        exit_time = row.get("closed_at")
    if is_missing_value(exit_time) and sell_rows:
        exit_time = execution_time_value(sell_rows[-1])
    if is_missing_value(exit_time):
        exit_time = None
        flags.add("MISSING_EXIT")

    entry_dt = parse_dt(entry_time)
    exit_dt = parse_dt(exit_time)
    if entry_dt and exit_dt and entry_dt.replace(microsecond=0) == exit_dt.replace(microsecond=0):
        true_same_second = False
        if buy_rows and sell_rows:
            buy_dt = parse_dt(execution_time_value(buy_rows[0]))
            sell_dt = parse_dt(execution_time_value(sell_rows[-1]))
            true_same_second = bool(buy_dt and sell_dt and buy_dt.replace(microsecond=0) == sell_dt.replace(microsecond=0))
        if not true_same_second:
            flags.add("SUSPECT_TIME_MATCH")
    return entry_time, exit_time, flags


def confirmed_commission_for_execution_rows(buy_rows: list[dict[str, Any]], sell_rows: list[dict[str, Any]]) -> tuple[float, str]:
    def side_commission(rows: list[dict[str, Any]]) -> tuple[float, int]:
        total = 0.0
        present = 0
        for row in rows:
            if str(row.get("commission_source") or "").lower() != "ibkr":
                continue
            value = to_float(row.get("commission"), None)
            if value is None:
                continue
            total += abs(float(value))
            present += 1
        return total, present

    buy_commission, buy_confirmed = side_commission(buy_rows)
    sell_commission, sell_confirmed = side_commission(sell_rows)
    total = buy_commission + sell_commission
    matched_count = len(buy_rows) + len(sell_rows)
    confirmed_count = buy_confirmed + sell_confirmed
    both_sides_matched = bool(buy_rows and sell_rows)
    if matched_count and both_sides_matched and confirmed_count == matched_count:
        return total, "OK"
    if confirmed_count:
        return total, "PARTIAL"
    return 0.0, "MISSING"


def confirmed_commission_execution_count(buy_rows: list[dict[str, Any]], sell_rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in [*buy_rows, *sell_rows]:
        if str(row.get("commission_source") or "").lower() == "ibkr" and to_float(row.get("commission"), None) is not None:
            count += 1
    return count


def quality_label(flags: set[str], commission_status: str) -> str:
    out = set(flags)
    if commission_status == "PARTIAL":
        out.add("COMMISSION_PARTIAL")
    elif commission_status == "MISSING":
        out.add("COMMISSION_MISSING")
    return "; ".join(sorted(out)) if out else "OK"


def quality_label_from_flags(flags: set[str]) -> str:
    return "; ".join(sorted(flags)) if flags else "OK"


def load_runtime_peak_map(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[tuple[str, str, str, str], tuple[float, str]]:
    clause, params = strategy_clause("r", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            trade_id,
            COALESCE(strategy_name, 'unknown') AS strategy,
            COALESCE(r.session_date, substr(r.event_time, 1, 10)) AS session_date,
            symbol,
            event_type,
            raw_json
        FROM runtime_events r
        WHERE COALESCE(r.session_date, substr(r.event_time, 1, 10)) BETWEEN ? AND ? {clause}
        ORDER BY r.event_time
        """,
        [window.start_date, window.end_date, *params],
    )
    peak_map: dict[tuple[str, str, str, str], tuple[float, str]] = {}
    for row in rows.to_dict("records"):
        raw = parse_raw_json(row.get("raw_json"))
        peak = raw_json_peak_value(raw)
        if peak is None:
            peak = to_float(raw.get("peak_gain_pct") or raw.get("mfe_pct"), None)
        if peak is None:
            continue
        key = (
            str(row.get("trade_id") or ""),
            str(row.get("session_date") or ""),
            str(row.get("strategy") or "unknown"),
            str(row.get("symbol") or "").upper(),
        )
        previous = peak_map.get(key)
        if previous is None or peak > previous[0]:
            peak_map[key] = (peak, "runtime_events")
    return peak_map


def merge_symbol_peak(
    out: dict[tuple[str, str, str], dict[str, Any]],
    key: tuple[str, str, str],
    *,
    peak_price: float | None = None,
    peak_pct: float | None = None,
    entry_price: float | None = None,
    exit_price: float | None = None,
    source: str,
) -> None:
    current = out.get(key, {"source": source})
    if entry_price is not None:
        current["entry_price"] = entry_price
    if exit_price is not None:
        current["exit_price"] = exit_price
    if peak_price is not None and (current.get("peak_price") is None or peak_price > current["peak_price"]):
        current["peak_price"] = peak_price
    if peak_pct is not None and (current.get("peak_pct") is None or peak_pct > current["peak_pct"]):
        current["peak_pct"] = peak_pct
    current["source"] = source
    calculated_peak_pct = pct_from_prices(current.get("peak_price"), current.get("entry_price"))
    if calculated_peak_pct is not None:
        current["peak_pct"] = calculated_peak_pct
    drop_pct = pct_from_prices(current.get("exit_price"), current.get("peak_price"))
    if drop_pct is not None:
        current["drop_from_peak_pct"] = drop_pct
    out[key] = current


def load_runtime_symbol_peak_map(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    clause, params = strategy_clause("r", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            COALESCE(strategy_name, 'unknown') AS strategy,
            COALESCE(r.session_date, substr(r.event_time, 1, 10)) AS session_date,
            symbol,
            event_type,
            raw_json
        FROM runtime_events r
        WHERE COALESCE(r.session_date, substr(r.event_time, 1, 10)) BETWEEN ? AND ? {clause}
        ORDER BY r.event_time
        """,
        [window.start_date, window.end_date, *params],
    )
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        session_date = str(row.get("session_date") or "")
        symbol = str(row.get("symbol") or "").upper()
        if not session_date or not symbol:
            continue
        raw = parse_raw_json(row.get("raw_json"))
        event_type = str(row.get("event_type") or "").upper()
        peak_price = raw_price_value(raw, ("peak_price", "high_watermark", "mfe_price")) if "PEAK" in event_type or raw.get("peak_price") is not None else None
        entry_price = raw_price_value(raw, ("entry_price", "buy", "buy_price"))
        exit_price = raw_price_value(raw, ("exit_price", "sell_price", "decision_price", "price")) if event_type in {"SELL_ORDER_SENT", "POSITION_VERIFIED_CLOSED", "POSITION_CLOSED"} else None
        peak_pct = raw_json_peak_value(raw)
        if peak_price is None and entry_price is not None and peak_pct is not None:
            peak_price = entry_price * (1.0 + peak_pct / 100.0)
        if peak_price is None and peak_pct is None and exit_price is None and entry_price is None:
            continue
        strategy_name = str(row.get("strategy") or "unknown")
        for key_strategy in (strategy_name, ""):
            merge_symbol_peak(
                out,
                (session_date, key_strategy, symbol),
                peak_price=peak_price,
                peak_pct=peak_pct,
                entry_price=entry_price,
                exit_price=exit_price,
                source="runtime_events_symbol_session",
            )
    return out


def recorder_root() -> Path:
    return Path(os.environ.get("TRADING_BOT_RECORDER_DIR") or os.environ.get("DASHBOARD_RECORDER_DIR") or DEFAULT_RECORDER_ROOT)


EXIT_REASON_EVENTS = {
    "SELL_ORDER_SENT",
    "EXIT_ORDER_SUBMITTED",
    "EOD_FLATTEN_SUBMIT",
    "EOD_FLATTEN_RETRY",
    "EOD_FLATTEN_SUCCESS",
    "MANUAL_FLATTEN_SENT",
    "POSITION_VERIFIED_CLOSED",
    "POSITION_CLOSED",
    "POSITION_CLOSED_UNVERIFIED",
    "RECONCILIATION_CLOSE_WITHOUT_FILL",
    "ORDER_CANCEL_BUT_POSITION_EXISTS",
}


def normalize_exit_reason(value: Any, event_type: str = "") -> str:
    text = str(value or "").strip()
    if text and text.lower() not in {"none", "nan", "null", "unknown"}:
        normalized = text.lower().replace(" ", "_").replace("-", "_")
        upper_text = text.upper()
        if "TRAIL" in upper_text:
            return "trailing_stop"
        if "STOP_LOSS" in upper_text or normalized == "stop_loss":
            return "stop_loss"
        if "EOD" in upper_text:
            return "eod_flatten"
        if "MANUAL" in upper_text:
            return "manual_flatten"
        if "ORPHAN" in upper_text:
            return "orphan_flatten"
        if "RECONCILIATION" in upper_text:
            return "reconciliation"
        return text
    event = str(event_type or "").upper()
    if event.startswith("EOD_FLATTEN"):
        return "eod_flatten"
    if event.startswith("MANUAL_FLATTEN"):
        return "manual_flatten"
    if "ORPHAN" in event:
        return "orphan_flatten"
    if "RECONCILIATION" in event:
        return "reconciliation"
    if "STOP_LOSS" in event:
        return "stop_loss"
    if "TRAIL" in event:
        return "trailing_stop"
    return ""


def exit_reason_from_payload(payload: dict[str, Any], event_type: str = "") -> str:
    exit_event = str(event_type or "").upper() in EXIT_REASON_EVENTS
    for key in (
        "exit_reason",
        "requested_exit_reason",
        "close_reason",
        "flatten_reason",
        "order_purpose",
        "purpose",
        "trigger",
    ):
        reason = normalize_exit_reason(payload.get(key), event_type)
        if reason:
            return reason
    if exit_event:
        reason = normalize_exit_reason(payload.get("reason"), event_type)
        if reason:
            return reason
    raw = payload.get("raw_json")
    if raw:
        reason = exit_reason_from_payload(parse_raw_json(raw), event_type)
        if reason:
            return reason
    return normalize_exit_reason("", event_type)


def put_exit_reason_event(
    out: dict[tuple[str, str, str], list[dict[str, Any]]],
    key: tuple[str, str, str],
    reason: str,
    *,
    event_time: Any = None,
    event_type: str = "",
    source: str = "",
    trade_id: str = "",
    order_id: Any = None,
    perm_id: Any = None,
    execution_id: Any = None,
) -> None:
    if not reason:
        return
    events = out.setdefault(key, [])
    item = {
        "reason": reason,
        "event_time": event_time,
        "event_type": str(event_type or ""),
        "source": source,
        "trade_id": str(trade_id or ""),
        "order_id": normalized_identifier(order_id),
        "perm_id": normalized_identifier(perm_id),
        "execution_id": normalized_identifier(execution_id),
    }
    identity = (
        item["reason"],
        item["event_time"],
        item["event_type"],
        item["source"],
        item["trade_id"],
        item["order_id"],
        item["perm_id"],
        item["execution_id"],
    )
    if identity not in {
        (
            str(existing.get("reason") or ""),
            existing.get("event_time"),
            str(existing.get("event_type") or ""),
            str(existing.get("source") or ""),
            str(existing.get("trade_id") or ""),
            normalized_identifier(existing.get("order_id")),
            normalized_identifier(existing.get("perm_id")),
            normalized_identifier(existing.get("execution_id")),
        )
        for existing in events
    }:
        events.append(item)


def load_runtime_exit_reason_map(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    clause, params = strategy_clause("r", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            COALESCE(r.session_date, substr(r.event_time, 1, 10)) AS session_date,
            COALESCE(r.strategy_name, 'unknown') AS strategy,
            r.event_time,
            r.symbol,
            r.trade_id,
            r.order_id,
            r.execution_id,
            r.event_type,
            r.reason,
            r.raw_json
        FROM runtime_events r
        WHERE COALESCE(r.session_date, substr(r.event_time, 1, 10)) BETWEEN ? AND ? {clause}
          AND (
              r.event_type IN ({",".join("?" for _ in EXIT_REASON_EVENTS)})
              OR COALESCE(r.raw_json, '') LIKE '%exit_reason%'
              OR COALESCE(r.raw_json, '') LIKE '%requested_exit_reason%'
          )
        ORDER BY r.event_time
        """,
        [window.start_date, window.end_date, *params, *sorted(EXIT_REASON_EVENTS)],
    )
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows.to_dict("records"):
        session_date = str(row.get("session_date") or "")
        symbol = str(row.get("symbol") or "").upper()
        if not session_date or not symbol:
            continue
        event_type = str(row.get("event_type") or "").upper()
        raw = parse_raw_json(row.get("raw_json"))
        payload = {**raw, "reason": row.get("reason")}
        reason = exit_reason_from_payload(payload, event_type)
        if not reason:
            continue
        strategy_name = str(row.get("strategy") or "unknown")
        trade_id = str(row.get("trade_id") or "")
        order_id = row.get("order_id") or raw.get("order_id") or raw.get("ib_order_id")
        perm_id = raw.get("perm_id") or raw.get("permId")
        execution_id = row.get("execution_id") or raw.get("execution_id") or raw.get("execId")
        for key_strategy in (strategy_name, ""):
            put_exit_reason_event(
                out,
                (session_date, key_strategy, symbol),
                reason,
                event_time=row.get("event_time"),
                event_type=event_type,
                source="runtime_events",
                trade_id=trade_id,
                order_id=order_id,
                perm_id=perm_id,
                execution_id=execution_id,
            )
        if trade_id:
            put_exit_reason_event(
                out,
                (session_date, f"trade:{trade_id}", symbol),
                reason,
                event_time=row.get("event_time"),
                event_type=event_type,
                source="runtime_events",
                trade_id=trade_id,
                order_id=order_id,
                perm_id=perm_id,
                execution_id=execution_id,
            )
    return out


def load_lifecycle_exit_reason_map(window: DateWindow, root: Path | None = None) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    base = root or recorder_root()
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for session_dir in sorted(base.glob("*")):
        if not session_dir.is_dir():
            continue
        session_date = session_dir.name
        if session_date < window.start_date or session_date > window.end_date:
            continue
        csv_path = session_dir / "trade_lifecycle.csv"
        if csv_path.exists():
            try:
                rows = pd.read_csv(csv_path).to_dict("records")
            except Exception:
                rows = []
            for row in rows:
                event_type = str(row.get("event") or row.get("event_type") or "").upper()
                raw = parse_raw_json(row.get("raw_json"))
                if event_type not in EXIT_REASON_EVENTS and not raw.get("exit_reason") and not raw.get("requested_exit_reason"):
                    continue
                symbol = str(row.get("symbol") or "").upper()
                if not symbol:
                    continue
                reason = exit_reason_from_payload(row, event_type)
                strategy_name = str(row.get("strategy") or "unknown")
                order_id = row.get("order_id") or raw.get("order_id") or raw.get("ib_order_id")
                perm_id = row.get("perm_id") or raw.get("perm_id") or raw.get("permId")
                execution_id = row.get("execution_id") or raw.get("execution_id") or raw.get("execId")
                for key_strategy in (strategy_name, ""):
                    put_exit_reason_event(
                        out,
                        (session_date, key_strategy, symbol),
                        reason,
                        event_time=row.get("recorded_at") or row.get("event_time") or row.get("timestamp"),
                        event_type=event_type,
                        source="trade_lifecycle.csv",
                        trade_id=str(row.get("trade_id") or ""),
                        order_id=order_id,
                        perm_id=perm_id,
                        execution_id=execution_id,
                    )
        jsonl_path = session_dir / "order_lifecycle.jsonl"
        if jsonl_path.exists():
            try:
                lines = jsonl_path.read_text(errors="replace").splitlines()
            except Exception:
                lines = []
            for line in lines:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                event_type = str(row.get("event_type") or row.get("type") or row.get("event") or "").upper()
                legacy = str(row.get("legacy_event") or "").upper()
                event_for_reason = legacy or event_type
                raw = parse_raw_json(row.get("raw_json"))
                if event_for_reason not in EXIT_REASON_EVENTS and not row.get("exit_reason") and not row.get("requested_exit_reason") and not raw.get("exit_reason") and not raw.get("requested_exit_reason"):
                    continue
                symbol = str(row.get("symbol") or "").upper()
                if not symbol:
                    continue
                reason = exit_reason_from_payload(row, event_for_reason)
                strategy_name = str(row.get("strategy_name") or row.get("strategy") or "unknown")
                order_id = (
                    row.get("ib_order_id")
                    or row.get("order_id")
                    or raw.get("ib_order_id")
                    or raw.get("order_id")
                )
                perm_id = row.get("perm_id") or raw.get("perm_id") or raw.get("permId")
                execution_id = row.get("execution_id") or raw.get("execution_id") or raw.get("execId")
                for key_strategy in (strategy_name, ""):
                    put_exit_reason_event(
                        out,
                        (session_date, key_strategy, symbol),
                        reason,
                        event_time=row.get("event_time") or row.get("recorded_at") or row.get("timestamp"),
                        event_type=event_for_reason,
                        source="order_lifecycle.jsonl",
                        trade_id=str(row.get("trade_id") or ""),
                        order_id=order_id,
                        perm_id=perm_id,
                        execution_id=execution_id,
                    )
    return out


def event_matches_exit_identifiers(
    event: dict[str, Any],
    *,
    order_id: Any = None,
    perm_id: Any = None,
    execution_id: Any = None,
) -> bool:
    if normalized_identifier(order_id) and normalized_identifier(order_id) == normalized_identifier(event.get("order_id")):
        return True
    if normalized_identifier(perm_id) and normalized_identifier(perm_id) == normalized_identifier(event.get("perm_id")):
        return True
    if normalized_identifier(execution_id) and normalized_identifier(execution_id) == normalized_identifier(event.get("execution_id")):
        return True
    return False


def exit_reason_event_matches_trade(
    event: dict[str, Any],
    exit_time: Any,
    *,
    exact_trade: bool = False,
    order_id: Any = None,
    perm_id: Any = None,
    execution_id: Any = None,
) -> bool:
    if exact_trade:
        return True
    if event_matches_exit_identifiers(event, order_id=order_id, perm_id=perm_id, execution_id=execution_id):
        return True
    event_type = str(event.get("event_type") or "").upper()
    event_dt = parse_dt(event.get("event_time"))
    exit_dt = parse_dt(exit_time)
    if event_dt is None or exit_dt is None:
        return False
    delta_seconds = abs((exit_dt - event_dt).total_seconds())
    if event_type.startswith("EOD_FLATTEN"):
        return delta_seconds <= 45 * 60
    return delta_seconds <= 10 * 60


def choose_exit_reason_event(
    reason_map: dict[tuple[str, str, str], list[dict[str, Any]]],
    keys: list[tuple[str, str, str]],
    exit_time: Any,
    *,
    exact_trade_key: tuple[str, str, str] | None = None,
    order_id: Any = None,
    perm_id: Any = None,
    execution_id: Any = None,
) -> dict[str, Any] | None:
    for key in keys:
        events = reason_map.get(key, [])
        if not events:
            continue
        exact = exact_trade_key is not None and key == exact_trade_key
        id_matching = [
            event for event in events
            if exact or event_matches_exit_identifiers(event, order_id=order_id, perm_id=perm_id, execution_id=execution_id)
        ]
        matching = id_matching or [
            event for event in events
            if exit_reason_event_matches_trade(
                event,
                exit_time,
                exact_trade=exact,
                order_id=order_id,
                perm_id=perm_id,
                execution_id=execution_id,
            )
        ]
        if not matching:
            continue
        exit_dt = parse_dt(exit_time)
        if exit_dt is not None:
            matching.sort(key=lambda event: abs((exit_dt - (parse_dt(event.get("event_time")) or exit_dt)).total_seconds()))
        return matching[0]
    return None


def load_lifecycle_peak_map(window: DateWindow, root: Path | None = None) -> dict[tuple[str, str], tuple[float, str]]:
    base = root or recorder_root()
    peak_map: dict[tuple[str, str], tuple[float, str]] = {}
    for session_dir in sorted(base.glob("*")):
        if not session_dir.is_dir():
            continue
        session_date = session_dir.name
        if session_date < window.start_date or session_date > window.end_date:
            continue
        path = session_dir / "trade_lifecycle.csv"
        if not path.exists():
            continue
        try:
            rows = pd.read_csv(path)
        except Exception:
            continue
        if rows.empty or "symbol" not in rows.columns:
            continue
        for row in rows.to_dict("records"):
            peak = to_float(row.get("peak_gain_pct") or row.get("mfe_pct") or row.get("peak_pct"), None)
            if peak is None:
                entry_price = to_float(row.get("entry_price"), None)
                peak_price = to_float(row.get("peak_price"), None)
                if entry_price and peak_price is not None:
                    peak = ((peak_price / entry_price) - 1.0) * 100.0
            if peak is None:
                continue
            key = (session_date, str(row.get("symbol") or "").upper())
            previous = peak_map.get(key)
            if previous is None or peak > previous[0]:
                peak_map[key] = (peak, "trade_lifecycle.csv")
    return peak_map


def load_lifecycle_symbol_peak_map(window: DateWindow, root: Path | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    base = root or recorder_root()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for session_dir in sorted(base.glob("*")):
        if not session_dir.is_dir():
            continue
        session_date = session_dir.name
        if session_date < window.start_date or session_date > window.end_date:
            continue
        path = session_dir / "trade_lifecycle.csv"
        if not path.exists():
            continue
        try:
            rows = pd.read_csv(path)
        except Exception:
            continue
        if rows.empty or "symbol" not in rows.columns:
            continue
        for row in rows.to_dict("records"):
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            event_type = str(row.get("event") or row.get("event_type") or "").upper()
            entry_price = to_float(row.get("entry_price") or row.get("buy") or row.get("buy_price"), None)
            peak_price = to_float(row.get("peak_price") or row.get("high_watermark") or row.get("mfe_price"), None)
            if peak_price is None and entry_price is not None:
                peak_pct = to_float(row.get("peak_gain_pct") or row.get("mfe_pct") or row.get("peak_pct"), None)
                if peak_pct is not None:
                    peak_price = entry_price * (1.0 + peak_pct / 100.0)
            else:
                peak_pct = to_float(row.get("peak_gain_pct") or row.get("mfe_pct") or row.get("peak_pct"), None)
            exit_price = None
            if event_type in {"SELL_ORDER_SENT", "POSITION_VERIFIED_CLOSED", "POSITION_CLOSED"}:
                exit_price = to_float(row.get("exit_price") or row.get("sell_price") or row.get("decision_price") or row.get("price"), None)
            if peak_price is None and peak_pct is None and entry_price is None and exit_price is None:
                continue
            merge_symbol_peak(
                out,
                (session_date, symbol),
                peak_price=peak_price,
                peak_pct=peak_pct,
                entry_price=entry_price,
                exit_price=exit_price,
                source="trade_lifecycle_symbol_session",
            )
    return out


def load_candle_rows(window: DateWindow, root: Path | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    base = root or recorder_root()
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for session_dir in sorted(base.glob("*")):
        if not session_dir.is_dir():
            continue
        session_date = session_dir.name
        if session_date < window.start_date or session_date > window.end_date:
            continue
        path = session_dir / "candles_1m.csv"
        if not path.exists():
            continue
        try:
            candles = pd.read_csv(path)
        except Exception:
            continue
        if candles.empty or not {"symbol", "high"}.issubset(candles.columns):
            continue
        for symbol, group in candles.groupby(candles["symbol"].fillna("").astype(str).str.upper()):
            if not symbol:
                continue
            out[(session_date, symbol)] = group.copy()
    return out


def candle_peak_for_trade(row: dict[str, Any], candle_rows: dict[tuple[str, str], pd.DataFrame]) -> tuple[float | None, str]:
    buy = to_float(row.get("buy"), None)
    if not buy:
        return None, "missing"
    candles = candle_rows.get((str(row.get("session_date") or ""), str(row.get("symbol") or "").upper()))
    if candles is None or candles.empty:
        return None, "missing"
    time_col = "bar_time" if "bar_time" in candles.columns else ("timestamp" if "timestamp" in candles.columns else None)
    scoped = candles
    if time_col:
        start_dt = parse_dt(row.get("entry_time"))
        end_dt = parse_dt(row.get("exit_time") or row.get("closed_at"))
        if start_dt or end_dt:
            times = pd.to_datetime(scoped[time_col], errors="coerce", utc=True)
            mask = pd.Series(True, index=scoped.index)
            if start_dt:
                mask &= times >= start_dt
            if end_dt:
                mask &= times <= end_dt
            scoped = scoped[mask]
    highs = pd.to_numeric(scoped["high"], errors="coerce")
    peak_price = highs.max()
    if pd.isna(peak_price):
        return None, "missing"
    return ((float(peak_price) / buy) - 1.0) * 100.0, "candles_1m.csv"


def infer_strategy_by_exit_symbol(executions: pd.DataFrame) -> dict[tuple[str, str], str]:
    if executions is None or executions.empty:
        return {}
    rows = executions.copy()
    rows["symbol_norm"] = rows.get("symbol", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    rows["strategy_norm"] = rows.get("strategy", pd.Series(dtype=str)).fillna("").astype(str)
    rows["action_norm"] = rows.get("action", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    rows["exit_date"] = rows.apply(lambda row: date_part(row.get("executed_at") or row.get("recorded_at") or row.get("session_date")), axis=1)
    rows = rows[
        (rows["symbol_norm"] != "")
        & (rows["exit_date"] != "")
        & (rows["strategy_norm"] != "")
        & (rows["strategy_norm"].str.lower() != "unknown")
        & (rows["action_norm"].isin({"SLD", "SELL"}))
    ]
    out: dict[tuple[str, str], str] = {}
    for (exit_date, symbol), group in rows.groupby(["exit_date", "symbol_norm"], dropna=False):
        strategies = sorted({str(value) for value in group["strategy_norm"].dropna().tolist() if str(value) and str(value).lower() != "unknown"})
        if len(strategies) == 1:
            out[(str(exit_date), str(symbol))] = strategies[0]
    return out


def peak_from_sources(
    row: dict[str, Any],
    runtime_peak_map: dict[tuple[str, str, str, str], tuple[float, str]],
    lifecycle_peak_map: dict[tuple[str, str], tuple[float, str]],
    candle_rows: dict[tuple[str, str], pd.DataFrame],
) -> tuple[float | None, str]:
    direct = to_float(row.get("peak_pct"), None)
    if direct is not None:
        return direct, "trades.mfe_pct"
    raw = parse_raw_json(row.get("raw_json"))
    raw_peak = raw_json_peak_value(raw)
    if raw_peak is not None:
        return raw_peak, "trades.raw_json"
    trade_key = (
        str(row.get("trade_id") or ""),
        str(row.get("session_date") or ""),
        str(row.get("strategy") or "unknown"),
        str(row.get("symbol") or "").upper(),
    )
    runtime_peak = runtime_peak_map.get(trade_key)
    if runtime_peak:
        return runtime_peak
    return None, "missing"


def sell_execution_candidates(executions: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    if executions is None or executions.empty:
        return {}, {}, {}
    by_execution_id: dict[str, dict[str, Any]] = {}
    by_trade_id: dict[str, list[dict[str, Any]]] = {}
    by_symbol_date: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in executions.to_dict("records"):
        side = str(row.get("side") or row.get("action") or "").upper()
        if side not in {"SLD", "SELL"}:
            continue
        execution_id = normalized_identifier(row.get("execution_id"))
        trade_id = normalized_identifier(row.get("trade_id"))
        symbol = str(row.get("symbol") or "").upper()
        execution_date = date_part(row.get("executed_at") or row.get("recorded_at") or row.get("session_date"))
        if execution_id:
            by_execution_id[execution_id] = row
        if trade_id:
            by_trade_id.setdefault(trade_id, []).append(row)
        if symbol and execution_date:
            by_symbol_date.setdefault((execution_date, symbol), []).append(row)
    for rows in [*by_trade_id.values(), *by_symbol_date.values()]:
        rows.sort(key=lambda item: parse_dt(item.get("executed_at") or item.get("recorded_at")) or datetime.min.replace(tzinfo=timezone.utc))
    return by_execution_id, by_trade_id, by_symbol_date


def closest_sell_execution(
    row: dict[str, Any],
    raw: dict[str, Any],
    *,
    by_execution_id: dict[str, dict[str, Any]],
    by_trade_id: dict[str, list[dict[str, Any]]],
    by_symbol_date: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    exit_execution_id = raw_trade_execution_id(raw, "exit")
    if exit_execution_id and exit_execution_id in by_execution_id:
        return by_execution_id[exit_execution_id]
    trade_id = normalized_identifier(row.get("trade_id"))
    exit_time = row.get("exit_time") or row.get("closed_at")
    exit_dt = parse_dt(exit_time)
    candidates = by_trade_id.get(trade_id, []) if trade_id else []
    if not candidates:
        symbol = str(row.get("symbol") or "").upper()
        exit_date = date_part(exit_time) or date_part(row.get("closed_at")) or str(row.get("session_date") or "")
        candidates = by_symbol_date.get((exit_date, symbol), [])
    if not candidates:
        return {}
    if exit_dt is None:
        return candidates[-1]
    return min(
        candidates,
        key=lambda item: abs((exit_dt - (parse_dt(item.get("executed_at") or item.get("recorded_at")) or exit_dt)).total_seconds()),
    )


def closed_from_trades(
    conn: sqlite3.Connection,
    window: DateWindow,
    strategy: str | None,
    executions: pd.DataFrame,
    runtime_peak_map: dict[tuple[str, str, str, str], tuple[float, str]],
    lifecycle_peak_map: dict[tuple[str, str], tuple[float, str]],
    candle_rows: dict[tuple[str, str], pd.DataFrame],
    runtime_exit_reason_map: dict[tuple[str, str, str], list[dict[str, Any]]] | None = None,
    lifecycle_exit_reason_map: dict[tuple[str, str, str], list[dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    clause, params = strategy_clause("t", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            trade_id,
            COALESCE(strategy_name, 'unknown') AS strategy,
            session_date,
            substr(entry_fill_time, 1, 10) AS entry_date,
            COALESCE(substr(exit_fill_time, 1, 10), substr(closed_at, 1, 10)) AS exit_date,
            symbol,
            status,
            entry_fill_time AS entry_time,
            exit_fill_time AS exit_time,
            closed_at,
            entry_price AS buy,
            exit_price AS sell,
            quantity AS qty,
            gross_pnl AS gross,
            commission AS persisted_commission,
            net_pnl AS persisted_net_pnl,
            mfe_pct AS peak_pct,
            mae_pct,
            peak_price,
            low_price,
            peak_unrealized_pnl,
            max_adverse_unrealized_pnl,
            giveback_from_peak,
            exit_reason,
            top100_rank,
            top100_score,
            top100_source_date,
            top100_features_json,
            live_entry_score,
            live_entry_rank,
            live_entry_features_json,
            signal_source,
            signal_time,
            ready_since,
            entry_order_id,
            entry_perm_id,
            raw_json,
            updated_at,
            trade_reduction_version
        FROM trades t
        WHERE (
            substr(t.exit_fill_time, 1, 10) BETWEEN ? AND ?
            OR substr(t.closed_at, 1, 10) BETWEEN ? AND ?
            OR (
                COALESCE(t.exit_fill_time, t.closed_at) IS NULL
                AND t.session_date BETWEEN ? AND ?
            )
        ) {clause}
          AND UPPER(COALESCE(t.status, '')) IN ({",".join("?" for _ in CLOSED_STATUSES)})
        ORDER BY COALESCE(t.closed_at, t.exit_fill_time), t.symbol
        """,
        [window.start_date, window.end_date, window.start_date, window.end_date, window.start_date, window.end_date, *params, *sorted(CLOSED_STATUSES)],
    )
    if rows.empty:
        return rows
    out = rows.copy()
    runtime_exit_reason_map = runtime_exit_reason_map or {}
    lifecycle_exit_reason_map = lifecycle_exit_reason_map or {}
    inferred_strategy_by_exit_symbol = infer_strategy_by_exit_symbol(executions)
    commissions: list[float] = []
    commission_statuses: list[str] = []
    data_quality: list[str] = []
    entry_times: list[Any] = []
    exit_times: list[Any] = []
    peak_values: list[float | None] = []
    peak_sources: list[str] = []
    peak_match_qualities: list[str] = []
    drop_values: list[float | None] = []
    entry_execution_counts: list[int] = []
    exit_execution_counts: list[int] = []
    confirmed_commission_counts: list[int] = []
    expected_commission_counts: list[int] = []
    commission_source_details: list[str] = []
    entry_execution_ids: list[str] = []
    exit_execution_ids: list[str] = []
    trade_sources: list[str] = []
    exit_reasons: list[str] = []
    exit_reason_sources: list[str] = []
    matched_event_types: list[str] = []
    matched_event_times: list[str] = []
    matched_order_ids: list[str] = []
    strategy_values: list[str] = []
    by_exit_execution_id, sell_executions_by_trade_id, sell_executions_by_symbol_date = sell_execution_candidates(executions)
    trade_raw_peak_by_symbol_session: dict[tuple[str, str, str], tuple[float, str]] = {}
    for peak_row in out.to_dict("records"):
        peak = to_float(peak_row.get("peak_pct"), None)
        source = "trades.mfe_pct"
        if peak is None:
            peak = raw_json_peak_value(parse_raw_json(peak_row.get("raw_json")))
            source = "trades.raw_json"
        if peak is None:
            continue
        key = (
            str(peak_row.get("session_date") or ""),
            str(peak_row.get("strategy") or "unknown"),
            str(peak_row.get("symbol") or "").upper(),
        )
        trade_raw_peak_by_symbol_session.setdefault(key, (peak, source))
    for row in out.to_dict("records"):
        buy_rows: list[dict[str, Any]] = []
        sell_rows: list[dict[str, Any]] = []
        matched_by = "trades_table"
        flags: set[str] = set()
        entry_time = row.get("entry_time")
        exit_time = row.get("exit_time") or row.get("closed_at")
        if not entry_time:
            flags.add("MISSING_ENTRY")
        if not exit_time:
            flags.add("MISSING_EXIT")
        enriched_row = {**row, "entry_time": entry_time, "exit_time": exit_time}
        raw = parse_raw_json(row.get("raw_json"))
        exit_date = date_part(exit_time) or date_part(row.get("closed_at")) or str(row.get("session_date") or "")
        symbol = str(row.get("symbol") or "").upper()
        row_strategy = str(row.get("strategy") or "unknown")
        if row_strategy == "unknown":
            inferred_strategy = inferred_strategy_by_exit_symbol.get((exit_date, symbol)) or inferred_strategy_by_exit_symbol.get((str(row.get("session_date") or ""), symbol))
            if inferred_strategy:
                row_strategy = inferred_strategy
                enriched_row["strategy"] = inferred_strategy
                row["strategy"] = inferred_strategy
                raw["strategy_attribution_source"] = "executions_symbol_exit_date"
        entry_execution_id = raw_trade_execution_id(raw, "entry")
        exit_execution_id = raw_trade_execution_id(raw, "exit")
        sell_execution = closest_sell_execution(
            row,
            raw,
            by_execution_id=by_exit_execution_id,
            by_trade_id=sell_executions_by_trade_id,
            by_symbol_date=sell_executions_by_symbol_date,
        )
        sell_order_id = raw.get("exit_order_id") or raw.get("sell_order_id") or raw.get("order_id") or sell_execution.get("order_id")
        sell_perm_id = raw.get("exit_perm_id") or raw.get("sell_perm_id") or raw.get("perm_id") or sell_execution.get("perm_id")
        sell_execution_id = exit_execution_id or sell_execution.get("execution_id")
        if sell_execution_id:
            exit_execution_id = normalized_identifier(sell_execution_id)
        trade_source = str(raw.get("reconstruction_source") or raw.get("source") or "trades_table")
        matched_event: dict[str, Any] | None = None
        exit_reason = ""
        exit_reason_source = ""
        if sell_execution:
            exit_reason = normalize_exit_reason(sell_execution.get("exit_reason"))
            exit_reason_source = str(sell_execution.get("exit_reason_source") or "executions.exit_reason") if exit_reason else ""
            if exit_reason:
                matched_event = {
                    "reason": exit_reason,
                    "event_type": "SELL_EXECUTION",
                    "event_time": sell_execution.get("executed_at") or sell_execution.get("recorded_at"),
                    "order_id": sell_execution.get("order_id"),
                    "source": "executions",
                }
        if not exit_reason:
            exit_reason = normalize_exit_reason(row.get("exit_reason"))
            exit_reason_source = "trades.exit_reason" if exit_reason else ""
        if not exit_reason:
            exact_runtime_trade_key = (exit_date, f"trade:{row.get('trade_id')}", symbol)
            runtime_candidates = [
                exact_runtime_trade_key,
                (exit_date, row_strategy, symbol),
                (exit_date, "", symbol),
                (str(row.get("session_date") or ""), row_strategy, symbol),
                (str(row.get("session_date") or ""), "", symbol),
            ]
            matched_event = choose_exit_reason_event(
                runtime_exit_reason_map,
                runtime_candidates,
                exit_time,
                exact_trade_key=exact_runtime_trade_key,
                order_id=sell_order_id,
                perm_id=sell_perm_id,
                execution_id=sell_execution_id,
            )
            if matched_event:
                exit_reason = str(matched_event.get("reason") or "")
                exit_reason_source = f"{matched_event.get('source') or 'runtime_events'}:{matched_event.get('event_type') or ''}"
        if not exit_reason:
            lifecycle_candidates = [
                (exit_date, row_strategy, symbol),
                (exit_date, "", symbol),
                (str(row.get("session_date") or ""), row_strategy, symbol),
                (str(row.get("session_date") or ""), "", symbol),
            ]
            matched_event = choose_exit_reason_event(
                lifecycle_exit_reason_map,
                lifecycle_candidates,
                exit_time,
                order_id=sell_order_id,
                perm_id=sell_perm_id,
                execution_id=sell_execution_id,
            )
            if matched_event:
                exit_reason = str(matched_event.get("reason") or "")
                exit_reason_source = f"{matched_event.get('source') or 'trade_lifecycle/order_lifecycle'}:{matched_event.get('event_type') or ''}"
        if not exit_reason:
            exit_reason = "unknown_exit_reason"
            exit_reason_source = "missing_explicit_exit_event"
        persisted_commission = to_float(row.get("persisted_commission"), None)
        commission = abs(float(persisted_commission)) if persisted_commission is not None else 0.0
        raw_commission_status = str(raw.get("commission_status") or raw.get("commission_source") or "").upper()
        if raw_commission_status in {"OK", "PARTIAL", "MISSING"}:
            commission_status = raw_commission_status
        elif commission > 0:
            commission_status = "OK"
        else:
            commission_status = "MISSING"
        peak_pct, peak_source = peak_from_sources(enriched_row, runtime_peak_map, lifecycle_peak_map, candle_rows)
        if peak_pct is None:
            peak_fallback = trade_raw_peak_by_symbol_session.get(
                (
                    str(row.get("session_date") or ""),
                    str(row.get("strategy") or "unknown"),
                    str(row.get("symbol") or "").upper(),
                )
            )
            if peak_fallback is not None:
                peak_pct, peak_source = peak_fallback
        drop_from_peak = to_float(raw.get("drop_from_peak_pct"), None)
        if drop_from_peak is None:
            drop_from_peak = to_float(raw.get("giveback_pct"), None)
        commissions.append(commission)
        commission_statuses.append(commission_status)
        data_quality.append(quality_label(flags, commission_status))
        entry_times.append(entry_time)
        exit_times.append(exit_time)
        peak_values.append(peak_pct)
        peak_sources.append(peak_source)
        peak_match_qualities.append("exact_trade_id" if peak_source != "missing" else "missing")
        drop_values.append(drop_from_peak)
        entry_execution_counts.append(int(raw.get("entry_execution_count") or raw.get("buy_execution_count") or 0))
        exit_execution_counts.append(int(raw.get("exit_execution_count") or raw.get("sell_execution_count") or 0))
        confirmed_count = int(raw.get("confirmed_commission_execution_count") or (2 if commission_status == "OK" else (1 if commission_status == "PARTIAL" else 0)))
        confirmed_commission_counts.append(confirmed_count)
        expected_count = int(raw.get("expected_commission_execution_count") or raw.get("execution_count") or 0)
        expected_commission_counts.append(expected_count)
        commission_source_details.append(f"matched_by={matched_by} persisted_commission={commission} ibkr={confirmed_count}")
        entry_execution_ids.append(entry_execution_id)
        exit_execution_ids.append(exit_execution_id)
        trade_sources.append(trade_source)
        exit_reasons.append(exit_reason)
        exit_reason_sources.append(exit_reason_source)
        matched_event_types.append(str((matched_event or {}).get("event_type") or ""))
        matched_event_times.append(str((matched_event or {}).get("event_time") or ""))
        matched_order_ids.append(str((matched_event or {}).get("order_id") or sell_order_id or ""))
        strategy_values.append(row_strategy)
    out["ibkr_commission"] = commissions
    out["commission_status"] = commission_statuses
    out["data_quality"] = data_quality
    out["entry_time"] = entry_times
    out["exit_time"] = exit_times
    out["entry_date"] = [date_part(value) or str(session_date or "") for value, session_date in zip(out["entry_time"], out["session_date"])]
    out["exit_date"] = [date_part(exit_time) or date_part(closed_at) or str(session_date or "") for exit_time, closed_at, session_date in zip(out["exit_time"], out["closed_at"], out["session_date"])]
    carried_flags: list[bool] = []
    enriched_quality: list[str] = []
    runtime_pnl_trusted: list[bool] = []
    untrusted_reasons: list[str] = []
    for quality, entry_date, exit_date, source in zip(out["data_quality"], out["entry_date"], out["exit_date"], trade_sources):
        carried = bool(entry_date and exit_date and entry_date < exit_date and window.start_date <= exit_date <= window.end_date)
        reconstructed_source = str(source or "").lower() in {
            "sqlite_execution_reducer",
            "executions_pair_repair",
            "executions_pair",
            "reconstructed_executions",
            "carried_recovered",
        }
        untrusted_carry_basis = carried and reconstructed_source
        carried_flags.append(carried)
        if carried:
            quality = append_quality(quality, "CARRIED_POSITION_CLOSED_TODAY")
        if untrusted_carry_basis:
            quality = append_quality(quality, "CARRY_BASIS_UNVERIFIED")
        enriched_quality.append(str(quality or "OK"))
        runtime_pnl_trusted.append(not untrusted_carry_basis)
        untrusted_reasons.append("carry_basis_unverified" if untrusted_carry_basis else "")
    out["data_quality"] = enriched_quality
    out["carried_closed_today"] = carried_flags
    out["runtime_pnl_trusted"] = runtime_pnl_trusted
    out["runtime_pnl_untrusted_reason"] = untrusted_reasons
    out["peak_pct"] = peak_values
    for excursion_col in (
        "mae_pct",
        "peak_price",
        "low_price",
        "peak_unrealized_pnl",
        "max_adverse_unrealized_pnl",
        "giveback_from_peak",
    ):
        if excursion_col in out.columns:
            out[excursion_col] = pd.to_numeric(out[excursion_col], errors="coerce")
    out["peak_source"] = peak_sources
    out["peak_match_quality"] = peak_match_qualities
    out["drop_from_peak_pct"] = drop_values
    out["entry_execution_count"] = entry_execution_counts
    out["exit_execution_count"] = exit_execution_counts
    out["confirmed_commission_execution_count"] = confirmed_commission_counts
    out["expected_commission_execution_count"] = expected_commission_counts
    out["commission_source_detail"] = commission_source_details
    out["entry_execution_id"] = entry_execution_ids
    out["exit_execution_id"] = exit_execution_ids
    out["source"] = trade_sources
    out["exit_reason"] = exit_reasons
    out["exit_reason_source"] = exit_reason_sources
    out["matched_event_type"] = matched_event_types
    out["matched_event_time"] = matched_event_times
    out["matched_order_id"] = matched_order_ids
    out["strategy"] = strategy_values
    out["closed_source"] = "trades"
    for col in ("top100_rank", "top100_score", "live_entry_score", "live_entry_rank", "entry_order_id", "entry_perm_id"):
        values: list[float | None] = []
        for row in out.to_dict("records"):
            raw = parse_raw_json(row.get("raw_json"))
            if col in {"entry_order_id", "entry_perm_id"}:
                values.append(row.get(col) if row.get(col) not in (None, "") else raw.get(col))
            else:
                values.append(to_float(row.get(col), None) if row.get(col) is not None else to_float(raw.get(col), None))
        out[col] = values
    out["gross"] = pd.to_numeric(out["gross"], errors="coerce").fillna(0.0)
    out["buy"] = pd.to_numeric(out["buy"], errors="coerce").fillna(0.0)
    out["sell"] = pd.to_numeric(out["sell"], errors="coerce").fillna(0.0)
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0)
    out["peak_pct"] = pd.to_numeric(out["peak_pct"], errors="coerce")
    persisted_net = pd.to_numeric(out.get("persisted_net_pnl"), errors="coerce")
    out["net_actual"] = persisted_net.fillna(out["gross"] - out["ibkr_commission"])
    untrusted_mask = ~pd.Series(out["runtime_pnl_trusted"]).fillna(True).astype(bool)
    if untrusted_mask.any():
        out.loc[untrusted_mask, "raw_sqlite_gross"] = out.loc[untrusted_mask, "gross"]
        out.loc[untrusted_mask, "raw_sqlite_net_actual"] = out.loc[untrusted_mask, "net_actual"]
        out.loc[untrusted_mask, "gross"] = pd.NA
        out.loc[untrusted_mask, "net_actual"] = pd.NA
    denominator = (out["buy"] * out["qty"].abs()).replace(0, pd.NA)
    out["net_pct"] = ((out["net_actual"] / denominator) * 100.0).fillna(0.0)
    out["pnl_pct"] = out["net_pct"]
    fallback_drop = out["net_pct"].fillna(0.0) - out["peak_pct"]
    out["drop_from_peak_pct"] = pd.to_numeric(out["drop_from_peak_pct"], errors="coerce").fillna(fallback_drop)
    out["hold_minutes"] = [hold_minutes(a, b or c) for a, b, c in zip(out["entry_time"], out["exit_time"], out["closed_at"])]
    reconstructed_mask = (
        out["trade_id"].fillna("").astype(str).str.startswith("reconstructed:")
        | out["raw_json"].fillna("").astype(str).str.contains("sqlite_execution_reducer|executions_pair", regex=True)
    )
    out["_is_reconstructed_trade"] = reconstructed_mask.astype(int)
    exec_pair_key = out["entry_execution_id"].fillna("").astype(str) + "|" + out["exit_execution_id"].fillna("").astype(str)
    out["_closed_identity"] = out["trade_id"].fillna("").astype(str)
    out.loc[exec_pair_key.str.strip("|") != "", "_closed_identity"] = "execpair:" + exec_pair_key
    fallback_identity = (
        "fallback:"
        + out["exit_date"].fillna("").astype(str)
        + "|"
        + out["symbol"].fillna("").astype(str)
        + "|"
        + out["qty"].fillna(0).astype(str)
        + "|"
        + out["buy"].fillna(0).astype(str)
        + "|"
        + out["sell"].fillna(0).astype(str)
        + "|"
        + out["entry_time"].fillna("").astype(str)
        + "|"
        + out["exit_time"].fillna("").astype(str)
    )
    out.loc[out["_closed_identity"].str.strip() == "", "_closed_identity"] = fallback_identity
    out = (
        out.sort_values(["_is_reconstructed_trade", "updated_at"], na_position="last")
        .drop_duplicates(subset=["_closed_identity"], keep="first")
        .drop(columns=["_is_reconstructed_trade", "_closed_identity"])
    )
    return out[
        [
            "trade_id", "symbol", "qty", "ibkr_commission", "buy", "sell", "gross", "net_actual", "net_pct", "pnl_pct", "peak_pct",
            "mae_pct", "peak_price", "low_price", "peak_unrealized_pnl", "max_adverse_unrealized_pnl",
            "giveback_from_peak", "drop_from_peak_pct", "top100_rank", "top100_score",
            "live_entry_score", "live_entry_rank", "entry_order_id", "entry_perm_id",
            "hold_minutes", "exit_reason", "strategy",
            "entry_time", "exit_time", "commission_status", "data_quality", "session_date",
            "entry_date", "exit_date", "carried_closed_today",
            "runtime_pnl_trusted", "runtime_pnl_untrusted_reason",
            "entry_execution_count", "exit_execution_count", "confirmed_commission_execution_count",
            "expected_commission_execution_count", "peak_source", "peak_match_quality", "commission_source_detail",
            "entry_execution_id", "exit_execution_id", "exit_reason_source", "matched_event_type",
            "matched_event_time", "matched_order_id", "source", "closed_source", "updated_at", "trade_reduction_version",
        ]
    ]


def closed_from_executions(
    executions: pd.DataFrame,
    window: DateWindow,
    *,
    trade_entry_times: dict[tuple[str, str, str], Any] | None = None,
    position_entry_times: dict[tuple[str, str, str], Any] | None = None,
    runtime_entry_times: dict[tuple[str, str, str], Any] | None = None,
) -> pd.DataFrame:
    if executions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    trade_entry_times = trade_entry_times or {}
    position_entry_times = position_entry_times or {}
    runtime_entry_times = runtime_entry_times or {}
    for (strategy, symbol_group), group in executions.groupby(["strategy", "symbol"], dropna=False):
        fill_rows = group.to_dict("records")
        for row in reconstruct_closed_trades_from_fills(fill_rows):
            buy_execution = next((x for x in fill_rows if str(x.get("execution_id") or "") == str(row.get("buy_execution_id") or "")), {})
            sell_execution = next((x for x in fill_rows if str(x.get("execution_id") or "") == str(row.get("sell_execution_id") or "")), {})
            buy_time = execution_time_value(buy_execution) if buy_execution else None
            sell_time = execution_time_value(sell_execution) if sell_execution else None
            if sell_time:
                if not window_contains_date(window, sell_time):
                    continue
            elif not (window.start_date <= str(sell_execution.get("session_date") or buy_execution.get("session_date") or "") <= window.end_date):
                continue
            symbol = str(row.get("symbol") or symbol_group or "").upper()
            strategy_name = str(strategy or "unknown")
            buy_session_date = date_part(buy_time) or str(buy_execution.get("session_date") or "")
            exit_date = date_part(sell_time) or str(sell_execution.get("session_date") or "")
            recovered_entry_time = (
                lookup_entry_time(trade_entry_times, session_date=buy_session_date, strategy=strategy_name, symbol=symbol, before_or_on=exit_date)
                or lookup_entry_time(position_entry_times, session_date=buy_session_date, strategy=strategy_name, symbol=symbol, before_or_on=exit_date)
                or lookup_entry_time(runtime_entry_times, session_date=buy_session_date, strategy=strategy_name, symbol=symbol, before_or_on=exit_date)
            )
            flags = {"RECONSTRUCTED_FROM_EXECUTIONS"}
            closed_source = "reconstructed_executions"
            if recovered_entry_time and date_part(recovered_entry_time) and exit_date and date_part(recovered_entry_time) < exit_date:
                buy_time = recovered_entry_time
                flags.add("CARRIED_ENTRY_TIME_RECOVERED")
                closed_source = "carried_recovered"
            elif not buy_time:
                flags.add("CARRIED_ENTRY_TIME_MISSING")
            buy = to_float(row.get("buy"), 0.0) or 0.0
            sell = to_float(row.get("sell"), 0.0) or 0.0
            pnl_pct = ((sell / buy - 1.0) * 100.0) if buy and sell else 0.0
            raw_peak = row.get("peak_gain_pct")
            peak_pct = to_float(raw_peak, None)
            qty = to_float(row.get("qty"), 0.0) or 0.0
            gross = to_float(row.get("gross"), 0.0) or 0.0
            commission = abs(to_float(row.get("actual_commission"), 0.0) or 0.0)
            net = gross - commission
            denominator = abs(buy * qty)
            net_pct = (net / denominator * 100.0) if denominator else 0.0
            if not buy_time or not sell_time:
                flags.add("MISSING_EXECUTION_TIME")
            commission_status = "OK" if str(row.get("commission_source") or "").lower() == "ibkr" else ("PARTIAL" if commission else "MISSING")
            if commission_status == "PARTIAL":
                flags.add("COMMISSION_PARTIAL")
            elif commission_status == "MISSING":
                flags.add("COMMISSION_MISSING")
            rows.append({
                "symbol": symbol,
                "trade_id": row.get("trade_id") or "",
                "gross": gross,
                "ibkr_commission": commission,
                "commission_status": commission_status,
                "net_actual": net,
                "net_pct": net_pct,
                "pnl_pct": net_pct,
                "peak_pct": peak_pct,
                "drop_from_peak_pct": to_float(row.get("drop_from_peak_pct"), net_pct - peak_pct if peak_pct is not None else None),
                "hold_minutes": hold_minutes(buy_time, sell_time) if buy_time and sell_time else None,
                "exit_reason": row.get("reason") or "",
                "strategy": strategy_name,
                "entry_time": buy_time,
                "exit_time": sell_time,
                "entry_date": date_part(buy_time) or "",
                "exit_date": exit_date,
                "data_quality": quality_label_from_flags(flags),
                "entry_execution_count": 1,
                "exit_execution_count": 1,
                "confirmed_commission_execution_count": 2 if commission_status == "OK" else (1 if commission_status == "PARTIAL" else 0),
                "expected_commission_execution_count": 2,
                "peak_source": "fills_reconstruction" if raw_peak is not None else "missing",
                "peak_match_quality": "exact_trade_id" if raw_peak is not None else "missing",
                "commission_source_detail": "reconstructed",
                "closed_source": closed_source,
                "qty": qty,
                "buy": buy,
                "sell": sell,
                "session_date": buy_session_date or exit_date,
            })
    return pd.DataFrame(rows)


def append_quality(existing: Any, flag: str) -> str:
    parts = {x.strip() for x in str(existing or "").split(";") if x.strip() and x.strip() != "OK"}
    parts.add(flag)
    return "; ".join(sorted(parts)) if parts else "OK"


def apply_symbol_session_peak_fallbacks(
    closed: pd.DataFrame,
    runtime_symbol_peak_map: dict[tuple[str, str, str], dict[str, Any]],
    lifecycle_symbol_peak_map: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    if closed.empty or "peak_pct" not in closed.columns:
        return closed
    out = closed.copy()
    symbol_counts = out.groupby(["session_date", "symbol"], dropna=False).size().to_dict()
    for idx, row in out.iterrows():
        if pd.notna(row.get("peak_pct")):
            out.at[idx, "peak_match_quality"] = out.at[idx, "peak_match_quality"] if "peak_match_quality" in out.columns and pd.notna(out.at[idx, "peak_match_quality"]) else "exact_trade_id"
            continue
        session_date = str(row.get("session_date") or "")
        symbol = str(row.get("symbol") or "").upper()
        if not session_date or not symbol:
            out.at[idx, "peak_match_quality"] = "missing"
            continue
        has_window = parse_dt(row.get("entry_time")) is not None or parse_dt(row.get("exit_time")) is not None
        is_unique = symbol_counts.get((session_date, row.get("symbol")), 0) == 1 or symbol_counts.get((session_date, symbol), 0) == 1
        if not is_unique and not has_window:
            out.at[idx, "peak_match_quality"] = "ambiguous"
            out.at[idx, "data_quality"] = append_quality(row.get("data_quality"), "AMBIGUOUS_PEAK_MATCH")
            continue
        strategy = str(row.get("strategy") or "unknown")
        match = (
            runtime_symbol_peak_map.get((session_date, strategy, symbol))
            or runtime_symbol_peak_map.get((session_date, "", symbol))
        )
        if match is None:
            match = lifecycle_symbol_peak_map.get((session_date, symbol))
        if match is None or match.get("peak_pct") is None:
            out.at[idx, "peak_match_quality"] = "missing"
            continue
        peak_pct = float(match["peak_pct"])
        out.at[idx, "peak_pct"] = peak_pct
        out.at[idx, "peak_source"] = match.get("source") or "runtime_events_symbol_session"
        out.at[idx, "peak_match_quality"] = "symbol_session_unique" if is_unique else "time_window"
        drop_pct = match.get("drop_from_peak_pct")
        if drop_pct is None:
            sell = to_float(row.get("sell"), None)
            peak_price = to_float(match.get("peak_price"), None)
            drop_pct = pct_from_prices(sell, peak_price)
        if drop_pct is not None:
            out.at[idx, "drop_from_peak_pct"] = float(drop_pct)
        elif pd.notna(row.get("net_pct")):
            out.at[idx, "drop_from_peak_pct"] = float(row.get("net_pct")) - peak_pct
    return out


def load_closed_positions(
    conn: sqlite3.Connection,
    window: DateWindow,
    strategy: str | None,
    executions: pd.DataFrame,
    *,
    include_reconstructed: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    lookup_window = expanded_lookup_window(window)
    runtime_peak_map = load_runtime_peak_map(conn, window, strategy)
    runtime_symbol_peak_map = load_runtime_symbol_peak_map(conn, window, strategy)
    lifecycle_peak_map = load_lifecycle_peak_map(window)
    lifecycle_symbol_peak_map = load_lifecycle_symbol_peak_map(window)
    candle_rows: dict[tuple[str, str], pd.DataFrame] = {}
    trades = closed_from_trades(
        conn,
        window,
        strategy,
        executions,
        runtime_peak_map,
        lifecycle_peak_map,
        candle_rows,
        runtime_exit_reason_map={},
        lifecycle_exit_reason_map={},
    )
    reconstructed = pd.DataFrame()
    reconstructed_count = 0
    if include_reconstructed:
        reconstructed = closed_from_executions(
            executions,
            window,
            trade_entry_times=trade_entry_map(conn, lookup_window, strategy),
            position_entry_times=position_entry_map(conn, lookup_window, strategy),
            runtime_entry_times=runtime_entry_event_map(conn, lookup_window, strategy),
        )
        reconstructed_count = int(len(reconstructed))
    if include_reconstructed and not trades.empty and not reconstructed.empty:
        trade_keys = {
            (str(row.get("exit_date") or date_part(row.get("exit_time")) or row.get("session_date") or ""), str(row.get("symbol") or "").upper())
            for row in trades.to_dict("records")
        }
        reconstructed = reconstructed[
            ~reconstructed.apply(
                lambda row: (str(row.get("exit_date") or date_part(row.get("exit_time")) or row.get("session_date") or ""), str(row.get("symbol") or "").upper()) in trade_keys,
                axis=1,
            )
        ]
    persisted_count = int(len(trades))
    displayed_frames = [trades]
    if include_reconstructed:
        displayed_frames.append(reconstructed)
    frames = [df for df in displayed_frames if not df.empty]
    diag = {
        "persisted_closed_trades_count": persisted_count,
        "reconstructed_execution_pairs_count": reconstructed_count,
        "displayed_closed_trades_count": 0,
        "execution_reconstruction_disabled": int(not include_reconstructed and persisted_count == 0 and not executions.empty),
        "runtime_closed_source": "persisted_trades",
    }
    if not frames:
        return pd.DataFrame(), diag
    closed = pd.concat(frames, ignore_index=True, sort=False)
    closed = apply_symbol_session_peak_fallbacks(closed, runtime_symbol_peak_map, lifecycle_symbol_peak_map)
    closed = closed.sort_values(["net_actual", "symbol"], na_position="last").reset_index(drop=True)
    diag["displayed_closed_trades_count"] = int(len(closed))
    return closed, diag


def execution_net_positions(executions: pd.DataFrame | None) -> dict[tuple[str, str, str], float]:
    if executions is None or executions.empty:
        return {}
    net: dict[tuple[str, str, str], float] = {}
    for row in executions.to_dict("records"):
        session_date = str(row.get("session_date") or "")
        strategy = str(row.get("strategy") or "unknown")
        symbol = str(row.get("symbol") or "").upper()
        if not session_date or not symbol:
            continue
        qty = abs(to_float(row.get("quantity"), 0.0) or 0.0)
        action = str(row.get("action") or "").upper()
        if action in {"BOT", "BUY"}:
            signed_qty = qty
        elif action in {"SLD", "SELL"}:
            signed_qty = -qty
        else:
            signed_qty = to_float(row.get("quantity"), 0.0) or 0.0
        key = (session_date, strategy, symbol)
        net[key] = net.get(key, 0.0) + signed_qty
    return net


def entry_execution_map(executions: pd.DataFrame | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if executions is None or executions.empty:
        return {}
    rows = executions.copy()
    rows["action"] = rows.get("action", "").fillna("").astype(str).str.upper()
    rows = rows[rows["action"].isin({"BOT", "BUY"})]
    if rows.empty:
        return {}
    rows["_time"] = rows.apply(lambda row: execution_time_value(row.to_dict()) or row.get("recorded_at"), axis=1)
    rows["_commission"] = pd.to_numeric(rows.get("commission"), errors="coerce")
    rows["_commission_source"] = rows.get("commission_source", "").fillna("").astype(str).str.lower()
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (session_date, strategy, symbol), group in rows.groupby(["session_date", "strategy", "symbol"], dropna=False):
        sorted_group = group.sort_values("_time", na_position="last")
        first = sorted_group.iloc[0].to_dict()
        commissions = sorted_group[
            (sorted_group["_commission_source"] == "ibkr")
            & sorted_group["_commission"].notna()
        ]["_commission"].abs().sum()
        out[(str(session_date), str(strategy or "unknown"), str(symbol).upper())] = {
            "entry_time": first.get("_time"),
            "entry_commission": float(commissions or 0.0),
        }
    return out


def trade_entry_map(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[tuple[str, str, str], Any]:
    clause, params = strategy_clause("t", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            session_date,
            COALESCE(strategy_name, 'unknown') AS strategy,
            symbol,
            entry_fill_time,
            entry_signal_time,
            entry_order_time
        FROM trades t
        WHERE t.session_date BETWEEN ? AND ? {clause}
        ORDER BY COALESCE(t.entry_fill_time, t.entry_order_time, t.entry_signal_time)
        """,
        [window.start_date, window.end_date, *params],
    )
    out: dict[tuple[str, str, str], Any] = {}
    for row in rows.to_dict("records"):
        key = (str(row.get("session_date") or ""), str(row.get("strategy") or "unknown"), str(row.get("symbol") or "").upper())
        out.setdefault(key, row.get("entry_fill_time") or row.get("entry_order_time") or row.get("entry_signal_time"))
    return out


def runtime_entry_event_map(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[tuple[str, str, str], Any]:
    clause, params = strategy_clause("r", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            COALESCE(r.session_date, substr(r.event_time, 1, 10)) AS session_date,
            COALESCE(r.strategy_name, 'unknown') AS strategy,
            r.symbol,
            r.event_time
        FROM runtime_events r
        WHERE COALESCE(r.session_date, substr(r.event_time, 1, 10)) BETWEEN ? AND ? {clause}
          AND r.event_type IN ('BUY_ORDER_SENT', 'ENTRY_ORDER_SUBMITTED', 'ENTRY_ORDER_FILLED', 'POSITION_OPENED')
        ORDER BY r.event_time
        """,
        [window.start_date, window.end_date, *params],
    )
    out: dict[tuple[str, str, str], Any] = {}
    for row in rows.to_dict("records"):
        key = (str(row.get("session_date") or ""), str(row.get("strategy") or "unknown"), str(row.get("symbol") or "").upper())
        out.setdefault(key, row.get("event_time"))
    return out


def position_entry_map(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[tuple[str, str, str], Any]:
    clause, params = strategy_clause("p", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            p.session_date,
            COALESCE(p.strategy_name, 'unknown') AS strategy,
            p.symbol,
            p.updated_at,
            p.raw_json
        FROM positions p
        WHERE p.session_date BETWEEN ? AND ? {clause}
        ORDER BY COALESCE(p.updated_at, '') ASC
        """,
        [window.start_date, window.end_date, *params],
    )
    out: dict[tuple[str, str, str], Any] = {}
    for row in rows.to_dict("records"):
        raw = parse_raw_json(row.get("raw_json"))
        raw_entry = raw.get("entry_time") or raw.get("buy_time") or raw.get("adopted_on_restart")
        entry_time = displayable_entry_time(raw_entry)
        if not entry_time:
            continue
        key = (str(row.get("session_date") or ""), str(row.get("strategy") or "unknown"), str(row.get("symbol") or "").upper())
        out.setdefault(key, entry_time)
    return out


def lookup_entry_time(
    mapping: dict[tuple[str, str, str], Any],
    *,
    session_date: str,
    strategy: str,
    symbol: str,
    before_or_on: str | None = None,
) -> Any:
    exact = mapping.get((session_date, strategy, symbol))
    if exact is not None:
        return exact
    candidates: list[tuple[str, Any]] = []
    for (date_value, strategy_value, symbol_value), value in mapping.items():
        if symbol_value != symbol:
            continue
        if strategy_value not in {strategy, "unknown", ""} and strategy not in {"unknown", ""}:
            continue
        if before_or_on and date_value > before_or_on:
            continue
        candidates.append((date_value, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def raw_now_price(raw: dict[str, Any]) -> tuple[float | None, str, Any]:
    market_price = to_float(raw.get("market_price"), None)
    if market_price is not None:
        return (
            market_price,
            str(raw.get("market_price_source") or "live_quote"),
            raw.get("market_price_at") or raw.get("price_time") or raw.get("quote_time") or raw.get("updated_at"),
        )
    live_keys = ("current_price", "last_price", "last", "bid_ask_mid", "mid_price")
    for key in live_keys:
        value = to_float(raw.get(key), None)
        if value is not None:
            return value, "live_quote", raw.get("price_time") or raw.get("quote_time") or raw.get("updated_at")
    portfolio_keys = ("portfolio_market_price", "portfolio_last_price", "ibkr_market_price")
    for key in portfolio_keys:
        value = to_float(raw.get(key), None)
        if value is not None:
            return value, "portfolio_snapshot", raw.get("portfolio_time") or raw.get("updated_at")
    candle_keys = ("latest_candle_close", "last_candle_close")
    for key in candle_keys:
        value = to_float(raw.get(key), None)
        if value is not None:
            return value, "latest_candle", raw.get("latest_candle_time") or raw.get("updated_at")
    return None, "missing", None


def lookup_by_position_key(mapping: dict[tuple[str, str, str], Any], session_date: str, strategy: str, symbol: str) -> Any:
    exact = mapping.get((session_date, strategy, symbol))
    if exact is not None:
        return exact
    matches = [
        (date_value, value)
        for (date_value, strategy_value, symbol_value), value in mapping.items()
        if strategy_value == strategy and symbol_value == symbol
    ]
    if not matches:
        matches = [
            (date_value, value)
            for (date_value, _strategy_value, symbol_value), value in mapping.items()
            if symbol_value == symbol
        ]
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def price_status_for(now: float | None, price_time: Any, window: DateWindow, stale_seconds: int = 900) -> str:
    if now is None:
        return "MISSING_PRICE"
    price_dt = parse_dt(price_time)
    if window.end_date == utc_today() and price_dt is not None:
        age_seconds = (datetime.now(timezone.utc) - price_dt).total_seconds()
        if age_seconds > stale_seconds:
            return "STALE"
    return "OK"


def broker_portfolio_by_symbol(broker_portfolio: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if broker_portfolio is None or broker_portfolio.empty or "symbol" not in broker_portfolio.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in broker_portfolio.to_dict("records"):
        symbol = str(row.get("symbol") or "").upper()
        if symbol:
            out[symbol] = row
    return out


def load_open_positions(
    conn: sqlite3.Connection,
    window: DateWindow,
    strategy: str | None,
    executions: pd.DataFrame | None = None,
    execution_lookup: pd.DataFrame | None = None,
    broker_portfolio: pd.DataFrame | None = None,
) -> pd.DataFrame:
    clause, params = strategy_clause("p", strategy)
    broker_rows = broker_portfolio_by_symbol(broker_portfolio)
    rows = read_sql(
        conn,
        f"""
        SELECT
            COALESCE(p.strategy_name, 'unknown') AS strategy,
            p.session_date,
            p.position_key,
            p.symbol,
            p.status,
            p.quantity,
            p.avg_price,
            p.ibkr_quantity,
            p.ibkr_avg_cost,
            p.active,
            p.exit_sent,
            p.top100_rank,
            p.top100_score,
            p.top100_source_date,
            p.top100_features_json,
            p.live_entry_score,
            p.live_entry_rank,
            p.live_entry_features_json,
            p.signal_source,
            p.signal_time,
            p.ready_since,
            p.entry_order_id,
            p.entry_perm_id,
            p.updated_at,
            p.source,
            p.raw_json
        FROM positions p
        WHERE COALESCE(p.active, 0) = 1 {clause}
          AND UPPER(COALESCE(p.status, '')) != 'ORPHAN_STALE_POSITION'
          AND UPPER(COALESCE(p.status, '')) NOT LIKE '%ORPHAN_STALE_POSITION%'
        ORDER BY COALESCE(p.updated_at, '') DESC, UPPER(p.symbol), p.rowid DESC
        """,
        params,
    )
    if rows.empty:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        row_strategy = str(row.get("strategy") or "unknown")
        symbol = str(row.get("symbol") or "").upper()
        session_date = str(row.get("session_date") or "")
        raw = parse_raw_json(row.get("raw_json"))
        status = str(row.get("status") or "").upper()
        if "ORPHAN_STALE_POSITION" in status:
            continue
        ibkr_qty = to_float(row.get("ibkr_quantity"), None)
        qty = to_float(row.get("quantity"), None)
        if qty is None:
            qty = to_float(row.get("ibkr_quantity"), 0.0)
        if qty is None:
            qty = 0.0
        buy = to_float(row.get("avg_price"), None)
        if buy is None:
            buy = to_float(raw.get("entry_price"), None)
        broker_row = broker_rows.get(symbol)
        broker_now = to_float(broker_row.get("market_price"), None) if broker_row else None
        broker_upnl = to_float(broker_row.get("unrealized_pnl"), None) if broker_row else None
        now = broker_now
        now_price_source = "broker_portfolio.market_price" if now is not None else ""
        price_time = broker_row.get("last_refresh_time") if broker_row else None
        if now is None:
            now = to_float(raw.get("market_price"), None)
            now_price_source = str(raw.get("market_price_source") or ("positions.raw_json.market_price" if now is not None else "missing"))
            price_time = raw.get("market_price_at")
        price_status = price_status_for(now, price_time, window)
        if now is None or not buy:
            upnl = None
            now_pct = None
        else:
            upnl = broker_upnl if broker_upnl is not None else (now - buy) * (qty or 0.0)
            now_pct = ((now - buy) / buy) * 100.0
        peak_pct = to_float(raw.get("peak_pct"), None)
        if peak_pct is None:
            peak_pct = to_float(raw.get("peak_unrealized_pct"), None)
        if peak_pct is None:
            peak_pct = to_float(raw.get("peak_gain_pct"), None)
        if peak_pct is None and buy:
            peak_price = to_float(raw.get("peak_price"), None)
            if peak_price is not None:
                peak_pct = ((peak_price / buy) - 1.0) * 100.0
        raw_entry_time = raw.get("entry_time")
        adopted_time = adopted_timestamp_value(raw_entry_time)
        entry_time = displayable_entry_time(raw_entry_time)
        entry_date = date_part(entry_time) or str(session_date or "")
        stale_carry = bool(entry_date and entry_date < window.end_date)
        if ibkr_qty is None:
            ibkr_confirmed: str | bool = "UNKNOWN"
        else:
            ibkr_confirmed = "TRUE" if abs((ibkr_qty or 0.0) - (qty or 0.0)) <= 1e-6 else "FALSE"
        data_quality = str(raw.get("data_quality") or "OK")
        position_bucket = "today"
        display_status = row.get("status") or ("EXIT_ORDER" if row.get("exit_sent") else "OPEN")
        if stale_carry:
            position_bucket = "carry_stale"
        entry_source = "ADOPTED" if adopted_time else ("position_raw" if raw_entry_time else "missing")
        execution_ids = raw.get("open_lot_execution_ids")
        if isinstance(execution_ids, list):
            execution_ids_value = ", ".join(str(x) for x in execution_ids)
        elif execution_ids is None:
            execution_ids_value = ""
        else:
            execution_ids_value = str(execution_ids)
        hold_base = entry_time
        out.append({
            "symbol": row.get("symbol"),
            "qty": qty or 0.0,
            "buy": buy,
            "now": now,
            "upnl": upnl,
            "now_dollars": upnl,
            "now_pct": now_pct,
            "peak_pct": peak_pct,
            "giveback_pct": (peak_pct - now_pct) if now_pct is not None and peak_pct is not None else None,
            "hold_minutes": hold_minutes(hold_base),
            "status": display_status,
            "strategy": row_strategy,
            "entry_time": entry_time,
            "entry_date": entry_date,
            "entry_source": entry_source,
            "session_date": session_date,
            "position_key": row.get("position_key"),
            "source": row.get("source"),
            "ibkr_confirmed": ibkr_confirmed,
            "data_quality": data_quality,
            "position_bucket": position_bucket,
            "price_status": price_status,
            "now_price_source": now_price_source,
            "market_price_at": price_time,
            "broker_market_price": broker_now,
            "broker_unrealized_pnl": broker_upnl,
            "dashboard_broker_price_diff": (now - broker_now) if now is not None and broker_now is not None else None,
            "last_update": row.get("updated_at"),
            "exit_sent": row.get("exit_sent"),
            "execution_ids": execution_ids_value,
            "top100_rank": to_float(row.get("top100_rank"), None) if row.get("top100_rank") is not None else to_float(raw.get("top100_rank"), None),
            "top100_score": to_float(row.get("top100_score"), None) if row.get("top100_score") is not None else to_float(raw.get("top100_score"), None),
            "live_entry_score": to_float(row.get("live_entry_score"), None) if row.get("live_entry_score") is not None else to_float(raw.get("live_entry_score"), None),
            "live_entry_rank": to_float(row.get("live_entry_rank"), None) if row.get("live_entry_rank") is not None else to_float(raw.get("live_entry_rank"), None),
            "entry_order_id": row.get("entry_order_id") if row.get("entry_order_id") not in (None, "") else raw.get("entry_order_id"),
            "entry_perm_id": row.get("entry_perm_id") if row.get("entry_perm_id") not in (None, "") else raw.get("entry_perm_id"),
        })
    return pd.DataFrame(out)


def load_raw_closed_trade_rows(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> pd.DataFrame:
    clause, params = strategy_clause("t", strategy)
    return read_sql(
        conn,
        f"""
        SELECT
            t.trade_id,
            t.symbol,
            t.quantity,
            t.status,
            t.session_date,
            t.entry_fill_time,
            t.exit_fill_time,
            t.closed_at,
            COALESCE(t.strategy_name, 'unknown') AS strategy,
            t.updated_at,
            t.raw_json
        FROM trades t
        WHERE (
            substr(t.exit_fill_time, 1, 10) BETWEEN ? AND ?
            OR substr(t.closed_at, 1, 10) BETWEEN ? AND ?
            OR (
                COALESCE(t.exit_fill_time, t.closed_at) IS NULL
                AND t.session_date BETWEEN ? AND ?
            )
        ) {clause}
          AND UPPER(COALESCE(t.status, '')) IN ({",".join("?" for _ in CLOSED_STATUSES)})
        ORDER BY COALESCE(t.closed_at, t.exit_fill_time), t.symbol, t.trade_id
        """,
        [window.start_date, window.end_date, window.start_date, window.end_date, window.start_date, window.end_date, *params, *sorted(CLOSED_STATUSES)],
    )


def load_pending_trades(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> pd.DataFrame:
    clause, params = strategy_clause("t", strategy)
    return read_sql(
        conn,
        f"""
        SELECT
            trade_id,
            COALESCE(strategy_name, 'unknown') AS strategy,
            session_date,
            symbol,
            status,
            entry_fill_time AS entry_time,
            exit_fill_time AS exit_time,
            closed_at,
            entry_price AS buy,
            exit_price AS sell,
            quantity AS qty,
            gross_pnl AS gross,
            commission,
            net_pnl,
            raw_json,
            updated_at,
            trade_reduction_version
        FROM trades t
        WHERE (
            substr(t.exit_fill_time, 1, 10) BETWEEN ? AND ?
            OR substr(t.closed_at, 1, 10) BETWEEN ? AND ?
            OR (
                COALESCE(t.exit_fill_time, t.closed_at) IS NULL
                AND t.session_date BETWEEN ? AND ?
            )
        ) {clause}
          AND UPPER(COALESCE(t.status, '')) IN ({",".join("?" for _ in PENDING_TRADE_STATUSES)})
        ORDER BY COALESCE(t.updated_at, t.closed_at, t.exit_fill_time, t.entry_fill_time) DESC, t.symbol, t.trade_id
        """,
        [window.start_date, window.end_date, window.start_date, window.end_date, window.start_date, window.end_date, *params, *sorted(PENDING_TRADE_STATUSES)],
    )


def load_excluded_open_positions(
    conn: sqlite3.Connection,
    window: DateWindow,
    strategy: str | None,
    displayed_open_positions: pd.DataFrame,
    orphan_stale_positions: pd.DataFrame,
) -> pd.DataFrame:
    rows = load_raw_active_positions(conn, strategy)
    if rows.empty:
        return pd.DataFrame(columns=[
            "symbol", "position_key", "session_date", "status", "quantity", "ibkr_quantity",
            "updated_at", "source", "exclusion_reason",
        ])
    displayed_keys = set()
    if not displayed_open_positions.empty and "position_key" in displayed_open_positions.columns:
        displayed_keys = {str(x) for x in displayed_open_positions["position_key"].dropna().tolist()}
    orphan_keys = set()
    if not orphan_stale_positions.empty and "position_key" in orphan_stale_positions.columns:
        orphan_keys = {str(x) for x in orphan_stale_positions["position_key"].dropna().tolist()}
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        key = str(row.get("position_key") or "")
        if key and key in displayed_keys:
            continue
        reason = "FILTERED_BY_OPEN_POSITION_RULES"
        if key and key in orphan_keys:
            reason = "ORPHAN_STALE_POSITION"
        elif "ORPHAN_STALE_POSITION" in str(row.get("status") or "").upper():
            reason = "ORPHAN_STALE_POSITION"
        out.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "position_key": row.get("position_key"),
                "session_date": row.get("session_date"),
                "status": row.get("status"),
                "quantity": row.get("quantity"),
                "ibkr_quantity": row.get("ibkr_quantity"),
                "updated_at": row.get("updated_at"),
                "source": row.get("source"),
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(out)


def load_raw_active_positions(conn: sqlite3.Connection, strategy: str | None) -> pd.DataFrame:
    clause, params = strategy_clause("p", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            COALESCE(p.strategy_name, 'unknown') AS strategy,
            p.session_date,
            p.position_key,
            p.symbol,
            p.status,
            p.quantity,
            p.avg_price,
            p.ibkr_quantity,
            p.active,
            p.updated_at,
            p.source,
            p.raw_json
        FROM positions p
        WHERE COALESCE(p.active, 0) = 1 {clause}
        ORDER BY COALESCE(p.updated_at, '') DESC, UPPER(p.symbol), p.rowid DESC
        """,
        params,
    )
    if rows.empty:
        return pd.DataFrame(columns=[
            "symbol", "quantity", "avg_price", "status", "active", "updated_at",
            "entry_time", "session_date", "strategy", "source", "position_key",
        ])
    out = rows.copy()
    entry_times: list[Any] = []
    for row in out.to_dict("records"):
        raw = parse_raw_json(row.get("raw_json"))
        entry_times.append(displayable_entry_time(raw.get("entry_time") or raw.get("buy_time")) or row.get("updated_at") or row.get("session_date"))
    out["entry_time"] = entry_times
    return out[
        [
            "symbol", "quantity", "avg_price", "status", "active", "updated_at",
            "entry_time", "session_date", "strategy", "source", "position_key",
        ]
    ]


def load_orphan_stale_positions(
    conn: sqlite3.Connection,
    window: DateWindow,
    strategy: str | None,
    *,
    stale_days: int = DEFAULT_ORPHAN_STALE_DAYS,
) -> pd.DataFrame:
    clause, params = strategy_clause("p", strategy)
    rows = read_sql(
        conn,
        f"""
        WITH active_candidates AS (
            SELECT
                p.*,
                p.rowid AS position_rowid,
                ROW_NUMBER() OVER (
                    PARTITION BY UPPER(p.symbol)
                    ORDER BY COALESCE(p.updated_at, '') DESC, COALESCE(p.session_date, '') DESC, p.rowid DESC
                ) AS rn
            FROM positions p
            WHERE COALESCE(p.session_date, '') <= ? {clause}
              AND COALESCE(p.active, 0) = 1
              AND {OPEN_POSITION_STATUS_SQL}
              AND COALESCE(p.raw_json, '') NOT LIKE '%"active": false%'
              AND COALESCE(p.raw_json, '') NOT LIKE '%"ibkr_position_flat_confirmed": true%'
        )
        SELECT
            COALESCE(strategy_name, 'unknown') AS strategy,
            session_date,
            position_key,
            symbol,
            status,
            quantity,
            avg_price,
            ibkr_quantity,
            updated_at,
            source,
            raw_json
        FROM active_candidates p
        WHERE p.rn = 1
        ORDER BY p.symbol
        """,
        [window.end_date, *params],
    )
    if rows.empty:
        return pd.DataFrame(columns=[
            "symbol", "qty", "buy", "entry_time", "entry_date", "age_days", "status",
            "strategy", "position_key", "source", "data_quality", "cleanup_recommendation",
        ])
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        raw = parse_raw_json(row.get("raw_json"))
        symbol = str(row.get("symbol") or "").upper()
        row_strategy = str(row.get("strategy") or "unknown")
        raw_entry_time = raw.get("entry_time") or raw.get("buy_time")
        entry_time = displayable_entry_time(raw_entry_time) or str(row.get("session_date") or "")
        entry_date = date_part(entry_time) or str(row.get("session_date") or "")
        if not entry_date or entry_date >= window.end_date:
            continue
        ibkr_qty = to_float(row.get("ibkr_quantity"), None)
        ibkr_confirmed = bool(
            (ibkr_qty is not None and abs(ibkr_qty) > 1e-9)
            or raw_bool(raw, "ibkr_confirmed", "ibkr_entry_confirmed", "entry_fill_verified", "ibkr_position_confirmed")
        )
        days = age_days(entry_time, f"{window.end_date}T23:59:59+00:00")
        if ibkr_confirmed or days is None or days <= stale_days:
            continue
        qty = to_float(row.get("quantity"), None)
        out.append(
            {
                "symbol": symbol,
                "qty": qty if qty is not None else to_float(row.get("ibkr_quantity"), 0.0),
                "buy": to_float(row.get("avg_price") or raw.get("entry_price"), None),
                "entry_time": entry_time,
                "entry_date": entry_date,
                "age_days": days,
                "status": f"{row.get('status') or 'OPEN'}|ORPHAN_STALE_POSITION",
                "strategy": row_strategy,
                "position_key": row.get("position_key"),
                "source": row.get("source"),
                "data_quality": "STALE_CARRY_OPEN;IBKR_UNCONFIRMED;ORPHAN_STALE_POSITION",
                "cleanup_recommendation": "Close stale orphan position",
            }
        )
    return pd.DataFrame(out)


def load_rejected_entries(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> pd.DataFrame:
    clause, params = strategy_clause("r", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            r.event_time,
            COALESCE(r.strategy_name, 'unknown') AS strategy,
            r.session_date,
            r.symbol,
            r.order_id,
            r.reason,
            r.raw_json
        FROM runtime_events r
        WHERE COALESCE(r.session_date, substr(r.event_time, 1, 10)) BETWEEN ? AND ?
          AND r.event_type = 'ENTRY_ORDER_REJECTED'
          {clause}
        ORDER BY r.event_time DESC, r.symbol
        """,
        [window.start_date, window.end_date, *params],
    )
    if rows.empty:
        return pd.DataFrame(columns=["symbol", "qty", "price", "order_id", "reason", "ibkr_error_code", "rejected_at", "strategy"])
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        raw = parse_raw_json(row.get("raw_json"))
        out.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "qty": to_float(raw.get("quantity"), None),
                "price": to_float(raw.get("price"), None),
                "order_id": row.get("order_id") or raw.get("order_id"),
                "reason": row.get("reason") or raw.get("reject_reason") or raw.get("reason"),
                "ibkr_error_code": raw.get("ibkr_error_code"),
                "rejected_at": row.get("event_time"),
                "strategy": row.get("strategy"),
            }
        )
    return pd.DataFrame(out)


def load_diagnostics(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[str, int]:
    event_clause, event_params = strategy_clause("r", strategy)
    risk_clause, risk_params = strategy_clause("q", strategy)
    trade_clause, trade_params = strategy_clause("t", strategy)
    trade_cols = table_columns(conn, "trades")
    trade_updated_at_expr = "t.updated_at" if "updated_at" in trade_cols else "NULL"
    trades_updated_last_60s_expr = (
        "SUM(CASE WHEN t.updated_at IS NOT NULL AND unixepoch('now') - unixepoch(t.updated_at) BETWEEN 0 AND 60 THEN 1 ELSE 0 END)"
        if "updated_at" in trade_cols
        else "0"
    )
    events = read_sql(
        conn,
        f"""
        SELECT event_type, reason, COUNT(*) AS count
        FROM runtime_events r
        WHERE COALESCE(r.session_date, substr(r.event_time, 1, 10)) BETWEEN ? AND ? {event_clause}
        GROUP BY event_type, reason
        """,
        [window.start_date, window.end_date, *event_params],
    )
    risks = read_sql(
        conn,
        f"""
        SELECT event_type, reason, SUM(COALESCE(repeat_count, 1)) AS count
        FROM risk_events q
        WHERE COALESCE(q.session_date, substr(q.event_time, 1, 10)) BETWEEN ? AND ? {risk_clause}
        GROUP BY event_type, reason
        """,
        [window.start_date, window.end_date, *risk_params],
    )
    rec = read_sql(
        conn,
        """
        SELECT *
        FROM reconciliation_runs
        WHERE COALESCE(substr(finished_at, 1, 10), substr(started_at, 1, 10)) BETWEEN ? AND ?
        ORDER BY COALESCE(finished_at, started_at) DESC
        LIMIT 1
        """,
        [window.start_date, window.end_date],
    )
    trade_diag = read_sql(
        conn,
        f"""
        SELECT
            COUNT(*) AS trades_count,
            SUM(CASE
                WHEN trade_id LIKE 'reconstructed:%'
                  OR raw_json LIKE '%sqlite_execution_reducer%'
                  OR raw_json LIKE '%executions_pair%'
                THEN 1 ELSE 0 END
            ) AS reconstructed_trades_count,
            {trades_updated_last_60s_expr} AS trades_updated_last_60s,
            MAX({trade_updated_at_expr}) AS last_reducer_run_at
        FROM trades t
        WHERE (
            substr(t.exit_fill_time, 1, 10) BETWEEN ? AND ?
            OR substr(t.closed_at, 1, 10) BETWEEN ? AND ?
            OR (
                COALESCE(t.exit_fill_time, t.closed_at) IS NULL
                AND t.session_date BETWEEN ? AND ?
            )
        ) {trade_clause}
        """,
        [window.start_date, window.end_date, window.start_date, window.end_date, window.start_date, window.end_date, *trade_params],
    )
    runtime_state = pd.DataFrame()
    if table_columns(conn, "runtime_state"):
        runtime_state = read_sql(
            conn,
            """
            SELECT key, value, updated_at, raw_json
            FROM runtime_state
            WHERE key IN ('fill_ingest', 'position_reconcile')
            """,
        )
    pending_counts = pd.DataFrame()
    if table_columns(conn, "executions"):
        pending_counts = read_sql(
            conn,
            """
            SELECT
                COUNT(*) AS execution_count,
                SUM(CASE
                    WHEN COALESCE(commission_source, '') != 'ibkr'
                         OR commission IS NULL
                    THEN 1 ELSE 0 END
                ) AS pending_commission_count,
                SUM(CASE
                    WHEN UPPER(COALESCE(side, '')) IN ('SLD', 'SELL')
                         AND COALESCE(commission_source, '') IN ('ibkr', 'missing')
                         AND realized_pnl IS NULL
                    THEN 1 ELSE 0 END
                ) AS pending_realized_pnl_count
            FROM executions
            WHERE COALESCE(substr(executed_at, 1, 10), substr(recorded_at, 1, 10), session_date) BETWEEN ? AND ?
            """,
            [window.start_date, window.end_date],
        )
    pending_trades = pd.DataFrame()
    pending_trade_cols = table_columns(conn, "trades")
    if "status" in pending_trade_cols:
        pending_trades = read_sql(
            conn,
            """
            SELECT COUNT(*) AS pending_trade_finalization_count
            FROM trades
            WHERE UPPER(COALESCE(status, '')) IN ('COMMISSION_PENDING', 'PNL_PENDING', 'RECONCILE_PENDING')
              AND (
                substr(exit_fill_time, 1, 10) BETWEEN ? AND ?
                OR substr(closed_at, 1, 10) BETWEEN ? AND ?
                OR (
                    COALESCE(exit_fill_time, closed_at) IS NULL
                    AND session_date BETWEEN ? AND ?
                )
              )
            """,
            [window.start_date, window.end_date, window.start_date, window.end_date, window.start_date, window.end_date],
        )
    out = {
        "orphans": 0,
        "missing_in_ibkr": 0,
        "partial_exits": 0,
        "delayed_fills": 0,
        "risk_guard_blocks": 0,
        "rejected_entries": 0,
        "sqlite_failures": 0,
        "reconnect_events": 0,
        "trades_count": 0,
        "reconstructed_trades_count": 0,
        "trades_updated_last_60s": 0,
        "reducer_running": 0,
        "last_reducer_run_at": "",
        "last_fill_ingest_started_at": "",
        "last_fill_ingest_finished_at": "",
        "last_position_reconcile_started_at": "",
        "last_position_reconcile_finished_at": "",
        "fill_ingest_running": 0,
        "position_reconcile_running": 0,
        "pending_execution_count": 0,
        "pending_commission_count": 0,
        "pending_realized_pnl_count": 0,
        "pending_trade_finalization_count": 0,
    }
    for row in events.to_dict("records"):
        event_type = str(row.get("event_type") or "")
        reason = str(row.get("reason") or "")
        count = int(row.get("count") or 0)
        blob = f"{event_type} {reason}"
        if "ORPHAN" in blob:
            out["orphans"] += count
        if "MISSING_IN_IBKR" in blob or "LOCAL_POSITION_MISSING_IN_IBKR" in blob:
            out["missing_in_ibkr"] += count
        if "EXIT_PARTIAL" in blob:
            out["partial_exits"] += count
        if "DELAYED_FILL_AFTER_CANCEL" in blob:
            out["delayed_fills"] += count
        if "SQLITE_WRITE_FAILED" in blob:
            out["sqlite_failures"] += count
        if "ENTRY_ORDER_REJECTED" in blob:
            out["rejected_entries"] += count
        if "RECONNECT" in blob:
            out["reconnect_events"] += count
    if not risks.empty:
        out["risk_guard_blocks"] = int(risks["count"].fillna(0).sum())
    if not rec.empty:
        last = rec.iloc[0].to_dict()
        out["orphans"] = max(out["orphans"], int(last.get("orphan_count") or 0))
        out["missing_in_ibkr"] = max(out["missing_in_ibkr"], int(last.get("drift_count") or 0))
    if not trade_diag.empty:
        row = trade_diag.iloc[0].to_dict()
        out["trades_count"] = int(row.get("trades_count") or 0)
        out["reconstructed_trades_count"] = int(row.get("reconstructed_trades_count") or 0)
        out["trades_updated_last_60s"] = int(row.get("trades_updated_last_60s") or 0)
        out["last_reducer_run_at"] = str(row.get("last_reducer_run_at") or "")
    if not runtime_state.empty:
        for row in runtime_state.to_dict("records"):
            key = str(row.get("key") or "")
            raw = parse_raw_json(row.get("raw_json"))
            if key == "fill_ingest":
                out["last_fill_ingest_started_at"] = str(raw.get("started_at") or "")
                out["last_fill_ingest_finished_at"] = str(raw.get("finished_at") or "")
                out["fill_ingest_running"] = 1 if str(raw.get("status") or "").lower() == "running" else 0
            elif key == "position_reconcile":
                out["last_position_reconcile_started_at"] = str(raw.get("started_at") or "")
                out["last_position_reconcile_finished_at"] = str(raw.get("finished_at") or "")
                out["position_reconcile_running"] = 1 if str(raw.get("status") or "").lower() == "running" else 0
    if not pending_counts.empty:
        row = pending_counts.iloc[0].to_dict()
        out["pending_execution_count"] = int(row.get("execution_count") or 0)
        out["pending_commission_count"] = int(row.get("pending_commission_count") or 0)
        out["pending_realized_pnl_count"] = int(row.get("pending_realized_pnl_count") or 0)
    if not pending_trades.empty:
        out["pending_trade_finalization_count"] = int(pending_trades.iloc[0].to_dict().get("pending_trade_finalization_count") or 0)
    return out


def load_position_row_diagnostics(
    conn: sqlite3.Connection,
    window: DateWindow,
    strategy: str | None,
    latest_open_count: int,
) -> dict[str, int]:
    clause, params = strategy_clause("p", strategy)
    active_rows = read_sql(
        conn,
        f"""
        SELECT COUNT(*) AS count
        FROM positions p
        WHERE COALESCE(p.session_date, '') <= ? {clause}
          AND COALESCE(p.active, 0) = 1
        """,
        [window.end_date, *params],
    )
    today_active_rows = read_sql(
        conn,
        f"""
        SELECT COUNT(*) AS count
        FROM positions p
        WHERE COALESCE(p.session_date, '') = ? {clause}
          AND COALESCE(p.active, 0) = 1
          AND {OPEN_POSITION_STATUS_SQL}
        """,
        [window.end_date, *params],
    )
    latest_candidate_rows = read_sql(
        conn,
        f"""
        WITH latest_positions AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(p.strategy_name, 'unknown'), UPPER(p.symbol)
                    ORDER BY COALESCE(p.updated_at, '') DESC, COALESCE(p.session_date, '') DESC, p.position_key DESC
                ) AS rn
            FROM positions p
            WHERE COALESCE(p.session_date, '') <= ? {clause}
        )
        SELECT COUNT(*) AS count
        FROM latest_positions p
        WHERE p.rn = 1
          AND COALESCE(p.active, 0) = 1
          AND {OPEN_POSITION_STATUS_SQL}
        """,
        [window.end_date, *params],
    )
    exit_order_stale_rows = read_sql(
        conn,
        f"""
        WITH latest_positions AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(p.strategy_name, 'unknown'), UPPER(p.symbol)
                    ORDER BY COALESCE(p.updated_at, '') DESC, COALESCE(p.session_date, '') DESC, p.position_key DESC
                ) AS rn
            FROM positions p
            WHERE COALESCE(p.session_date, '') <= ? {clause}
        )
        SELECT COUNT(*) AS count
        FROM latest_positions p
        WHERE p.rn = 1
          AND COALESCE(p.active, 0) = 1
          AND (
              UPPER(COALESCE(p.status, '')) = 'EXIT_ORDER'
              OR UPPER(COALESCE(p.status, '')) LIKE 'EXIT_ORDER|%'
          )
          AND COALESCE(p.ibkr_quantity, 0) = 0
        """,
        [window.end_date, *params],
    )
    duplicate_rows = read_sql(
        conn,
        f"""
        SELECT COUNT(*) AS count
        FROM (
            SELECT UPPER(symbol) AS symbol, COUNT(*) AS active_rows
            FROM positions p
            WHERE COALESCE(p.session_date, '') <= ? {clause}
              AND COALESCE(p.active, 0) = 1
              AND {OPEN_POSITION_STATUS_SQL}
            GROUP BY UPPER(symbol)
            HAVING COUNT(*) > 1
        )
        """,
        [window.end_date, *params],
    )
    ibkr_rows = read_sql(
        conn,
        """
        SELECT ibkr_positions_count
        FROM reconciliation_runs
        WHERE COALESCE(substr(finished_at, 1, 10), substr(started_at, 1, 10)) BETWEEN ? AND ?
        ORDER BY COALESCE(finished_at, started_at) DESC
        LIMIT 1
        """,
        [window.start_date, window.end_date],
    )
    sqlite_active = int(active_rows.iloc[0]["count"] or 0) if not active_rows.empty else 0
    latest_candidates = int(latest_candidate_rows.iloc[0]["count"] or 0) if not latest_candidate_rows.empty else 0
    ibkr_positions = int(ibkr_rows.iloc[0]["ibkr_positions_count"] or 0) if not ibkr_rows.empty else 0
    stale = max(0, sqlite_active - int(latest_open_count))
    return {
        "active_positions_raw_count": sqlite_active,
        "active_positions_today_count": int(today_active_rows.iloc[0]["count"] or 0) if not today_active_rows.empty else 0,
        "displayed_open_positions_count": int(latest_open_count),
        "sqlite_active_positions_count": sqlite_active,
        "latest_active_positions_count": int(latest_open_count),
        "latest_active_position_candidates_count": latest_candidates,
        "open_positions_count": int(latest_open_count),
        "ibkr_positions_count": ibkr_positions,
        "stale_active_positions_count": stale,
        "duplicate_active_symbol_count": int(duplicate_rows.iloc[0]["count"] or 0) if not duplicate_rows.empty else 0,
        "exit_order_stale_count": int(exit_order_stale_rows.iloc[0]["count"] or 0) if not exit_order_stale_rows.empty else 0,
    }


def build_summary(
    open_positions: pd.DataFrame,
    closed_positions: pd.DataFrame,
    execution_pnl: dict[str, float] | None = None,
) -> dict[str, Any]:
    trusted_closed = closed_positions
    if not closed_positions.empty and "runtime_pnl_trusted" in closed_positions.columns:
        trusted_closed = closed_positions[closed_positions["runtime_pnl_trusted"].fillna(True).astype(bool)]
    if execution_pnl is None:
        gross = float(trusted_closed["gross"].fillna(0).sum()) if not trusted_closed.empty else 0.0
        commissions = float(trusted_closed["ibkr_commission"].fillna(0).sum()) if not trusted_closed.empty else 0.0
        net = float(trusted_closed["net_actual"].fillna(0).sum()) if not trusted_closed.empty else 0.0
        pnl_source = "trades"
    else:
        gross = float(execution_pnl.get("gross_pnl") or 0.0)
        commissions = float(execution_pnl.get("commissions") or 0.0)
        net = float(execution_pnl.get("net_actual_pnl") or 0.0)
        pnl_source = str(execution_pnl.get("main_pnl_source") or "executions_realized_pnl_minus_sell_commission")
    open_upnl = float(pd.to_numeric(open_positions["upnl"], errors="coerce").fillna(0).sum()) if not open_positions.empty else 0.0
    wins = trusted_closed[trusted_closed["gross"].fillna(0) > 0] if not trusted_closed.empty else trusted_closed
    win_rate = (len(wins) / len(trusted_closed) * 100.0) if len(trusted_closed) else 0.0
    peak_values = pd.to_numeric(trusted_closed["peak_pct"], errors="coerce") if not trusted_closed.empty else pd.Series(dtype=float)
    giveback_values = pd.to_numeric(trusted_closed["drop_from_peak_pct"], errors="coerce") if not trusted_closed.empty else pd.Series(dtype=float)
    return {
        "gross_pnl": gross,
        "net_actual_pnl": net,
        "open_upnl": open_upnl,
        "total_pnl": net + open_upnl,
        "win_rate": win_rate,
        "avg_peak": float(peak_values.fillna(0).mean()) if not trusted_closed.empty else 0.0,
        "avg_giveback": float(giveback_values.fillna(0).mean()) if not trusted_closed.empty else 0.0,
        "commissions": commissions,
        "expectancy": gross / len(trusted_closed) if len(trusted_closed) else 0.0,
        "closed_trades": float((execution_pnl or {}).get("closed_symbols") or len(closed_positions)),
        "open_trades": float(len(open_positions)),
        "closed_pnl_source": pnl_source,
        "main_pnl_source": pnl_source,
        "realized_pnl_semantics": "gross_before_commission" if execution_pnl is not None else "trades_table",
        "net_formula": "sum_realized_pnl_minus_commission" if execution_pnl is not None else "trades_net_actual",
        "execution_rows": float((execution_pnl or {}).get("execution_rows") or 0),
        "execution_symbols": float((execution_pnl or {}).get("symbols") or 0),
        "closed_symbols": float((execution_pnl or {}).get("closed_symbols") or 0),
    }


def build_data_quality_summary(closed_positions: pd.DataFrame) -> dict[str, int]:
    if closed_positions.empty:
        return {
            "closed_trades_count": 0,
            "commission_ok": 0,
            "commission_partial": 0,
            "commission_missing": 0,
            "peak_ok": 0,
            "peak_missing": 0,
            "mfe_missing": 0,
            "mae_missing": 0,
            "peak_price_missing": 0,
            "data_quality_warning_count": 0,
        }
    commission_status = closed_positions.get("commission_status", pd.Series(dtype=str)).fillna("").astype(str)
    peak_source = closed_positions.get("peak_source", pd.Series(dtype=str)).fillna("").astype(str)
    data_quality = closed_positions.get("data_quality", pd.Series(dtype=str)).fillna("OK").astype(str)
    mfe_values = pd.to_numeric(closed_positions.get("peak_pct", pd.Series(dtype=float)), errors="coerce")
    mae_values = pd.to_numeric(closed_positions.get("mae_pct", pd.Series(dtype=float)), errors="coerce")
    peak_prices = pd.to_numeric(closed_positions.get("peak_price", pd.Series(dtype=float)), errors="coerce")
    return {
        "closed_trades_count": int(len(closed_positions)),
        "commission_ok": int((commission_status == "OK").sum()),
        "commission_partial": int((commission_status == "PARTIAL").sum()),
        "commission_missing": int((commission_status == "MISSING").sum()),
        "peak_ok": int((peak_source != "missing").sum()),
        "peak_missing": int((peak_source == "missing").sum()),
        "mfe_missing": int(mfe_values.isna().sum()),
        "mae_missing": int(mae_values.isna().sum()),
        "peak_price_missing": int(peak_prices.isna().sum()),
        "data_quality_warning_count": int((data_quality != "OK").sum()),
    }


def exit_simulation(closed_positions: pd.DataFrame) -> pd.DataFrame:
    if closed_positions.empty:
        return pd.DataFrame(columns=["scenario", "trades", "captured", "gross", "net"])
    rows = []
    for row in closed_positions.to_dict("records"):
        rows.append({
            "symbol": row.get("symbol"),
            "qty": row.get("qty") or 0,
            "buy": row.get("buy") or 0,
            "sell": row.get("sell") or 0,
            "gross": row.get("gross") or 0,
            "peak_gain_pct": row.get("peak_pct") or 0,
        })
    sim = simulate_exit_strategies(rows, commission_per_trade=0.0)
    return pd.DataFrame([
        {"scenario": x["name"], "trades": x["trades"], "captured": x["captured"], "gross": x["gross"], "net": x["net"]}
        for x in sim
    ])


def load_dashboard_snapshot(
    sqlite_path: str | Path | None,
    window: DateWindow,
    strategy: str | None = "All",
    *,
    include_reconstructed: bool = False,
    broker_portfolio: pd.DataFrame | None = None,
) -> dict[str, Any]:
    path = Path(resolve_sqlite_path(sqlite_path))
    if not path.exists():
        empty = pd.DataFrame()
        return {
            "summary": build_summary(empty, empty),
            "data_quality_summary": build_data_quality_summary(empty),
            "open_positions": empty,
            "orphan_stale_positions": empty,
            "rejected_entries": empty,
            "closed_positions": empty,
            "pending_trades": empty,
            "exit_simulation": empty,
            "diagnostics": {},
            "executions": empty,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "source": str(path),
        }
    conn = connect(path)
    try:
        conn.execute("BEGIN")
        executions = load_executions(conn, window, strategy)
        execution_pnl = load_execution_pnl_summary(conn, window, strategy)
        execution_lookup = load_executions(conn, expanded_lookup_window(window), strategy)
        raw_closed_rows = load_raw_closed_trade_rows(conn, window, strategy)
        closed, closed_diag = load_closed_positions(
            conn,
            window,
            strategy,
            execution_lookup,
            include_reconstructed=include_reconstructed,
        )
        open_positions = load_open_positions(conn, window, strategy, executions, execution_lookup, broker_portfolio)
        pending_trades = load_pending_trades(conn, window, strategy)
        raw_active_positions = load_raw_active_positions(conn, strategy)
        orphan_stale_positions = load_orphan_stale_positions(conn, window, strategy)
        excluded_open_positions = load_excluded_open_positions(conn, window, strategy, open_positions, orphan_stale_positions)
        rejected_entries = load_rejected_entries(conn, window, strategy)
        diagnostics = load_diagnostics(conn, window, strategy)
        diagnostics.update(closed_diag)
        diagnostics.update(load_position_row_diagnostics(conn, window, strategy, len(open_positions)))
        diagnostics["active_positions_raw_count"] = int(len(raw_active_positions))
        diagnostics["raw_active_sqlite_count"] = int(len(raw_active_positions))
        raw_open_symbols = {
            str(row.get("symbol") or "").upper()
            for row in raw_active_positions.to_dict("records")
            if str(row.get("symbol") or "")
        }
        displayed_open_symbols = {
            str(row.get("symbol") or "").upper()
            for row in open_positions.to_dict("records")
            if str(row.get("symbol") or "")
        }
        dropped_symbols = sorted(symbol for symbol in raw_open_symbols - displayed_open_symbols if symbol)
        diagnostics["displayed_open_count"] = int(len(open_positions))
        diagnostics["dropped_open_count"] = int(len(dropped_symbols))
        diagnostics["dropped_symbols"] = ",".join(dropped_symbols)
        if not open_positions.empty and "position_bucket" in open_positions.columns:
            diagnostics["today_open_positions_count"] = int((open_positions["position_bucket"].fillna("") == "today").sum())
            diagnostics["stale_carry_open_count"] = int((open_positions["position_bucket"].fillna("") == "carry_stale").sum())
        else:
            diagnostics["today_open_positions_count"] = 0
            diagnostics["stale_carry_open_count"] = 0
        diagnostics["orphan_stale_position_count"] = int(len(orphan_stale_positions))
        diagnostics["stale_carry_count"] = int(diagnostics.get("stale_carry_open_count", 0) or 0)
        diagnostics["carry_active_positions_count"] = int(diagnostics.get("stale_carry_open_count", 0) or 0)
        diagnostics["orphan_stale_count"] = int(diagnostics.get("orphan_stale_position_count", 0) or 0)
        diagnostics["active_positions_after_orphan_filter_count"] = int(len(open_positions))
        diagnostics["displayed_today_open_count"] = int(diagnostics.get("today_open_positions_count", 0) or 0)
        diagnostics["displayed_carry_open_count"] = int(diagnostics.get("stale_carry_open_count", 0) or 0)
        diagnostics["excluded_open_positions_count"] = int(len(excluded_open_positions))
        if not orphan_stale_positions.empty:
            oldest = orphan_stale_positions.sort_values(["age_days", "symbol"], ascending=[False, True]).iloc[0].to_dict()
            diagnostics["oldest_orphan_stale_position"] = str(oldest.get("symbol") or "")
            diagnostics["oldest_orphan_stale_position_age_days"] = float(oldest.get("age_days") or 0.0)
            diagnostics["cleanup_recommendation"] = "Close stale orphan position"
        else:
            diagnostics["oldest_orphan_stale_position"] = ""
            diagnostics["oldest_orphan_stale_position_age_days"] = 0.0
            diagnostics["cleanup_recommendation"] = ""
        diagnostics["closed_trades_count"] = int(len(closed))
        diagnostics["pending_trades_count"] = int(len(pending_trades))
        if not closed.empty and "runtime_pnl_trusted" in closed.columns:
            untrusted_closed = closed[~closed["runtime_pnl_trusted"].fillna(True).astype(bool)]
            diagnostics["untrusted_carry_closed_count"] = int(len(untrusted_closed))
            diagnostics["untrusted_carry_closed_symbols"] = ",".join(sorted({str(x).upper() for x in untrusted_closed["symbol"].dropna().tolist()}))
        else:
            diagnostics["untrusted_carry_closed_count"] = 0
            diagnostics["untrusted_carry_closed_symbols"] = ""
        raw_closed_ids = {
            str(row.get("trade_id") or "")
            for row in raw_closed_rows.to_dict("records")
            if str(row.get("trade_id") or "")
        }
        displayed_closed_ids = {
            str(row.get("trade_id") or "")
            for row in closed.to_dict("records")
            if str(row.get("trade_id") or "")
        }
        dropped_closed_ids = sorted(raw_closed_ids - displayed_closed_ids)
        diagnostics["raw_closed_trade_count"] = int(len(raw_closed_rows))
        diagnostics["displayed_closed_trade_count"] = int(len(closed))
        diagnostics["dropped_closed_trade_count"] = int(len(dropped_closed_ids))
        diagnostics["dropped_closed_trade_ids"] = ",".join(dropped_closed_ids)
        diagnostics["broker_portfolio_valuation_symbols"] = int(len(broker_portfolio_by_symbol(broker_portfolio)))
        if not open_positions.empty and "dashboard_broker_price_diff" in open_positions.columns:
            price_diffs = pd.to_numeric(open_positions["dashboard_broker_price_diff"], errors="coerce").dropna()
            diagnostics["dashboard_broker_price_mismatch_count"] = int((price_diffs.abs() > 1e-9).sum())
            diagnostics["dashboard_broker_price_max_abs_diff"] = float(price_diffs.abs().max()) if not price_diffs.empty else 0.0
        else:
            diagnostics["dashboard_broker_price_mismatch_count"] = 0
            diagnostics["dashboard_broker_price_max_abs_diff"] = 0.0
        if not closed.empty and "carried_closed_today" in closed.columns:
            diagnostics["carried_closed_today_count"] = int(pd.Series(closed["carried_closed_today"]).fillna(False).astype(bool).sum())
        else:
            diagnostics["carried_closed_today_count"] = 0
        trades_updated_last_60s = int(diagnostics.get("trades_updated_last_60s", 0) or 0)
        diagnostics["runtime_trust_status"] = "SQLITE_UNTRUSTED_REDUCER_ACTIVE" if trades_updated_last_60s > 0 else "SQLITE_PERSISTED_TRADES"
        diagnostics["broker_closed_trades_count"] = int(execution_pnl.get("closed_symbols") or 0)
        diagnostics["broker_net_pnl"] = float(execution_pnl.get("net_actual_pnl") or 0.0)
        diagnostics["broker_commissions"] = float(execution_pnl.get("commissions") or 0.0)
        diagnostics["reducer_closed_rows_count"] = int(len(closed))
        diagnostics["reducer_net_pnl"] = float(pd.to_numeric(closed.get("net_actual", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not closed.empty else 0.0
        diagnostics["reducer_commissions"] = float(pd.to_numeric(closed.get("ibkr_commission", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not closed.empty else 0.0
        diagnostics["closed_pnl_source"] = str(execution_pnl.get("main_pnl_source") or "executions_realized_pnl_minus_sell_commission")
        diagnostics["main_pnl_source"] = str(execution_pnl.get("main_pnl_source") or "executions_realized_pnl_minus_sell_commission")
        diagnostics["realized_pnl_semantics"] = "gross_before_commission"
        diagnostics["net_formula"] = "sum_sell_realized_pnl_minus_sell_commission"
        diagnostics["execution_pnl_rows"] = int(execution_pnl.get("execution_rows") or 0)
        diagnostics["execution_pnl_symbols"] = int(execution_pnl.get("symbols") or 0)
        diagnostics["execution_closed_symbols"] = int(execution_pnl.get("closed_symbols") or 0)
        diagnostics["execution_gross_realized"] = float(execution_pnl.get("gross_pnl") or 0.0)
        diagnostics["execution_commissions"] = float(execution_pnl.get("commissions") or 0.0)
        diagnostics["execution_sell_commissions"] = float(execution_pnl.get("sell_commissions") or 0.0)
        diagnostics["execution_all_commissions"] = float(execution_pnl.get("all_commissions") or 0.0)
        diagnostics["execution_net_pnl"] = float(execution_pnl.get("net_actual_pnl") or 0.0)
        snapshot_version = (
            f"closed={len(closed)};"
            f"trades={diagnostics.get('trades_count', 0)};"
            f"reconstructed={diagnostics.get('reconstructed_trades_count', 0)};"
            f"last_reducer={diagnostics.get('last_reducer_run_at', '')}"
        )
        snapshot = {
            "summary": build_summary(open_positions, closed, execution_pnl),
            "data_quality_summary": build_data_quality_summary(closed),
            "open_positions": open_positions,
            "raw_active_positions": raw_active_positions,
            "orphan_stale_positions": orphan_stale_positions,
            "excluded_open_positions": excluded_open_positions,
            "rejected_entries": rejected_entries,
            "closed_positions": closed,
            "pending_trades": pending_trades,
            "exit_simulation": exit_simulation(closed),
            "diagnostics": diagnostics,
            "executions": executions,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_version": snapshot_version,
            "trust_status": diagnostics["runtime_trust_status"],
            "source": str(path),
        }
        conn.commit()
        return snapshot
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
