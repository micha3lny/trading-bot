from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analytics.v67_daily_report import (
    reconstruct_closed_trades_from_fills,
    simulate_exit_strategies,
)
from src.live_trading.storage.sqlite_store import DEFAULT_SQLITE_PATH, resolve_sqlite_path


CLOSED_STATUSES = {"CLOSED", "DONE", "EXIT_FILLED", "FLAT"}
DEFAULT_RECORDER_ROOT = Path("data/live/recorder")


@dataclass(frozen=True)
class DateWindow:
    start_date: str
    end_date: str


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%F")


def connect(sqlite_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(resolve_sqlite_path(sqlite_path or DEFAULT_SQLITE_PATH))
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def read_sql(conn: sqlite3.Connection, sql: str, params: list[Any] | tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def parse_raw_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def raw_json_peak_value(raw: dict[str, Any]) -> float | None:
    for key in ("mfe_pct", "peak_pct", "peak_gain_pct", "max_gain_pct"):
        value = to_float(raw.get(key), None)
        if value is not None:
            return value
    entry_price = to_float(raw.get("entry_price") or raw.get("buy") or raw.get("buy_price"), None)
    peak_price = to_float(raw.get("peak_price") or raw.get("high_watermark") or raw.get("mfe_price"), None)
    if entry_price and peak_price is not None:
        return ((peak_price / entry_price) - 1.0) * 100.0
    return None


def raw_execution_time_value(raw_value: Any) -> Any:
    raw = parse_raw_json(raw_value)
    execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
    if execution:
        return execution.get("time") or execution.get("executionTime") or execution.get("executed_at")
    return raw.get("executed_at") or raw.get("execution_time") or raw.get("time")


def to_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value in (None, ""):
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
    return read_sql(
        conn,
        f"""
        SELECT
            execution_id,
            trade_id,
            COALESCE(strategy_name, 'unknown') AS strategy,
            session_date,
            symbol,
            side AS action,
            quantity,
            price AS fill_price,
            order_id,
            perm_id,
            exchange,
            liquidity,
            executed_at,
            recorded_at,
            commission,
            commission_currency,
            realized_pnl,
            commission_source,
            raw_json
        FROM executions e
        WHERE e.session_date BETWEEN ? AND ? {clause}
        ORDER BY COALESCE(e.executed_at, e.recorded_at), e.execution_id
        """,
        [window.start_date, window.end_date, *params],
    )


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
    return row.get("executed_at") or raw_execution_time_value(row.get("raw_json"))


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


def execution_matches_for_trade(row: dict[str, Any], executions: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if executions.empty:
        return [], []
    trade_id = str(row.get("trade_id") or "")
    matched = pd.DataFrame()
    if trade_id and "trade_id" in executions.columns:
        matched = executions[executions["trade_id"].fillna("").astype(str) == trade_id]
    if matched.empty:
        same = executions[
            (executions["session_date"].fillna("").astype(str) == str(row.get("session_date") or ""))
            & (executions["symbol"].fillna("").astype(str).str.upper() == str(row.get("symbol") or "").upper())
        ]
        has_time_window = parse_dt(row.get("entry_time")) is not None or parse_dt(row.get("exit_time") or row.get("closed_at")) is not None
        same = time_window_rows(same, row.get("entry_time"), row.get("exit_time") or row.get("closed_at"))
        buy_count = len(side_rows(same, action_values={"BOT", "BUY"}))
        sell_count = len(side_rows(same, action_values={"SLD", "SELL"}))
        if has_time_window and buy_count >= 1 and sell_count >= 1:
            matched = same
        elif buy_count == 1 and sell_count == 1:
            matched = same
    return side_rows(matched, action_values={"BOT", "BUY"}), side_rows(matched, action_values={"SLD", "SELL"})


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
        SELECT trade_id, COALESCE(strategy_name, 'unknown') AS strategy, session_date, symbol, event_type, raw_json
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


def recorder_root() -> Path:
    return Path(os.environ.get("TRADING_BOT_RECORDER_DIR") or os.environ.get("DASHBOARD_RECORDER_DIR") or DEFAULT_RECORDER_ROOT)


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
    runtime_symbol_peak = runtime_peak_map.get(("", trade_key[1], trade_key[2], trade_key[3]))
    if runtime_symbol_peak:
        return runtime_symbol_peak
    symbol_key = (str(row.get("session_date") or ""), str(row.get("symbol") or "").upper())
    lifecycle_peak = lifecycle_peak_map.get(symbol_key)
    if lifecycle_peak:
        return lifecycle_peak
    candle_peak = candle_peak_for_trade(row, candle_rows)
    if candle_peak[0] is not None:
        return candle_peak
    return None, "missing"


def closed_from_trades(
    conn: sqlite3.Connection,
    window: DateWindow,
    strategy: str | None,
    executions: pd.DataFrame,
    runtime_peak_map: dict[tuple[str, str, str, str], tuple[float, str]],
    lifecycle_peak_map: dict[tuple[str, str], tuple[float, str]],
    candle_rows: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    clause, params = strategy_clause("t", strategy)
    rows = read_sql(
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
            mfe_pct AS peak_pct,
            exit_reason,
            raw_json
        FROM trades t
        WHERE t.session_date BETWEEN ? AND ? {clause}
          AND UPPER(COALESCE(t.status, '')) IN ({",".join("?" for _ in CLOSED_STATUSES)})
        ORDER BY COALESCE(t.closed_at, t.exit_fill_time), t.symbol
        """,
        [window.start_date, window.end_date, *params, *sorted(CLOSED_STATUSES)],
    )
    if rows.empty:
        return rows
    out = rows.copy()
    commissions: list[float] = []
    commission_statuses: list[str] = []
    data_quality: list[str] = []
    entry_times: list[Any] = []
    exit_times: list[Any] = []
    peak_values: list[float | None] = []
    peak_sources: list[str] = []
    entry_execution_counts: list[int] = []
    exit_execution_counts: list[int] = []
    confirmed_commission_counts: list[int] = []
    expected_commission_counts: list[int] = []
    commission_source_details: list[str] = []
    for row in out.to_dict("records"):
        buy_rows, sell_rows = execution_matches_for_trade(row, executions)
        entry_time, exit_time, flags = infer_entry_exit_times(row, buy_rows, sell_rows)
        enriched_row = {**row, "entry_time": entry_time, "exit_time": exit_time}
        commission, commission_status = confirmed_commission_for_execution_rows(buy_rows, sell_rows)
        peak_pct, peak_source = peak_from_sources(enriched_row, runtime_peak_map, lifecycle_peak_map, candle_rows)
        commissions.append(commission)
        commission_statuses.append(commission_status)
        data_quality.append(quality_label(flags, commission_status))
        entry_times.append(entry_time)
        exit_times.append(exit_time)
        peak_values.append(peak_pct)
        peak_sources.append(peak_source)
        entry_execution_counts.append(len(buy_rows))
        exit_execution_counts.append(len(sell_rows))
        confirmed_count = confirmed_commission_execution_count(buy_rows, sell_rows)
        confirmed_commission_counts.append(confirmed_count)
        expected_count = len(buy_rows) + len(sell_rows)
        expected_commission_counts.append(expected_count)
        commission_source_details.append(f"matched={len(buy_rows) + len(sell_rows)} ibkr={confirmed_count}")
    out["ibkr_commission"] = commissions
    out["commission_status"] = commission_statuses
    out["data_quality"] = data_quality
    out["entry_time"] = entry_times
    out["exit_time"] = exit_times
    out["peak_pct"] = peak_values
    out["peak_source"] = peak_sources
    out["entry_execution_count"] = entry_execution_counts
    out["exit_execution_count"] = exit_execution_counts
    out["confirmed_commission_execution_count"] = confirmed_commission_counts
    out["expected_commission_execution_count"] = expected_commission_counts
    out["commission_source_detail"] = commission_source_details
    out["gross"] = pd.to_numeric(out["gross"], errors="coerce").fillna(0.0)
    out["buy"] = pd.to_numeric(out["buy"], errors="coerce").fillna(0.0)
    out["sell"] = pd.to_numeric(out["sell"], errors="coerce").fillna(0.0)
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0)
    out["peak_pct"] = pd.to_numeric(out["peak_pct"], errors="coerce")
    out["net_actual"] = out["gross"] - out["ibkr_commission"]
    denominator = (out["buy"] * out["qty"].abs()).replace(0, pd.NA)
    out["net_pct"] = ((out["net_actual"] / denominator) * 100.0).fillna(0.0)
    out["pnl_pct"] = out["net_pct"]
    out["drop_from_peak_pct"] = out["net_pct"].fillna(0.0) - out["peak_pct"]
    out["hold_minutes"] = [hold_minutes(a, b or c) for a, b, c in zip(out["entry_time"], out["exit_time"], out["closed_at"])]
    return out[
        [
            "trade_id", "symbol", "qty", "ibkr_commission", "buy", "sell", "gross", "net_actual", "net_pct", "pnl_pct", "peak_pct",
            "drop_from_peak_pct", "hold_minutes", "exit_reason", "strategy",
            "entry_time", "exit_time", "commission_status", "data_quality", "session_date",
            "entry_execution_count", "exit_execution_count", "confirmed_commission_execution_count",
            "expected_commission_execution_count", "peak_source", "commission_source_detail",
        ]
    ]


def closed_from_executions(executions: pd.DataFrame) -> pd.DataFrame:
    if executions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (session_date, strategy), group in executions.groupby(["session_date", "strategy"], dropna=False):
        fill_rows = group.to_dict("records")
        for row in reconstruct_closed_trades_from_fills(fill_rows):
            buy_execution = next((x for x in fill_rows if str(x.get("execution_id") or "") == str(row.get("buy_execution_id") or "")), {})
            sell_execution = next((x for x in fill_rows if str(x.get("execution_id") or "") == str(row.get("sell_execution_id") or "")), {})
            buy_time = execution_time_value(buy_execution) if buy_execution else None
            sell_time = execution_time_value(sell_execution) if sell_execution else None
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
            flags = {"RECONSTRUCTED_FROM_EXECUTIONS"}
            if not buy_time or not sell_time:
                flags.add("MISSING_EXECUTION_TIME")
            commission_status = "OK" if str(row.get("commission_source") or "").lower() == "ibkr" else ("PARTIAL" if commission else "MISSING")
            if commission_status == "PARTIAL":
                flags.add("COMMISSION_PARTIAL")
            elif commission_status == "MISSING":
                flags.add("COMMISSION_MISSING")
            rows.append({
                "symbol": row.get("symbol"),
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
                "strategy": strategy or "unknown",
                "entry_time": buy_time,
                "exit_time": sell_time,
                "data_quality": quality_label_from_flags(flags),
                "entry_execution_count": 1,
                "exit_execution_count": 1,
                "confirmed_commission_execution_count": 2 if commission_status == "OK" else (1 if commission_status == "PARTIAL" else 0),
                "expected_commission_execution_count": 2,
                "peak_source": "fills_reconstruction" if raw_peak is not None else "missing",
                "commission_source_detail": "reconstructed",
                "qty": qty,
                "buy": buy,
                "sell": sell,
                "session_date": session_date,
            })
    return pd.DataFrame(rows)


def load_closed_positions(conn: sqlite3.Connection, window: DateWindow, strategy: str | None, executions: pd.DataFrame) -> pd.DataFrame:
    runtime_peak_map = load_runtime_peak_map(conn, window, strategy)
    lifecycle_peak_map = load_lifecycle_peak_map(window)
    candle_rows = load_candle_rows(window)
    trades = closed_from_trades(conn, window, strategy, executions, runtime_peak_map, lifecycle_peak_map, candle_rows)
    reconstructed = closed_from_executions(executions)
    if not trades.empty and not reconstructed.empty:
        trade_keys = {
            (str(row.get("session_date") or ""), str(row.get("symbol") or "").upper())
            for row in trades.to_dict("records")
        }
        reconstructed = reconstructed[
            ~reconstructed.apply(
                lambda row: (str(row.get("session_date") or ""), str(row.get("symbol") or "").upper()) in trade_keys,
                axis=1,
            )
        ]
    frames = [df for df in (trades, reconstructed) if not df.empty]
    if not frames:
        return pd.DataFrame()
    closed = pd.concat(frames, ignore_index=True, sort=False)
    return closed.sort_values(["net_actual", "symbol"], na_position="last").reset_index(drop=True)


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


def load_open_positions(
    conn: sqlite3.Connection,
    window: DateWindow,
    strategy: str | None,
    executions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    clause, params = strategy_clause("p", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
            COALESCE(strategy_name, 'unknown') AS strategy,
            session_date,
            symbol,
            status,
            quantity,
            avg_price,
            ibkr_quantity,
            ibkr_avg_cost,
            active,
            exit_sent,
            updated_at,
            raw_json
        FROM positions p
        WHERE p.session_date BETWEEN ? AND ? {clause}
          AND COALESCE(p.active, 0) = 1
        ORDER BY p.symbol
        """,
        [window.start_date, window.end_date, *params],
    )
    if rows.empty:
        return rows
    net_by_execution = execution_net_positions(executions)
    historical_window = window.end_date < utc_today()
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        row_strategy = str(row.get("strategy") or "unknown")
        symbol = str(row.get("symbol") or "").upper()
        session_date = str(row.get("session_date") or "")
        net_key = (session_date, row_strategy, symbol)
        execution_net = net_by_execution.get(net_key)
        if historical_window and (execution_net is None or execution_net <= 1e-9):
            continue
        if execution_net is not None and abs(execution_net) < 1e-9:
            continue
        raw = parse_raw_json(row.get("raw_json"))
        qty = to_float(row.get("quantity"), None)
        if qty is None:
            qty = to_float(row.get("ibkr_quantity"), 0.0)
        buy = to_float(row.get("avg_price") or raw.get("entry_price"), 0.0) or 0.0
        now = to_float(raw.get("market_price") or raw.get("current_price") or raw.get("last_price"), None)
        if now is None:
            now = buy if buy else to_float(row.get("ibkr_avg_cost"), 0.0) or 0.0
        upnl = to_float(raw.get("unrealized_pnl") or raw.get("unrealizedPNL"), None)
        if upnl is None:
            upnl = (now - buy) * (qty or 0.0) if buy and now else 0.0
        now_pct = ((now / buy - 1.0) * 100.0) if buy and now else 0.0
        peak_price = to_float(raw.get("peak_price"), max(buy, now))
        peak_pct = to_float(raw.get("peak_gain_pct"), ((peak_price / buy - 1.0) * 100.0 if buy and peak_price else 0.0))
        entry_time = raw.get("entry_time") or raw.get("buy_time") or row.get("updated_at")
        out.append({
            "symbol": row.get("symbol"),
            "qty": qty or 0.0,
            "buy": buy,
            "now": now,
            "upnl": upnl,
            "now_pct": now_pct,
            "peak_pct": peak_pct,
            "giveback_pct": now_pct - peak_pct,
            "hold_minutes": hold_minutes(entry_time),
            "ibkr_commission": to_float(raw.get("ibkr_commission"), 0.0) or 0.0,
            "status": row.get("status") or ("EXIT_SENT" if row.get("exit_sent") else "OPEN"),
            "strategy": row_strategy,
            "entry_time": entry_time,
            "session_date": session_date,
        })
    return pd.DataFrame(out)


def load_diagnostics(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> dict[str, int]:
    event_clause, event_params = strategy_clause("r", strategy)
    risk_clause, risk_params = strategy_clause("q", strategy)
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
    out = {
        "orphans": 0,
        "missing_in_ibkr": 0,
        "partial_exits": 0,
        "delayed_fills": 0,
        "risk_guard_blocks": 0,
        "sqlite_failures": 0,
        "reconnect_events": 0,
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
        if "RECONNECT" in blob:
            out["reconnect_events"] += count
    if not risks.empty:
        out["risk_guard_blocks"] = int(risks["count"].fillna(0).sum())
    if not rec.empty:
        last = rec.iloc[0].to_dict()
        out["orphans"] = max(out["orphans"], int(last.get("orphan_count") or 0))
        out["missing_in_ibkr"] = max(out["missing_in_ibkr"], int(last.get("drift_count") or 0))
    return out


def build_summary(open_positions: pd.DataFrame, closed_positions: pd.DataFrame) -> dict[str, float]:
    gross = float(closed_positions["gross"].fillna(0).sum()) if not closed_positions.empty else 0.0
    commissions = float(closed_positions["ibkr_commission"].fillna(0).sum()) if not closed_positions.empty else 0.0
    net = float(closed_positions["net_actual"].fillna(0).sum()) if not closed_positions.empty else 0.0
    open_upnl = float(open_positions["upnl"].fillna(0).sum()) if not open_positions.empty else 0.0
    wins = closed_positions[closed_positions["gross"].fillna(0) > 0] if not closed_positions.empty else closed_positions
    win_rate = (len(wins) / len(closed_positions) * 100.0) if len(closed_positions) else 0.0
    peak_values = pd.to_numeric(closed_positions["peak_pct"], errors="coerce") if not closed_positions.empty else pd.Series(dtype=float)
    giveback_values = pd.to_numeric(closed_positions["drop_from_peak_pct"], errors="coerce") if not closed_positions.empty else pd.Series(dtype=float)
    return {
        "gross_pnl": gross,
        "net_actual_pnl": net,
        "open_upnl": open_upnl,
        "total_pnl": net + open_upnl,
        "win_rate": win_rate,
        "avg_peak": float(peak_values.fillna(0).mean()) if not closed_positions.empty else 0.0,
        "avg_giveback": float(giveback_values.fillna(0).mean()) if not closed_positions.empty else 0.0,
        "commissions": commissions,
        "expectancy": gross / len(closed_positions) if len(closed_positions) else 0.0,
        "closed_trades": float(len(closed_positions)),
        "open_trades": float(len(open_positions)),
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
            "data_quality_warning_count": 0,
        }
    commission_status = closed_positions.get("commission_status", pd.Series(dtype=str)).fillna("").astype(str)
    peak_source = closed_positions.get("peak_source", pd.Series(dtype=str)).fillna("").astype(str)
    data_quality = closed_positions.get("data_quality", pd.Series(dtype=str)).fillna("OK").astype(str)
    return {
        "closed_trades_count": int(len(closed_positions)),
        "commission_ok": int((commission_status == "OK").sum()),
        "commission_partial": int((commission_status == "PARTIAL").sum()),
        "commission_missing": int((commission_status == "MISSING").sum()),
        "peak_ok": int((peak_source != "missing").sum()),
        "peak_missing": int((peak_source == "missing").sum()),
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


def load_dashboard_snapshot(sqlite_path: str | Path | None, window: DateWindow, strategy: str | None = "All") -> dict[str, Any]:
    path = Path(resolve_sqlite_path(sqlite_path))
    if not path.exists():
        empty = pd.DataFrame()
        return {
            "summary": build_summary(empty, empty),
            "data_quality_summary": build_data_quality_summary(empty),
            "open_positions": empty,
            "closed_positions": empty,
            "exit_simulation": empty,
            "diagnostics": {},
            "executions": empty,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "source": str(path),
        }
    conn = connect(path)
    try:
        executions = load_executions(conn, window, strategy)
        closed = load_closed_positions(conn, window, strategy, executions)
        open_positions = load_open_positions(conn, window, strategy, executions)
        return {
            "summary": build_summary(open_positions, closed),
            "data_quality_summary": build_data_quality_summary(closed),
            "open_positions": open_positions,
            "closed_positions": closed,
            "exit_simulation": exit_simulation(closed),
            "diagnostics": load_diagnostics(conn, window, strategy),
            "executions": executions,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "source": str(path),
        }
    finally:
        conn.close()
