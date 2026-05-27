from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_trading.analytics.v67_daily_report import (
    primary_net,
    reconstruct_closed_trades_from_fills,
    simulate_exit_strategies,
)
from src.live_trading.storage.sqlite_store import DEFAULT_SQLITE_PATH, resolve_sqlite_path


CLOSED_STATUSES = {"CLOSED", "DONE", "EXIT_FILLED", "FLAT"}


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
    clause, params = strategy_clause("e", strategy)
    return read_sql(
        conn,
        f"""
        SELECT
            execution_id,
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


def closed_from_trades(conn: sqlite3.Connection, window: DateWindow, strategy: str | None) -> pd.DataFrame:
    clause, params = strategy_clause("t", strategy)
    rows = read_sql(
        conn,
        f"""
        SELECT
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
            commission AS ibkr_commission,
            net_pnl AS net_actual,
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
    out["pnl_pct"] = ((out["sell"] / out["buy"] - 1.0) * 100.0).where(out["buy"].fillna(0) != 0, 0.0)
    out["drop_from_peak_pct"] = out["pnl_pct"].fillna(0.0) - out["peak_pct"].fillna(0.0)
    out["hold_minutes"] = [hold_minutes(a, b or c) for a, b, c in zip(out["entry_time"], out["exit_time"], out["closed_at"])]
    out["ibkr_commission"] = out["ibkr_commission"].fillna(0.0)
    out["net_actual"] = out["net_actual"].fillna(out["gross"].fillna(0.0) - out["ibkr_commission"])
    return out[
        [
            "symbol", "gross", "ibkr_commission", "net_actual", "pnl_pct", "peak_pct",
            "drop_from_peak_pct", "hold_minutes", "exit_reason", "strategy",
            "entry_time", "exit_time", "qty", "buy", "sell", "session_date",
        ]
    ]


def closed_from_executions(executions: pd.DataFrame) -> pd.DataFrame:
    if executions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (session_date, strategy), group in executions.groupby(["session_date", "strategy"], dropna=False):
        fill_rows = group.to_dict("records")
        for row in reconstruct_closed_trades_from_fills(fill_rows):
            buy = to_float(row.get("buy"), 0.0) or 0.0
            sell = to_float(row.get("sell"), 0.0) or 0.0
            pnl_pct = ((sell / buy - 1.0) * 100.0) if buy and sell else 0.0
            peak_pct = to_float(row.get("peak_gain_pct"), max(pnl_pct, 0.0)) or 0.0
            rows.append({
                "symbol": row.get("symbol"),
                "gross": to_float(row.get("gross"), 0.0),
                "ibkr_commission": to_float(row.get("actual_commission"), 0.0),
                "net_actual": primary_net(row),
                "pnl_pct": pnl_pct,
                "peak_pct": peak_pct,
                "drop_from_peak_pct": to_float(row.get("drop_from_peak_pct"), pnl_pct - peak_pct),
                "hold_minutes": hold_minutes(row.get("buy_time"), row.get("sell_time")),
                "exit_reason": row.get("reason") or "",
                "strategy": strategy or "unknown",
                "entry_time": row.get("buy_time"),
                "exit_time": row.get("sell_time"),
                "qty": to_float(row.get("qty"), 0.0),
                "buy": buy,
                "sell": sell,
                "session_date": session_date,
            })
    return pd.DataFrame(rows)


def load_closed_positions(conn: sqlite3.Connection, window: DateWindow, strategy: str | None, executions: pd.DataFrame) -> pd.DataFrame:
    trades = closed_from_trades(conn, window, strategy)
    if not trades.empty:
        return trades.sort_values(["exit_time", "symbol"], na_position="last").reset_index(drop=True)
    closed = closed_from_executions(executions)
    if closed.empty:
        return closed
    return closed.sort_values(["exit_time", "symbol"], na_position="last").reset_index(drop=True)


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
    return {
        "gross_pnl": gross,
        "net_actual_pnl": net,
        "open_upnl": open_upnl,
        "total_pnl": net + open_upnl,
        "win_rate": win_rate,
        "avg_peak": float(closed_positions["peak_pct"].fillna(0).mean()) if not closed_positions.empty else 0.0,
        "avg_giveback": float(closed_positions["drop_from_peak_pct"].fillna(0).mean()) if not closed_positions.empty else 0.0,
        "commissions": commissions,
        "expectancy": gross / len(closed_positions) if len(closed_positions) else 0.0,
        "closed_trades": float(len(closed_positions)),
        "open_trades": float(len(open_positions)),
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
