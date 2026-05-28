from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.live_trading.unified_logger import append_unified_log, unified_logger_installed


DEFAULT_SQLITE_PATH = "data/runtime/trading_runtime.sqlite"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_date_utc() -> str:
    return date.today().isoformat()


def resolve_sqlite_path(path: str | Path | None = None) -> str:
    return str(path or os.environ.get("TRADING_BOT_SQLITE_PATH") or DEFAULT_SQLITE_PATH)


def open_sqlite_store(path: str | Path | None = None) -> SQLiteRuntimeStore | None:
    try:
        return SQLiteRuntimeStore(resolve_sqlite_path(path))
    except Exception as exc:
        line = f"{utc_now_iso()} SQLITE_WRITE_FAILED method=init error={exc!r}"
        print(line, flush=True)
        if not unified_logger_installed():
            append_unified_log(line)
        return None


def safe_json(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return str(value)


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


def raw_execution_time(value: Any) -> Any:
    raw = parse_jsonish(value)
    execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
    if execution:
        return execution.get("time") or execution.get("executionTime") or execution.get("executed_at")
    return raw.get("executed_at") or raw.get("execution_time") or raw.get("time")


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except Exception:
        return None


def bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)):
        return int(bool(value))
    return int(str(value).strip().lower() in {"1", "true", "yes", "y", "on"})


def clean_row(row: dict[str, Any], columns: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        value = row.get(col)
        if col.endswith("_json") or col == "raw_json" or col == "components_json" or col == "details_json":
            value = safe_json(value)
        out[col] = value
    return out


class SQLiteRuntimeStore:
    def __init__(self, path: str | Path | None = None, *, init: bool = True) -> None:
        self.path = Path(resolve_sqlite_path(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if init:
            self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runtime_events (
                id INTEGER PRIMARY KEY,
                event_time TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'INFO',
                event_type TEXT NOT NULL,
                strategy_name TEXT,
                strategy_version TEXT,
                session_date TEXT,
                symbol TEXT,
                trade_id TEXT,
                order_id TEXT,
                execution_id TEXT,
                source TEXT,
                reason TEXT,
                action_required INTEGER DEFAULT 0,
                acknowledged INTEGER DEFAULT 0,
                resolved INTEGER DEFAULT 0,
                first_seen_at TEXT,
                last_seen_at TEXT,
                repeat_count INTEGER DEFAULT 1,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_runtime_events_event_time ON runtime_events(event_time);
            CREATE INDEX IF NOT EXISTS idx_runtime_events_event_type ON runtime_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_runtime_events_severity ON runtime_events(severity);
            CREATE INDEX IF NOT EXISTS idx_runtime_events_symbol ON runtime_events(symbol);
            CREATE INDEX IF NOT EXISTS idx_runtime_events_session_date ON runtime_events(session_date);
            CREATE INDEX IF NOT EXISTS idx_runtime_events_resolved ON runtime_events(resolved);

            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                strategy_version TEXT,
                session_date TEXT,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL,
                side TEXT DEFAULT 'LONG',
                entry_signal_time TEXT,
                entry_order_time TEXT,
                entry_fill_time TEXT,
                exit_signal_time TEXT,
                exit_order_time TEXT,
                exit_fill_time TEXT,
                closed_at TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                remaining_quantity REAL,
                gross_pnl REAL,
                commission REAL,
                net_pnl REAL,
                mfe_pct REAL,
                mae_pct REAL,
                exit_reason TEXT,
                ibkr_entry_confirmed INTEGER DEFAULT 0,
                ibkr_exit_confirmed INTEGER DEFAULT 0,
                ibkr_position_flat_confirmed INTEGER DEFAULT 0,
                ibkr_position_flat_confirmed_at TEXT,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_trades_session_date ON trades(session_date);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy_name ON trades(strategy_name);

            CREATE TABLE IF NOT EXISTS orders (
                order_key TEXT PRIMARY KEY,
                trade_id TEXT,
                strategy_name TEXT,
                session_date TEXT,
                symbol TEXT,
                side TEXT,
                order_type TEXT,
                quantity REAL,
                limit_price REAL,
                status TEXT,
                ibkr_status TEXT,
                order_id TEXT,
                perm_id TEXT,
                client_id TEXT,
                submitted_at TEXT,
                acknowledged_at TEXT,
                cancelled_at TEXT,
                filled_at TEXT,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_orders_trade_id ON orders(trade_id);
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);
            CREATE INDEX IF NOT EXISTS idx_orders_perm_id ON orders(perm_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                trade_id TEXT,
                order_key TEXT,
                order_id TEXT,
                perm_id TEXT,
                strategy_name TEXT,
                session_date TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL,
                price REAL,
                exchange TEXT,
                liquidity TEXT,
                executed_at TEXT,
                recorded_at TEXT,
                commission REAL,
                commission_currency TEXT,
                realized_pnl REAL,
                commission_source TEXT,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_executions_session_date ON executions(session_date);
            CREATE INDEX IF NOT EXISTS idx_executions_symbol ON executions(symbol);
            CREATE INDEX IF NOT EXISTS idx_executions_trade_id ON executions(trade_id);
            CREATE INDEX IF NOT EXISTS idx_executions_order_id ON executions(order_id);
            CREATE INDEX IF NOT EXISTS idx_executions_perm_id ON executions(perm_id);

            CREATE TABLE IF NOT EXISTS positions (
                position_key TEXT PRIMARY KEY,
                strategy_name TEXT,
                session_date TEXT,
                symbol TEXT,
                status TEXT,
                quantity REAL,
                avg_price REAL,
                source TEXT,
                ibkr_quantity REAL,
                ibkr_avg_cost REAL,
                active INTEGER,
                exit_sent INTEGER,
                updated_at TEXT,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_positions_session_date ON positions(session_date);
            CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_positions_active ON positions(active);

            CREATE TABLE IF NOT EXISTS reconciliation_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT,
                finished_at TEXT,
                mode TEXT,
                clean INTEGER,
                ibkr_positions_count INTEGER,
                managed_positions_count INTEGER,
                orphan_count INTEGER,
                fractional_orphan_count INTEGER,
                drift_count INTEGER,
                pending_orders_count INTEGER,
                details_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_reconciliation_started_at ON reconciliation_runs(started_at);
            CREATE INDEX IF NOT EXISTS idx_reconciliation_mode ON reconciliation_runs(mode);
            CREATE INDEX IF NOT EXISTS idx_reconciliation_clean ON reconciliation_runs(clean);

            CREATE TABLE IF NOT EXISTS risk_events (
                risk_event_id TEXT PRIMARY KEY,
                event_time TEXT,
                severity TEXT,
                category TEXT,
                event_type TEXT,
                strategy_name TEXT,
                session_date TEXT,
                symbol TEXT,
                blocked INTEGER DEFAULT 0,
                reason TEXT,
                daily_pnl REAL,
                gross_exposure REAL,
                active_positions INTEGER,
                trades_today INTEGER,
                action_required INTEGER DEFAULT 0,
                acknowledged INTEGER DEFAULT 0,
                resolved INTEGER DEFAULT 0,
                first_seen_at TEXT,
                last_seen_at TEXT,
                repeat_count INTEGER DEFAULT 1,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_risk_events_event_time ON risk_events(event_time);
            CREATE INDEX IF NOT EXISTS idx_risk_events_severity ON risk_events(severity);
            CREATE INDEX IF NOT EXISTS idx_risk_events_category ON risk_events(category);
            CREATE INDEX IF NOT EXISTS idx_risk_events_event_type ON risk_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_risk_events_resolved ON risk_events(resolved);

            CREATE TABLE IF NOT EXISTS market_data_sessions (
                key TEXT PRIMARY KEY,
                date TEXT,
                session_type TEXT,
                symbol TEXT,
                parquet_path TEXT,
                rows INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                dollar_volume REAL,
                collection_status TEXT,
                collected_at TEXT,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_market_data_sessions_date ON market_data_sessions(date);
            CREATE INDEX IF NOT EXISTS idx_market_data_sessions_symbol ON market_data_sessions(symbol);
            CREATE INDEX IF NOT EXISTS idx_market_data_sessions_collection_status ON market_data_sessions(collection_status);

            CREATE TABLE IF NOT EXISTS symbol_daily_features (
                key TEXT PRIMARY KEY,
                date TEXT,
                symbol TEXT,
                feature_version TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                dollar_volume REAL,
                intraday_high_pct REAL,
                range_pct REAL,
                close_open_pct REAL,
                gap_pct REAL,
                multi_day_return_pct REAL,
                relative_volume REAL,
                median_1m_range_bps REAL,
                avg_abs_1m_return_bps REAL,
                momentum_score REAL,
                liquidity_score REAL,
                final_score REAL,
                rank INTEGER,
                ranking_version TEXT,
                components_json TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_symbol_daily_features_date ON symbol_daily_features(date);
            CREATE INDEX IF NOT EXISTS idx_symbol_daily_features_symbol ON symbol_daily_features(symbol);
            CREATE INDEX IF NOT EXISTS idx_symbol_daily_features_rank ON symbol_daily_features(rank);
            CREATE INDEX IF NOT EXISTS idx_symbol_daily_features_final_score ON symbol_daily_features(final_score);
            CREATE INDEX IF NOT EXISTS idx_symbol_daily_features_feature_version ON symbol_daily_features(feature_version);

            CREATE TABLE IF NOT EXISTS collector_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT,
                finished_at TEXT,
                start_date TEXT,
                end_date TEXT,
                session_type TEXT,
                mode TEXT,
                status TEXT,
                expected_symbols INTEGER,
                processed INTEGER,
                complete INTEGER,
                partial INTEGER,
                no_data INTEGER,
                failed INTEGER,
                parquet_files INTEGER,
                duration_seconds REAL,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_collector_runs_started_at ON collector_runs(started_at);
            CREATE INDEX IF NOT EXISTS idx_collector_runs_status ON collector_runs(status);
            CREATE INDEX IF NOT EXISTS idx_collector_runs_start_date ON collector_runs(start_date);
            CREATE INDEX IF NOT EXISTS idx_collector_runs_end_date ON collector_runs(end_date);
            """
        )
        self.conn.commit()

    def _upsert(self, table: str, row: dict[str, Any], key_columns: list[str], columns: list[str]) -> None:
        clean = clean_row(row, columns)
        col_sql = ", ".join(columns)
        val_sql = ", ".join(f":{col}" for col in columns)
        update_cols = [col for col in columns if col not in key_columns]
        update_sql = ", ".join(f"{col}=excluded.{col}" for col in update_cols)
        self.execute(
            f"INSERT INTO {table} ({col_sql}) VALUES ({val_sql}) "
            f"ON CONFLICT({', '.join(key_columns)}) DO UPDATE SET {update_sql}",
            clean,
        )

    def record_runtime_event(self, **kwargs: Any) -> int:
        now = kwargs.get("event_time") or utc_now_iso()
        row = {
            "event_time": now,
            "severity": kwargs.get("severity") or "INFO",
            "event_type": kwargs.get("event_type") or kwargs.get("event") or "UNKNOWN",
            "strategy_name": kwargs.get("strategy_name"),
            "strategy_version": kwargs.get("strategy_version"),
            "session_date": kwargs.get("session_date") or session_date_utc(),
            "symbol": kwargs.get("symbol"),
            "trade_id": kwargs.get("trade_id"),
            "order_id": kwargs.get("order_id"),
            "execution_id": kwargs.get("execution_id"),
            "source": kwargs.get("source"),
            "reason": kwargs.get("reason"),
            "action_required": bool_int(kwargs.get("action_required")),
            "acknowledged": bool_int(kwargs.get("acknowledged")),
            "resolved": bool_int(kwargs.get("resolved")),
            "first_seen_at": kwargs.get("first_seen_at") or now,
            "last_seen_at": kwargs.get("last_seen_at") or now,
            "repeat_count": safe_int(kwargs.get("repeat_count")) or 1,
            "raw_json": kwargs.get("raw_json"),
        }
        cur = self.execute(
            """
            INSERT INTO runtime_events (
                event_time, severity, event_type, strategy_name, strategy_version, session_date,
                symbol, trade_id, order_id, execution_id, source, reason, action_required,
                acknowledged, resolved, first_seen_at, last_seen_at, repeat_count, raw_json
            ) VALUES (
                :event_time, :severity, :event_type, :strategy_name, :strategy_version, :session_date,
                :symbol, :trade_id, :order_id, :execution_id, :source, :reason, :action_required,
                :acknowledged, :resolved, :first_seen_at, :last_seen_at, :repeat_count, :raw_json
            )
            """,
            clean_row(row, row.keys()),
        )
        return int(cur.lastrowid)

    def record_risk_event(self, **kwargs: Any) -> str:
        now = kwargs.get("event_time") or utc_now_iso()
        risk_event_id = str(
            kwargs.get("risk_event_id")
            or f"{kwargs.get('session_date') or session_date_utc()}:{kwargs.get('category') or 'risk'}:{kwargs.get('event_type') or 'UNKNOWN'}:{kwargs.get('symbol') or ''}:{kwargs.get('reason') or ''}"
        )
        row = {
            "risk_event_id": risk_event_id,
            "event_time": now,
            "severity": kwargs.get("severity") or "WARN",
            "category": kwargs.get("category") or "risk_guard",
            "event_type": kwargs.get("event_type") or "RISK_EVENT",
            "strategy_name": kwargs.get("strategy_name"),
            "session_date": kwargs.get("session_date") or session_date_utc(),
            "symbol": kwargs.get("symbol"),
            "blocked": bool_int(kwargs.get("blocked")),
            "reason": kwargs.get("reason"),
            "daily_pnl": safe_float(kwargs.get("daily_pnl")),
            "gross_exposure": safe_float(kwargs.get("gross_exposure")),
            "active_positions": safe_int(kwargs.get("active_positions")),
            "trades_today": safe_int(kwargs.get("trades_today")),
            "action_required": bool_int(kwargs.get("action_required")),
            "acknowledged": bool_int(kwargs.get("acknowledged")),
            "resolved": bool_int(kwargs.get("resolved")),
            "first_seen_at": kwargs.get("first_seen_at") or now,
            "last_seen_at": now,
            "repeat_count": 1,
            "raw_json": kwargs.get("raw_json"),
        }
        columns = list(row.keys())
        self.execute(
            f"""
            INSERT INTO risk_events ({', '.join(columns)}) VALUES ({', '.join(':' + c for c in columns)})
            ON CONFLICT(risk_event_id) DO UPDATE SET
                event_time=excluded.event_time,
                severity=excluded.severity,
                blocked=excluded.blocked,
                daily_pnl=excluded.daily_pnl,
                gross_exposure=excluded.gross_exposure,
                active_positions=excluded.active_positions,
                trades_today=excluded.trades_today,
                action_required=excluded.action_required,
                last_seen_at=excluded.last_seen_at,
                repeat_count=risk_events.repeat_count + 1,
                raw_json=excluded.raw_json
            """,
            clean_row(row, columns),
        )
        return risk_event_id

    def upsert_execution(self, row: dict[str, Any]) -> None:
        execution_id = str(row.get("execution_id") or "").strip()
        if not execution_id:
            raise ValueError("execution row missing execution_id")
        data = {
            "execution_id": execution_id,
            "trade_id": row.get("trade_id"),
            "order_key": row.get("order_key"),
            "order_id": row.get("order_id"),
            "perm_id": row.get("perm_id"),
            "strategy_name": row.get("strategy_name"),
            "session_date": row.get("session_date") or session_date_utc(),
            "symbol": str(row.get("symbol") or "").upper(),
            "side": row.get("side") or row.get("action") or "",
            "quantity": safe_float(row.get("quantity")),
            "price": safe_float(row.get("price") or row.get("fill_price")),
            "exchange": row.get("exchange"),
            "liquidity": row.get("liquidity"),
            "executed_at": row.get("executed_at") or row.get("execution_time") or raw_execution_time(row.get("raw_json")),
            "recorded_at": row.get("recorded_at") or utc_now_iso(),
            "commission": safe_float(row.get("commission")),
            "commission_currency": row.get("commission_currency"),
            "realized_pnl": safe_float(row.get("realized_pnl")),
            "commission_source": row.get("commission_source"),
            "raw_json": row.get("raw_json") or row,
        }
        columns = list(data.keys())
        self._upsert("executions", data, ["execution_id"], columns)

    def upsert_order(self, row: dict[str, Any]) -> str:
        order_key = str(
            row.get("order_key")
            or row.get("perm_id")
            or row.get("order_id")
            or f"{row.get('client_id') or ''}:{row.get('symbol') or ''}:{row.get('submitted_at') or utc_now_iso()}"
        )
        data = {
            "order_key": order_key,
            "trade_id": row.get("trade_id"),
            "strategy_name": row.get("strategy_name"),
            "session_date": row.get("session_date") or session_date_utc(),
            "symbol": row.get("symbol"),
            "side": row.get("side") or row.get("action"),
            "order_type": row.get("order_type"),
            "quantity": safe_float(row.get("quantity")),
            "limit_price": safe_float(row.get("limit_price")),
            "status": row.get("status"),
            "ibkr_status": row.get("ibkr_status"),
            "order_id": row.get("order_id"),
            "perm_id": row.get("perm_id"),
            "client_id": row.get("client_id"),
            "submitted_at": row.get("submitted_at"),
            "acknowledged_at": row.get("acknowledged_at"),
            "cancelled_at": row.get("cancelled_at"),
            "filled_at": row.get("filled_at"),
            "raw_json": row.get("raw_json") or row,
        }
        self._upsert("orders", data, ["order_key"], list(data.keys()))
        return order_key

    def upsert_trade(self, row: dict[str, Any]) -> str:
        trade_id = str(row.get("trade_id") or uuid.uuid4().hex)
        data = {
            "trade_id": trade_id,
            "strategy_name": row.get("strategy_name") or "unknown",
            "strategy_version": row.get("strategy_version"),
            "session_date": row.get("session_date") or session_date_utc(),
            "symbol": str(row.get("symbol") or "").upper(),
            "status": row.get("status") or "RECONCILING",
            "side": row.get("side") or "LONG",
            "entry_signal_time": row.get("entry_signal_time"),
            "entry_order_time": row.get("entry_order_time"),
            "entry_fill_time": row.get("entry_fill_time"),
            "exit_signal_time": row.get("exit_signal_time"),
            "exit_order_time": row.get("exit_order_time"),
            "exit_fill_time": row.get("exit_fill_time"),
            "closed_at": row.get("closed_at"),
            "entry_price": safe_float(row.get("entry_price")),
            "exit_price": safe_float(row.get("exit_price")),
            "quantity": safe_float(row.get("quantity")),
            "remaining_quantity": safe_float(row.get("remaining_quantity")),
            "gross_pnl": safe_float(row.get("gross_pnl")),
            "commission": safe_float(row.get("commission")),
            "net_pnl": safe_float(row.get("net_pnl")),
            "mfe_pct": safe_float(row.get("mfe_pct")),
            "mae_pct": safe_float(row.get("mae_pct")),
            "exit_reason": row.get("exit_reason"),
            "ibkr_entry_confirmed": bool_int(row.get("ibkr_entry_confirmed")),
            "ibkr_exit_confirmed": bool_int(row.get("ibkr_exit_confirmed")),
            "ibkr_position_flat_confirmed": bool_int(row.get("ibkr_position_flat_confirmed")),
            "ibkr_position_flat_confirmed_at": row.get("ibkr_position_flat_confirmed_at"),
            "raw_json": row.get("raw_json") or row,
        }
        self._upsert("trades", data, ["trade_id"], list(data.keys()))
        return trade_id

    def upsert_position(self, row: dict[str, Any]) -> str:
        strategy = row.get("strategy_name") or "unknown"
        session = row.get("session_date") or session_date_utc()
        symbol = str(row.get("symbol") or "").upper()
        position_key = str(row.get("position_key") or f"{strategy}:{session}:{symbol}")
        data = {
            "position_key": position_key,
            "strategy_name": strategy,
            "session_date": session,
            "symbol": symbol,
            "status": row.get("status") or ("OPEN" if bool_int(row.get("active")) else "CLOSED"),
            "quantity": safe_float(row.get("quantity")),
            "avg_price": safe_float(row.get("avg_price") or row.get("entry_price")),
            "source": row.get("source"),
            "ibkr_quantity": safe_float(row.get("ibkr_quantity")),
            "ibkr_avg_cost": safe_float(row.get("ibkr_avg_cost")),
            "active": bool_int(row.get("active")),
            "exit_sent": bool_int(row.get("exit_sent")),
            "updated_at": row.get("updated_at") or utc_now_iso(),
            "raw_json": row.get("raw_json") or row,
        }
        self._upsert("positions", data, ["position_key"], list(data.keys()))
        return position_key

    def record_reconciliation_run(self, **kwargs: Any) -> str:
        run_id = str(kwargs.get("run_id") or uuid.uuid4().hex)
        data = {
            "run_id": run_id,
            "started_at": kwargs.get("started_at") or utc_now_iso(),
            "finished_at": kwargs.get("finished_at") or utc_now_iso(),
            "mode": kwargs.get("mode"),
            "clean": bool_int(kwargs.get("clean")),
            "ibkr_positions_count": safe_int(kwargs.get("ibkr_positions_count")),
            "managed_positions_count": safe_int(kwargs.get("managed_positions_count")),
            "orphan_count": safe_int(kwargs.get("orphan_count")),
            "fractional_orphan_count": safe_int(kwargs.get("fractional_orphan_count")),
            "drift_count": safe_int(kwargs.get("drift_count")),
            "pending_orders_count": safe_int(kwargs.get("pending_orders_count")),
            "details_json": kwargs.get("details_json") or kwargs.get("raw_json") or kwargs,
        }
        self._upsert("reconciliation_runs", data, ["run_id"], list(data.keys()))
        return run_id

    def upsert_market_data_session(self, row: dict[str, Any]) -> str:
        key = str(row.get("key") or f"{row.get('date')}:{row.get('session_type')}:{row.get('symbol')}")
        data = {
            "key": key,
            "date": row.get("date"),
            "session_type": row.get("session_type"),
            "symbol": row.get("symbol"),
            "parquet_path": row.get("parquet_path"),
            "rows": safe_int(row.get("rows")),
            "open": safe_float(row.get("open")),
            "high": safe_float(row.get("high")),
            "low": safe_float(row.get("low")),
            "close": safe_float(row.get("close")),
            "volume": safe_float(row.get("volume")),
            "dollar_volume": safe_float(row.get("dollar_volume")),
            "collection_status": row.get("collection_status"),
            "collected_at": row.get("collected_at") or utc_now_iso(),
            "raw_json": row.get("raw_json") or row,
        }
        self._upsert("market_data_sessions", data, ["key"], list(data.keys()))
        return key

    def upsert_symbol_daily_feature(self, row: dict[str, Any]) -> str:
        feature_version = str(row.get("feature_version") or "v1")
        key = str(row.get("key") or f"{row.get('date')}:{row.get('symbol')}:{feature_version}")
        data = {
            "key": key,
            "date": row.get("date"),
            "symbol": row.get("symbol"),
            "feature_version": feature_version,
            "open": safe_float(row.get("open")),
            "high": safe_float(row.get("high")),
            "low": safe_float(row.get("low")),
            "close": safe_float(row.get("close") or row.get("last_close")),
            "volume": safe_float(row.get("volume")),
            "dollar_volume": safe_float(row.get("dollar_volume")),
            "intraday_high_pct": safe_float(row.get("intraday_high_pct")),
            "range_pct": safe_float(row.get("range_pct")),
            "close_open_pct": safe_float(row.get("close_open_pct")),
            "gap_pct": safe_float(row.get("gap_pct")),
            "multi_day_return_pct": safe_float(row.get("multi_day_return_pct")),
            "relative_volume": safe_float(row.get("relative_volume")),
            "median_1m_range_bps": safe_float(row.get("median_1m_range_bps")),
            "avg_abs_1m_return_bps": safe_float(row.get("avg_abs_1m_return_bps")),
            "momentum_score": safe_float(row.get("momentum_score")),
            "liquidity_score": safe_float(row.get("liquidity_score")),
            "final_score": safe_float(row.get("final_score") or row.get("score")),
            "rank": safe_int(row.get("rank")),
            "ranking_version": row.get("ranking_version"),
            "components_json": row.get("components_json"),
            "created_at": row.get("created_at") or utc_now_iso(),
        }
        self._upsert("symbol_daily_features", data, ["key"], list(data.keys()))
        return key

    def record_collector_run(self, **kwargs: Any) -> str:
        run_id = str(kwargs.get("run_id") or uuid.uuid4().hex)
        data = {
            "run_id": run_id,
            "started_at": kwargs.get("started_at") or utc_now_iso(),
            "finished_at": kwargs.get("finished_at"),
            "start_date": kwargs.get("start_date"),
            "end_date": kwargs.get("end_date"),
            "session_type": kwargs.get("session_type"),
            "mode": kwargs.get("mode"),
            "status": kwargs.get("status"),
            "expected_symbols": safe_int(kwargs.get("expected_symbols")),
            "processed": safe_int(kwargs.get("processed")),
            "complete": safe_int(kwargs.get("complete")),
            "partial": safe_int(kwargs.get("partial")),
            "no_data": safe_int(kwargs.get("no_data")),
            "failed": safe_int(kwargs.get("failed")),
            "parquet_files": safe_int(kwargs.get("parquet_files")),
            "duration_seconds": safe_float(kwargs.get("duration_seconds")),
            "raw_json": kwargs.get("raw_json") or kwargs,
        }
        self._upsert("collector_runs", data, ["run_id"], list(data.keys()))
        return run_id

    def get_open_trades(self, strategy_name: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM trades WHERE status NOT IN ('CLOSED', 'ERROR')"
        params: list[Any] = []
        if strategy_name:
            sql += " AND strategy_name = ?"
            params.append(strategy_name)
        sql += " ORDER BY session_date DESC, entry_signal_time DESC"
        return self.query(sql, params)

    def get_unresolved_risk_events(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM risk_events WHERE resolved = 0 ORDER BY event_time DESC")

    def get_latest_position(self, symbol: str, strategy_name: str | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM positions WHERE symbol = ?"
        params: list[Any] = [symbol.upper()]
        if strategy_name:
            sql += " AND strategy_name = ?"
            params.append(strategy_name)
        sql += " ORDER BY updated_at DESC LIMIT 1"
        rows = self.query(sql, params)
        return rows[0] if rows else None


def safe_sqlite_call(store: SQLiteRuntimeStore | None, method: str, *args: Any, **kwargs: Any) -> Any:
    if store is None:
        return None
    try:
        return getattr(store, method)(*args, **kwargs)
    except Exception as exc:
        line = f"{utc_now_iso()} SQLITE_WRITE_FAILED method={method} error={exc!r}"
        print(line, flush=True)
        if not unified_logger_installed():
            append_unified_log(line)
        return None
