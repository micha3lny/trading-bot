from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.live_trading.unified_logger import append_unified_log, unified_logger_installed


DEFAULT_SQLITE_PATH = "data/runtime/trading_runtime.sqlite"
SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("TRADING_BOT_SQLITE_BUSY_TIMEOUT_MS", "30000"))
SQLITE_LOCK_RETRY_ATTEMPTS = int(os.environ.get("TRADING_BOT_SQLITE_LOCK_RETRY_ATTEMPTS", "6"))
SQLITE_LOCK_RETRY_BASE_SECONDS = float(os.environ.get("TRADING_BOT_SQLITE_LOCK_RETRY_BASE_SECONDS", "0.05"))
SQLITE_LOCK_RETRY_MAX_SECONDS = float(os.environ.get("TRADING_BOT_SQLITE_LOCK_RETRY_MAX_SECONDS", "1.0"))
SQLITE_WRITER_QUEUE_MAXSIZE = int(os.environ.get("TRADING_BOT_SQLITE_WRITER_QUEUE_MAXSIZE", "10000"))
SQLITE_WRITER_CRITICAL_TIMEOUT_SECONDS = float(os.environ.get("TRADING_BOT_SQLITE_WRITER_CRITICAL_TIMEOUT_SECONDS", "2.0"))
SQLITE_WRITER_BEST_EFFORT_TIMEOUT_SECONDS = float(os.environ.get("TRADING_BOT_SQLITE_WRITER_BEST_EFFORT_TIMEOUT_SECONDS", "0.01"))
SQLITE_WRITER_JOIN_TIMEOUT_SECONDS = float(os.environ.get("TRADING_BOT_SQLITE_WRITER_JOIN_TIMEOUT_SECONDS", "5.0"))
SQLITE_WRITER_TIMEOUT_LOG_INTERVAL_SECONDS = float(os.environ.get("TRADING_BOT_SQLITE_WRITER_TIMEOUT_LOG_INTERVAL_SECONDS", "30.0"))
SQLITE_WRITER_SLOW_WRITE_SECONDS = float(os.environ.get("TRADING_BOT_SQLITE_WRITER_SLOW_WRITE_SECONDS", "5.0"))
SQLITE_WRITER_METHOD_ACK_TIMEOUT_SECONDS = {
    "set_broker_net_positions": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_BROKER_POSITIONS", "8.0")),
    "reconcile_active_positions_to_broker_snapshot": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_RECONCILE", "12.0")),
    "rebuild_positions_from_executions": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_REBUILD", "20.0")),
    "upsert_execution": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_EXECUTION", "8.0")),
    "upsert_position": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_POSITION", "8.0")),
    "upsert_order": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_ORDER", "8.0")),
    "upsert_trade": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_TRADE", "8.0")),
    "finalize_pending_trades": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_FINALIZE", "15.0")),
    "runtime_pending_counts": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_PENDING_COUNTS", "5.0")),
    "mark_operation_status": float(os.environ.get("TRADING_BOT_SQLITE_ACK_TIMEOUT_OPERATION_STATUS", "5.0")),
}
CRITICAL_SQLITE_WRITE_METHODS = {
    "upsert_execution",
    "upsert_order",
    "upsert_trade",
    "upsert_position",
    "record_reconciliation_run",
    "finalize_pending_trades",
    "rebuild_positions_from_executions",
    "reconcile_active_positions_to_broker_snapshot",
    "mark_position_flat",
    "mark_all_positions_flat",
}
COALESCED_SQLITE_WRITE_METHODS = {
    "finalize_pending_trades",
    "mark_operation_status",
    "runtime_pending_counts",
}
BEST_EFFORT_SQLITE_WRITE_METHODS = {
    "mark_operation_status",
    "record_runtime_event",
    "record_risk_event",
    "upsert_market_data_session",
    "upsert_symbol_daily_feature",
    "record_collector_run",
}
HIGH_FREQUENCY_RUNTIME_EVENT_THROTTLE_SECONDS = {
    "BUY_BLOCKED": 300,
    "RISK_GUARD_BLOCK_ENTRY": 300,
    "PEAK_UPDATED": 300,
    "SIGNAL_READY": 300,
    "POSITION_MISSING_IN_IBKR": 300,
    "POSITION_DRIFT_DETECTED": 300,
}
SUMMARY_RUNTIME_EVENT_TYPES = {
    "BUY_BLOCKED": "BUY_BLOCKED_SUMMARY",
}
TERMINAL_COMMISSION_SOURCES = {
    "ibkr",
    "buy_commission_unavailable_after_eod",
    "inferred_missing_buy_commission",
}
FALLBACK_BUY_COMMISSION_SOURCE = "buy_commission_unavailable_after_eod"
ENTRY_METADATA_FIELDS = (
    "top100_rank",
    "top100_score",
    "top100_source_date",
    "top100_features_json",
    "live_entry_score",
    "live_entry_rank",
    "live_entry_features_json",
    "signal_source",
    "signal_time",
    "ready_since",
    "entry_order_id",
    "entry_perm_id",
)
OVERNIGHT_HOLD_TRADE_COLUMNS: dict[str, str] = {
    "overnight_hold_score": "REAL",
    "overnight_hold_bucket": "TEXT",
    "overnight_hold_reason": "TEXT",
    "overnight_hold_features_json": "TEXT",
    "next_session_open": "REAL",
    "next_session_high": "REAL",
    "next_session_low": "REAL",
    "next_session_close": "REAL",
    "next_session_open_gap_pct": "REAL",
    "next_session_high_from_entry_pct": "REAL",
    "next_session_close_from_entry_pct": "REAL",
    "next_session_max_drawdown_from_entry_pct": "REAL",
    "overnight_hold_updated_at": "TEXT",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_date_utc() -> str:
    return date.today().isoformat()


def resolve_sqlite_path(path: str | Path | None = None) -> str:
    return str(path or os.environ.get("TRADING_BOT_SQLITE_PATH") or DEFAULT_SQLITE_PATH)


def open_sqlite_store(path: str | Path | None = None, *, use_writer_queue: bool | None = None) -> SQLiteRuntimeStore | SQLiteWriteQueue | None:
    try:
        resolved = resolve_sqlite_path(path)
        if use_writer_queue is None:
            use_writer_queue = os.environ.get("TRADING_BOT_SQLITE_WRITER_QUEUE", "0").strip().lower() in {"1", "true", "yes", "on"}
        if use_writer_queue:
            return SQLiteWriteQueue(resolved)
        return SQLiteRuntimeStore(resolved)
    except Exception as exc:
        line = f"{utc_now_iso()} SQLITE_WRITE_FAILED method=init error={exc!r}"
        print(line, flush=True)
        if not unified_logger_installed():
            append_unified_log(line)
        return None


def configure_sqlite_connection(conn: sqlite3.Connection, *, read_only: bool = False) -> sqlite3.Connection:
    """Apply runtime pragmas to every SQLite connection.

    WAL only helps if every process is patient enough to wait for short writer
    windows. The busy timeout is intentionally applied to read-only dashboard
    connections too, so they wait instead of failing during live trader commits.
    """
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    if not read_only:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
    return conn


def connect_sqlite(path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    timeout_seconds = max(1.0, SQLITE_BUSY_TIMEOUT_MS / 1000.0)
    if read_only:
        uri = f"file:{Path(path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    else:
        conn = sqlite3.connect(str(path), timeout=timeout_seconds)
    conn.row_factory = sqlite3.Row
    return configure_sqlite_connection(conn, read_only=read_only)


def migrate_runtime_schema(path: str | Path | None = None) -> None:
    store = SQLiteRuntimeStore(resolve_sqlite_path(path))
    store.close()


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


def normalized_execution_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"BOT", "BUY"}:
        return "BUY"
    if text in {"SLD", "SELL"}:
        return "SELL"
    return text


def normalized_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null", "0.0"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def position_source_priority(source: Any, raw_json: Any = None, ibkr_quantity: Any = None) -> int:
    source_text = str(source or "").strip().lower()
    raw = parse_jsonish(raw_json)
    if source_text == "sqlite_execution_reducer":
        return 40
    qty = safe_float(ibkr_quantity)
    if qty is not None and abs(qty) > 1e-9:
        return 30
    if bool_int(raw.get("ibkr_entry_confirmed")) or bool_int(raw.get("ibkr_confirmed")):
        return 25
    if bool_int(raw.get("entry_fill_verified")):
        return 20
    if source_text == "live_buy":
        return 10
    return 0


def iso_date_part(value: Any) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%F")
    except Exception:
        text = str(value or "")
        return text[:10] if len(text) >= 10 else ""


def execution_sort_time(row: dict[str, Any]) -> str:
    return str(row.get("executed_at") or row.get("recorded_at") or "")


def execution_commission(row: dict[str, Any], fraction: float = 1.0) -> float:
    if str(row.get("commission_source") or "").strip().lower() != "ibkr":
        return 0.0
    value = safe_float(row.get("commission"))
    if value is None:
        return 0.0
    return abs(value) * max(0.0, fraction)


def first_safe_float(*values: Any) -> float | None:
    for value in values:
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None


def pct_from_prices(price: float | None, entry_price: float | None) -> float | None:
    if price is None or entry_price is None or entry_price <= 0:
        return None
    return ((price / entry_price) - 1.0) * 100.0


def raw_float_any(raw: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = first_safe_float(raw.get(key))
        if value is not None:
            return value
    return None


def reconstructed_trade_id(entry_date: str, exit_date: str, symbol: str, buy_exec_id: str, sell_exec_id: str) -> str:
    return f"reconstructed:{entry_date}:{exit_date}:{str(symbol or '').upper()}:{buy_exec_id}:{sell_exec_id}"


def clean_row(row: dict[str, Any], columns: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        value = row.get(col)
        if col.endswith("_json") or col == "raw_json" or col == "components_json" or col == "details_json":
            value = safe_json(value)
        out[col] = value
    return out


def is_sqlite_busy_error(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message or "database busy" in message


def log_sqlite_line(line: str) -> None:
    print(line, flush=True)
    if not unified_logger_installed():
        append_unified_log(line)


class SQLiteRuntimeStore:
    def __init__(self, path: str | Path | None = None, *, init: bool = True) -> None:
        self.path = Path(resolve_sqlite_path(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = connect_sqlite(self.path)
        self._transaction_depth = 0
        self._broker_net_positions: dict[str, float] | None = None
        if init:
            self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def set_broker_net_positions(self, positions: dict[str, float] | None) -> None:
        if positions is None:
            self._broker_net_positions = None
            return
        self._broker_net_positions = {
            str(symbol or "").upper().strip(): float(quantity or 0.0)
            for symbol, quantity in positions.items()
            if str(symbol or "").strip()
        }

    def reconcile_active_positions_to_broker_snapshot(self, positions: dict[str, float] | None) -> dict[str, Any]:
        """Constrain current active position rows to a fresh broker snapshot.

        This is intentionally scoped to symbols SQLite currently marks active. A
        broker snapshot can contain symbols whose executions have not been
        ingested yet; those should not be synthesized here. The goal is to clear
        or trim stale active rows once broker truth says they are flat/reduced.
        """
        self.set_broker_net_positions(positions)
        if self._broker_net_positions is None:
            return {"broker_constrained": False, "symbols_processed": 0}
        rows = self.query(
            """
            SELECT DISTINCT UPPER(symbol) AS symbol
            FROM positions
            WHERE COALESCE(active, 0) = 1
              AND COALESCE(symbol, '') != ''
            ORDER BY UPPER(symbol)
            """
        )
        symbols = {str(row.get("symbol") or "").upper() for row in rows if row.get("symbol")}
        symbols.update(
            symbol
            for symbol, quantity in self._broker_net_positions.items()
            if symbol and abs(float(quantity or 0.0)) > 1e-9
        )
        if not symbols:
            return {
                "broker_constrained": True,
                "symbols_processed": 0,
                "open_symbols_count": 0,
                "suppressed_historical_open_symbols_count": 0,
            }
        return self.rebuild_positions_from_executions(sorted(symbols), broker_net_positions=self._broker_net_positions)

    @contextmanager
    def transaction(self):
        outermost = self._transaction_depth == 0
        if outermost:
            self.conn.execute("BEGIN IMMEDIATE")
        self._transaction_depth += 1
        try:
            yield
        except BaseException:
            self._transaction_depth -= 1
            if outermost:
                self.conn.rollback()
            raise
        else:
            self._transaction_depth -= 1
            if outermost:
                self.conn.commit()

    def execute(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        if self._transaction_depth == 0:
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

            CREATE TABLE IF NOT EXISTS runtime_event_counters (
                date TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                count INTEGER DEFAULT 0,
                first_seen_at TEXT,
                last_seen_at TEXT,
                PRIMARY KEY (date, event_type, symbol, reason)
            );
            CREATE INDEX IF NOT EXISTS idx_runtime_event_counters_date ON runtime_event_counters(date);
            CREATE INDEX IF NOT EXISTS idx_runtime_event_counters_event_type ON runtime_event_counters(event_type);
            CREATE INDEX IF NOT EXISTS idx_runtime_event_counters_symbol ON runtime_event_counters(symbol);

            CREATE TABLE IF NOT EXISTS runtime_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_runtime_state_updated_at ON runtime_state(updated_at);

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
                peak_price REAL,
                low_price REAL,
                peak_unrealized_pnl REAL,
                max_adverse_unrealized_pnl REAL,
                giveback_from_peak REAL,
                exit_reason TEXT,
                top100_rank INTEGER,
                top100_score REAL,
                top100_source_date TEXT,
                top100_features_json TEXT,
                live_entry_score REAL,
                live_entry_rank INTEGER,
                live_entry_features_json TEXT,
                signal_source TEXT,
                signal_time TEXT,
                ready_since TEXT,
                entry_order_id TEXT,
                entry_perm_id TEXT,
                overnight_hold_score REAL,
                overnight_hold_bucket TEXT,
                overnight_hold_reason TEXT,
                overnight_hold_features_json TEXT,
                next_session_open REAL,
                next_session_high REAL,
                next_session_low REAL,
                next_session_close REAL,
                next_session_open_gap_pct REAL,
                next_session_high_from_entry_pct REAL,
                next_session_close_from_entry_pct REAL,
                next_session_max_drawdown_from_entry_pct REAL,
                overnight_hold_updated_at TEXT,
                ibkr_entry_confirmed INTEGER DEFAULT 0,
                ibkr_exit_confirmed INTEGER DEFAULT 0,
                ibkr_position_flat_confirmed INTEGER DEFAULT 0,
                ibkr_position_flat_confirmed_at TEXT,
                raw_json TEXT,
                updated_at TEXT,
                trade_reduction_version INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_trades_session_date ON trades(session_date);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy_name ON trades(strategy_name);

            CREATE TABLE IF NOT EXISTS orders (
                order_key TEXT PRIMARY KEY,
                trade_id TEXT,
                position_key TEXT,
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
                exit_reason TEXT,
                exit_reason_source TEXT,
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
                exit_reason TEXT,
                exit_reason_source TEXT,
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
                top100_rank INTEGER,
                top100_score REAL,
                top100_source_date TEXT,
                top100_features_json TEXT,
                live_entry_score REAL,
                live_entry_rank INTEGER,
                live_entry_features_json TEXT,
                signal_source TEXT,
                signal_time TEXT,
                ready_since TEXT,
                entry_order_id TEXT,
                entry_perm_id TEXT,
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
        self._ensure_column("trades", "updated_at", "TEXT")
        self._ensure_column("trades", "trade_reduction_version", "INTEGER DEFAULT 1")
        self._ensure_column("trades", "peak_price", "REAL")
        self._ensure_column("trades", "low_price", "REAL")
        self._ensure_column("trades", "peak_unrealized_pnl", "REAL")
        self._ensure_column("trades", "max_adverse_unrealized_pnl", "REAL")
        self._ensure_column("trades", "giveback_from_peak", "REAL")
        for column, definition in OVERNIGHT_HOLD_TRADE_COLUMNS.items():
            self._ensure_column("trades", column, definition)
        for table in ("trades", "positions"):
            self._ensure_column(table, "top100_rank", "INTEGER")
            self._ensure_column(table, "top100_score", "REAL")
            self._ensure_column(table, "top100_source_date", "TEXT")
            self._ensure_column(table, "top100_features_json", "TEXT")
            self._ensure_column(table, "live_entry_score", "REAL")
            self._ensure_column(table, "live_entry_rank", "INTEGER")
            self._ensure_column(table, "live_entry_features_json", "TEXT")
            self._ensure_column(table, "signal_source", "TEXT")
            self._ensure_column(table, "signal_time", "TEXT")
            self._ensure_column(table, "ready_since", "TEXT")
            self._ensure_column(table, "entry_order_id", "TEXT")
            self._ensure_column(table, "entry_perm_id", "TEXT")
        self._ensure_column("orders", "position_key", "TEXT")
        self._ensure_column("orders", "exit_reason", "TEXT")
        self._ensure_column("orders", "exit_reason_source", "TEXT")
        self._ensure_column("executions", "exit_reason", "TEXT")
        self._ensure_column("executions", "exit_reason_source", "TEXT")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

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

    def _execution_rows_equivalent(self, existing: dict[str, Any] | None, incoming: dict[str, Any]) -> bool:
        if not existing:
            return False
        stable_fields = [
            "execution_id",
            "trade_id",
            "order_key",
            "order_id",
            "perm_id",
            "strategy_name",
            "session_date",
            "symbol",
            "side",
            "quantity",
            "price",
            "exchange",
            "liquidity",
            "executed_at",
            "commission",
            "commission_currency",
            "realized_pnl",
            "commission_source",
            "exit_reason",
            "exit_reason_source",
        ]
        for field in stable_fields:
            left = existing.get(field)
            right = incoming.get(field)
            if field in {"quantity", "price", "commission", "realized_pnl"}:
                left_float = safe_float(left)
                right_float = safe_float(right)
                if left_float is None or right_float is None:
                    if ("" if left in (None, "") else str(left)) != ("" if right in (None, "") else str(right)):
                        return False
                elif abs(left_float - right_float) > 1e-9:
                    return False
            else:
                if ("" if left in (None, "") else str(left)) != ("" if right in (None, "") else str(right)):
                    return False
        return True

    def _entry_metadata_from_trade_row(self, row: dict[str, Any] | sqlite3.Row | None) -> dict[str, Any]:
        if not row:
            return {}
        record = dict(row)
        raw = parse_jsonish(record.get("raw_json"))
        meta: dict[str, Any] = {
            "trade_id": record.get("trade_id"),
            "strategy_name": record.get("strategy_name"),
            "session_date": record.get("session_date"),
        }
        for field in ENTRY_METADATA_FIELDS:
            value = record.get(field)
            if value in (None, ""):
                value = raw.get(field)
            if value not in (None, ""):
                meta[field] = value
        if raw.get("entry_decision_time"):
            meta["entry_decision_time"] = raw.get("entry_decision_time")
        return meta

    def _entry_metadata_for_execution(self, row: dict[str, Any], raw: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = raw or parse_jsonish(row.get("raw_json"))
        symbol = str(row.get("symbol") or raw.get("symbol") or "").upper()
        session = row.get("session_date") or raw.get("session_date") or session_date_utc()
        trade_id = str(row.get("trade_id") or raw.get("trade_id") or "").strip()
        order_id = normalized_identifier(row.get("order_id") or raw.get("order_id") or raw.get("entry_order_id"))
        perm_id = normalized_identifier(row.get("perm_id") or raw.get("perm_id") or raw.get("entry_perm_id"))

        predicates: list[str] = []
        params: list[Any] = []
        if trade_id:
            predicates.append("trade_id = ?")
            params.append(trade_id)
        if order_id:
            predicates.append("COALESCE(entry_order_id, '') = ?")
            params.append(order_id)
        if perm_id:
            predicates.append("COALESCE(entry_perm_id, '') = ?")
            params.append(perm_id)
        if predicates:
            rows = self.query(
                f"""
                SELECT
                    trade_id,
                    strategy_name,
                    session_date,
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
                    raw_json
                FROM trades
                WHERE {' OR '.join(predicates)}
                ORDER BY COALESCE(entry_order_time, entry_signal_time, entry_fill_time, updated_at, '') DESC
                LIMIT 1
                """,
                params,
            )
            if rows:
                return self._entry_metadata_from_trade_row(rows[0])

        if symbol and session:
            rows = self.query(
                """
                SELECT
                    trade_id,
                    strategy_name,
                    session_date,
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
                    raw_json
                FROM trades
                WHERE UPPER(symbol) = ?
                  AND session_date = ?
                  AND (
                    trade_id LIKE 'entry:%'
                    OR UPPER(COALESCE(status, '')) IN ('ENTRY_PENDING', 'ENTRY_PARTIAL', 'OPEN')
                  )
                ORDER BY COALESCE(entry_order_time, entry_signal_time, entry_fill_time, updated_at, '') DESC
                LIMIT 1
                """,
                [symbol, session],
            )
            if rows:
                return self._entry_metadata_from_trade_row(rows[0])
        return {}

    def _merge_entry_metadata_into_raw(self, raw: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        if not meta:
            return raw
        for field in ENTRY_METADATA_FIELDS:
            value = meta.get(field)
            if value not in (None, "") and raw.get(field) in (None, ""):
                raw[field] = value
        if meta.get("entry_decision_time") and raw.get("entry_decision_time") in (None, ""):
            raw["entry_decision_time"] = meta.get("entry_decision_time")
        if meta.get("trade_id") and raw.get("entry_trade_id") in (None, ""):
            raw["entry_trade_id"] = meta.get("trade_id")
        raw.setdefault("entry_metadata_source", "trade_entry_order")
        return raw

    def record_runtime_event(self, **kwargs: Any) -> int:
        now = kwargs.get("event_time") or utc_now_iso()
        event_type = str(kwargs.get("event_type") or kwargs.get("event") or "UNKNOWN")
        event_type_upper = event_type.upper()
        session_date = kwargs.get("session_date") or session_date_utc()
        symbol = str(kwargs.get("symbol") or "").upper()
        reason = str(kwargs.get("reason") or "")
        repeat_count = safe_int(kwargs.get("repeat_count")) or 1
        counter_count = self._increment_runtime_event_counter(
            date=str(session_date or iso_date_part(now) or session_date_utc()),
            event_type=event_type_upper,
            symbol=symbol,
            reason=reason,
            count=repeat_count,
            event_time=str(now),
        )
        should_persist, persisted_event_type = self._should_persist_runtime_event(
            event_type=event_type_upper,
            persisted_event_type=SUMMARY_RUNTIME_EVENT_TYPES.get(event_type_upper, event_type_upper),
            session_date=str(session_date or ""),
            symbol=symbol,
            reason=reason,
            event_time=str(now),
        )
        if not should_persist:
            return 0
        raw_json = kwargs.get("raw_json")
        if persisted_event_type != event_type_upper:
            raw_json = {
                "aggregated_event_type": event_type_upper,
                "count": counter_count,
                "sqlite_throttle_seconds": HIGH_FREQUENCY_RUNTIME_EVENT_THROTTLE_SECONDS.get(event_type_upper),
            }
        row = {
            "event_time": now,
            "severity": kwargs.get("severity") or "INFO",
            "event_type": persisted_event_type,
            "strategy_name": kwargs.get("strategy_name"),
            "strategy_version": kwargs.get("strategy_version"),
            "session_date": session_date,
            "symbol": symbol,
            "trade_id": kwargs.get("trade_id"),
            "order_id": kwargs.get("order_id"),
            "execution_id": kwargs.get("execution_id"),
            "source": kwargs.get("source"),
            "reason": reason,
            "action_required": bool_int(kwargs.get("action_required")),
            "acknowledged": bool_int(kwargs.get("acknowledged")),
            "resolved": bool_int(kwargs.get("resolved")),
            "first_seen_at": kwargs.get("first_seen_at") or now,
            "last_seen_at": kwargs.get("last_seen_at") or now,
            "repeat_count": repeat_count,
            "raw_json": raw_json,
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

    def _increment_runtime_event_counter(self, *, date: str, event_type: str, symbol: str, reason: str, count: int, event_time: str) -> int:
        self.execute(
            """
            INSERT INTO runtime_event_counters (date, event_type, symbol, reason, count, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, event_type, symbol, reason) DO UPDATE SET
                count=runtime_event_counters.count + excluded.count,
                last_seen_at=excluded.last_seen_at
            """,
            [date, event_type, symbol or "", reason or "", count, event_time, event_time],
        )
        row = self.query(
            """
            SELECT count
            FROM runtime_event_counters
            WHERE date = ? AND event_type = ? AND symbol = ? AND reason = ?
            """,
            [date, event_type, symbol or "", reason or ""],
        )
        return safe_int(row[0].get("count")) if row else count

    def _should_persist_runtime_event(
        self,
        *,
        event_type: str,
        persisted_event_type: str,
        session_date: str,
        symbol: str,
        reason: str,
        event_time: str,
    ) -> tuple[bool, str]:
        throttle_seconds = HIGH_FREQUENCY_RUNTIME_EVENT_THROTTLE_SECONDS.get(event_type)
        if not throttle_seconds:
            return True, persisted_event_type
        rows = self.query(
            """
            SELECT last_seen_at
            FROM runtime_events
            WHERE event_type = ?
              AND COALESCE(session_date, '') = ?
              AND COALESCE(symbol, '') = ?
              AND COALESCE(reason, '') = ?
            ORDER BY last_seen_at DESC, event_time DESC
            LIMIT 1
            """,
            [persisted_event_type, session_date or "", symbol or "", reason or ""],
        )
        if not rows:
            return True, persisted_event_type
        try:
            previous = datetime.fromisoformat(str(rows[0].get("last_seen_at") or "").replace("Z", "+00:00"))
            current = datetime.fromisoformat(str(event_time).replace("Z", "+00:00"))
        except Exception:
            return True, persisted_event_type
        return (current - previous).total_seconds() >= throttle_seconds, persisted_event_type

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
        raw = parse_jsonish(row.get("raw_json"))
        side = row.get("side") or row.get("action") or ""
        exit_reason = row.get("exit_reason") or raw.get("exit_reason")
        exit_reason_source = row.get("exit_reason_source") or raw.get("exit_reason_source")
        matched_order: dict[str, Any] | None = None
        if normalized_execution_side(side) == "SELL" and not exit_reason:
            matched_order = self._exit_order_intent_for_execution(row)
            if matched_order:
                order_raw = parse_jsonish(matched_order.get("raw_json"))
                exit_reason = matched_order.get("exit_reason") or order_raw.get("exit_reason")
                exit_reason_source = matched_order.get("exit_reason_source") or order_raw.get("exit_reason_source") or "orders"
                if not row.get("trade_id") and matched_order.get("trade_id"):
                    row = {**row, "trade_id": matched_order.get("trade_id")}
                if not row.get("order_key") and matched_order.get("order_key"):
                    row = {**row, "order_key": matched_order.get("order_key")}
        entry_metadata: dict[str, Any] = {}
        if normalized_execution_side(side) == "BUY":
            entry_metadata = self._entry_metadata_for_execution(row, raw)
            if entry_metadata:
                updates: dict[str, Any] = {}
                if not row.get("trade_id") and entry_metadata.get("trade_id"):
                    updates["trade_id"] = entry_metadata.get("trade_id")
                if not row.get("strategy_name") and entry_metadata.get("strategy_name"):
                    updates["strategy_name"] = entry_metadata.get("strategy_name")
                if not row.get("session_date") and entry_metadata.get("session_date"):
                    updates["session_date"] = entry_metadata.get("session_date")
                if not row.get("order_id") and entry_metadata.get("entry_order_id"):
                    updates["order_id"] = entry_metadata.get("entry_order_id")
                if not row.get("perm_id") and entry_metadata.get("entry_perm_id"):
                    updates["perm_id"] = entry_metadata.get("entry_perm_id")
                if updates:
                    row = {**row, **updates}
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
            "exit_reason": exit_reason,
            "exit_reason_source": exit_reason_source,
            "raw_json": row.get("raw_json") or row,
        }
        raw = parse_jsonish(data.get("raw_json"))
        if exit_reason and not raw.get("exit_reason"):
            raw["exit_reason"] = exit_reason
        if exit_reason_source and not raw.get("exit_reason_source"):
            raw["exit_reason_source"] = exit_reason_source
        if matched_order:
            raw.setdefault("exit_reason_order_key", matched_order.get("order_key"))
            raw.setdefault("exit_reason_order_id", matched_order.get("order_id"))
        if entry_metadata:
            raw = self._merge_entry_metadata_into_raw(raw, entry_metadata)
            raw.setdefault("entry_metadata_source", "trade_entry_order")
        if not raw.get("execution_insert_time") and data.get("recorded_at"):
            raw["execution_insert_time"] = data.get("recorded_at")
        if data.get("commission_source") == "ibkr" and not raw.get("commission_report_time"):
            raw["commission_report_time"] = row.get("commission_report_time") or data.get("recorded_at")
        if data.get("realized_pnl") is not None and not raw.get("realized_pnl_ready_time"):
            raw["realized_pnl_ready_time"] = row.get("realized_pnl_ready_time") or raw.get("commission_report_time") or data.get("recorded_at")
        data["raw_json"] = raw
        columns = list(data.keys())
        with self.transaction():
            existing = self.query("SELECT * FROM executions WHERE execution_id = ?", [execution_id])
            if existing:
                current = existing[0]
                for field in ("trade_id", "order_key", "strategy_name", "session_date", "order_id", "perm_id", "exit_reason", "exit_reason_source"):
                    if data.get(field) in (None, "") and current.get(field) not in (None, ""):
                        data[field] = current.get(field)
                current_commission_source = str(current.get("commission_source") or "").lower()
                incoming_commission_source = str(data.get("commission_source") or "").lower()
                if current_commission_source in TERMINAL_COMMISSION_SOURCES and incoming_commission_source != "ibkr":
                    data["commission_source"] = current.get("commission_source")
                    data["commission"] = current.get("commission")
                    data["commission_currency"] = current.get("commission_currency")
                    if data.get("realized_pnl") is None and current.get("realized_pnl") is not None:
                        data["realized_pnl"] = current.get("realized_pnl")
            if self._execution_rows_equivalent(existing[0] if existing else None, data):
                if self._equivalent_execution_needs_position_reconcile(data):
                    self.rebuild_symbol_trade_state(
                        data["symbol"],
                        allow_historical_open_lots=False,
                        broker_net_positions=self._broker_net_positions,
                    )
                return
            self._upsert("executions", data, ["execution_id"], columns)
            self._reconcile_trade_state_after_execution(data)

    def set_runtime_state(self, key: str, value: Any = None, raw_json: Any = None, *, updated_at: str | None = None) -> None:
        key = str(key or "").strip()
        if not key:
            raise ValueError("runtime_state key is required")
        now = updated_at or utc_now_iso()
        payload = parse_jsonish(raw_json)
        if not payload and raw_json not in (None, ""):
            payload = {"value": raw_json}
        row = {
            "key": key,
            "value": "" if value is None else str(value),
            "updated_at": now,
            "raw_json": payload,
        }
        self._upsert("runtime_state", row, ["key"], list(row.keys()))

    def get_runtime_state(self, keys: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        if keys is None:
            rows = self.query("SELECT key, value, updated_at, raw_json FROM runtime_state ORDER BY key")
        else:
            key_list = [str(key) for key in keys if str(key or "").strip()]
            if not key_list:
                return {}
            placeholders = ",".join("?" for _ in key_list)
            rows = self.query(
                f"SELECT key, value, updated_at, raw_json FROM runtime_state WHERE key IN ({placeholders}) ORDER BY key",
                key_list,
            )
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            out[str(row.get("key") or "")] = {
                "value": row.get("value"),
                "updated_at": row.get("updated_at"),
                "raw_json": parse_jsonish(row.get("raw_json")),
            }
        return out

    def mark_operation_status(self, name: str, status: str, **details: Any) -> None:
        name = str(name or "").strip()
        status = str(status or "").strip() or "unknown"
        if not name:
            raise ValueError("operation status name is required")
        now = utc_now_iso()
        current = self.get_runtime_state([name]).get(name, {}).get("raw_json", {})
        payload = dict(current)
        if status == "running":
            payload["started_at"] = details.pop("started_at", None) or now
            payload["finished_at"] = ""
            payload["error"] = ""
        else:
            payload.setdefault("started_at", details.get("started_at") or "")
            payload["finished_at"] = details.pop("finished_at", None) or now
        payload["status"] = status
        payload["updated_at"] = now
        payload.update(details)
        self.set_runtime_state(name, status, payload, updated_at=now)

    def _exit_order_intent_for_execution(self, row: dict[str, Any]) -> dict[str, Any] | None:
        order_id = normalized_identifier(row.get("order_id"))
        perm_id = normalized_identifier(row.get("perm_id"))
        symbol = str(row.get("symbol") or "").upper().strip()
        params: list[Any] = []
        predicates: list[str] = []
        if order_id:
            predicates.append("COALESCE(order_id, '') = ?")
            params.append(order_id)
        if perm_id:
            predicates.append("COALESCE(perm_id, '') = ?")
            params.append(perm_id)
        if not predicates:
            return None
        symbol_clause = ""
        if symbol:
            symbol_clause = " AND UPPER(COALESCE(symbol, '')) = ?"
            params.append(symbol)
        rows = self.query(
            f"""
            SELECT *
            FROM orders
            WHERE ({' OR '.join(predicates)})
              {symbol_clause}
              AND UPPER(COALESCE(side, '')) IN ('SELL', 'SLD')
              AND COALESCE(exit_reason, '') != ''
            ORDER BY COALESCE(submitted_at, acknowledged_at, filled_at, '') DESC
            LIMIT 1
            """,
            params,
        )
        return rows[0] if rows else None

    def record_exit_order_intent(
        self,
        *,
        order_id: Any,
        symbol: str,
        exit_reason: str,
        quantity: Any = None,
        submitted_at: str | None = None,
        trade_id: str = "",
        position_key: str = "",
        strategy_name: str = "",
        session_date: str = "",
        raw_json: Any = None,
    ) -> str:
        submitted_at = submitted_at or utc_now_iso()
        raw = parse_jsonish(raw_json)
        raw.update(
            {
                "exit_reason": exit_reason,
                "exit_reason_source": "exit_order_submit",
                "exit_reason_persisted_at": submitted_at,
            }
        )
        return self.upsert_order(
            {
                "order_key": f"exit:{normalized_identifier(order_id) or uuid.uuid4().hex}",
                "trade_id": trade_id,
                "position_key": position_key,
                "strategy_name": strategy_name or "unknown",
                "session_date": session_date or session_date_utc(),
                "symbol": str(symbol or "").upper(),
                "side": "SELL",
                "order_type": "MKT",
                "quantity": quantity,
                "status": "SUBMITTED",
                "ibkr_status": "Submitted",
                "order_id": normalized_identifier(order_id),
                "submitted_at": submitted_at,
                "exit_reason": exit_reason,
                "exit_reason_source": "exit_order_submit",
                "raw_json": raw,
            }
        )

    def runtime_pending_counts(self, session_date: str | None = None) -> dict[str, Any]:
        where = ""
        params: list[Any] = []
        if session_date:
            where = "WHERE COALESCE(substr(executed_at, 1, 10), substr(recorded_at, 1, 10), session_date) = ?"
            params.append(session_date)
        execution_rows = self.query(
            f"""
            SELECT
                SUM(CASE
                    WHEN COALESCE(execution_id, '') = ''
                         OR COALESCE(symbol, '') = ''
                         OR COALESCE(side, '') = ''
                         OR quantity IS NULL
                         OR price IS NULL
                    THEN 1 ELSE 0 END
                ) AS pending_execution_count,
                SUM(CASE
                    WHEN COALESCE(commission_source, '') NOT IN ('ibkr', 'buy_commission_unavailable_after_eod', 'inferred_missing_buy_commission')
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
            {where}
            """,
            params,
        )
        trade_where = ""
        trade_params: list[Any] = []
        if session_date:
            trade_where = """
            WHERE (
                substr(exit_fill_time, 1, 10) = ?
                OR substr(closed_at, 1, 10) = ?
                OR (
                    COALESCE(exit_fill_time, closed_at) IS NULL
                    AND session_date = ?
                )
            )
            """
            trade_params.extend([session_date, session_date, session_date])
        trade_rows = self.query(
            f"""
            SELECT COUNT(*) AS pending_trade_finalization_count
            FROM trades
            {trade_where}
            {"AND" if trade_where else "WHERE"} UPPER(COALESCE(status, '')) IN ('COMMISSION_PENDING', 'PNL_PENDING', 'RECONCILE_PENDING')
            """,
            trade_params,
        )
        exec_row = execution_rows[0] if execution_rows else {}
        trade_row = trade_rows[0] if trade_rows else {}
        return {
            "session_date": session_date or "",
            "pending_execution_count": int(exec_row.get("pending_execution_count") or 0),
            "pending_commission_count": int(exec_row.get("pending_commission_count") or 0),
            "pending_realized_pnl_count": int(exec_row.get("pending_realized_pnl_count") or 0),
            "pending_trade_finalization_count": int(trade_row.get("pending_trade_finalization_count") or 0),
        }

    def pending_trade_finalization_diagnostics(self, session_date: str | None = None) -> list[dict[str, Any]]:
        """Explain why pending closed trade rows are not finalized.

        Pending trade rows can outlive execution ingestion when the trade was
        first reduced before commission reports arrived. This diagnostic reads
        the reducer's raw execution-pair metadata and compares it with the
        current executions table so operators can distinguish a real missing
        commission from a stale trade row that simply needs a symbol rebuild.
        """
        where = "WHERE UPPER(COALESCE(status, '')) IN ('COMMISSION_PENDING', 'PNL_PENDING', 'RECONCILE_PENDING')"
        params: list[Any] = []
        if session_date:
            where += """
              AND (
                  substr(exit_fill_time, 1, 10) = ?
                  OR substr(closed_at, 1, 10) = ?
                  OR (
                      COALESCE(exit_fill_time, closed_at) IS NULL
                      AND session_date = ?
                  )
              )
            """
            params.extend([session_date, session_date, session_date])
        trades = self.query(
            f"""
            SELECT trade_id, symbol, status, session_date, entry_fill_time,
                   exit_fill_time, closed_at, raw_json
            FROM trades
            {where}
            ORDER BY COALESCE(exit_fill_time, closed_at, ''), symbol, trade_id
            """,
            params,
        )
        diagnostics: list[dict[str, Any]] = []
        for trade in trades:
            raw = parse_jsonish(trade.get("raw_json"))
            buy_execution_id = str(raw.get("buy_execution_id") or raw.get("entry_execution_id") or "").strip()
            sell_execution_id = str(raw.get("sell_execution_id") or raw.get("exit_execution_id") or "").strip()
            exec_rows: dict[str, dict[str, Any]] = {}
            ids = [execution_id for execution_id in (buy_execution_id, sell_execution_id) if execution_id]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                exec_rows = {
                    str(row.get("execution_id") or ""): row
                    for row in self.query(
                        f"""
                        SELECT execution_id, symbol, side, commission, commission_source,
                               realized_pnl, executed_at, recorded_at
                        FROM executions
                        WHERE execution_id IN ({placeholders})
                        """,
                        ids,
                    )
                }
            blockers: list[str] = []
            buy_exec = exec_rows.get(buy_execution_id) if buy_execution_id else None
            sell_exec = exec_rows.get(sell_execution_id) if sell_execution_id else None
            if buy_execution_id and buy_exec is None:
                blockers.append("BUY_EXECUTION_MISSING")
            if sell_execution_id and sell_exec is None:
                blockers.append("SELL_EXECUTION_MISSING")
            if buy_exec is not None:
                buy_source = str(buy_exec.get("commission_source") or "")
                if buy_source not in TERMINAL_COMMISSION_SOURCES or safe_float(buy_exec.get("commission")) is None:
                    blockers.append("BUY_COMMISSION_NOT_IBKR")
            else:
                buy_source = ""
            if sell_exec is not None:
                sell_source = str(sell_exec.get("commission_source") or "")
                if sell_source != "ibkr" or safe_float(sell_exec.get("commission")) is None:
                    blockers.append("SELL_COMMISSION_NOT_IBKR")
                if safe_float(sell_exec.get("realized_pnl")) is None:
                    blockers.append("SELL_REALIZED_PNL_MISSING")
            else:
                sell_source = ""
            if not blockers:
                blockers.append("STALE_PENDING_TRADE_NEEDS_REBUILD")
            diagnostics.append(
                {
                    "trade_id": trade.get("trade_id"),
                    "symbol": trade.get("symbol"),
                    "status": trade.get("status"),
                    "entry_fill_time": trade.get("entry_fill_time"),
                    "exit_fill_time": trade.get("exit_fill_time") or trade.get("closed_at"),
                    "buy_execution_id": buy_execution_id,
                    "sell_execution_id": sell_execution_id,
                    "buy_commission_source": buy_source,
                    "buy_commission": buy_exec.get("commission") if buy_exec else None,
                    "sell_commission_source": sell_source,
                    "sell_commission": sell_exec.get("commission") if sell_exec else None,
                    "sell_realized_pnl": sell_exec.get("realized_pnl") if sell_exec else None,
                    "blockers": ",".join(blockers),
                }
            )
        return diagnostics

    def finalize_pending_trades(self, session_date: str | None = None) -> dict[str, Any]:
        """Rebuild symbols that still have pending trade rows.

        This is intentionally conservative: it does not invent fills or
        commissions. It re-runs the deterministic execution reducer for the
        affected symbols, then returns before/after diagnostics so callers can
        see which condition remains if a trade is still pending.
        """
        requested_session_date = str(session_date or "").strip()
        full_rebuild = not bool(requested_session_date)
        if full_rebuild and os.environ.get("TRADING_BOT_ALLOW_FULL_PENDING_TRADE_REBUILD") != "1":
            requested_session_date = datetime.now(timezone.utc).date().isoformat()
            full_rebuild = False
            print(
                f"{utc_now_iso()} SQLITE_HEAVY_REBUILD_DEFERRED method=finalize_pending_trades "
                f"requested_scope=all effective_session_date={requested_session_date} "
                f"reason=full_pending_trade_rebuild_requires_TRADING_BOT_ALLOW_FULL_PENDING_TRADE_REBUILD",
                flush=True,
            )
        print(
            f"{utc_now_iso()} FINALIZE_PENDING_TRADES_SCOPE "
            f"session_date={requested_session_date or 'all'} full_rebuild={int(full_rebuild)}",
            flush=True,
        )
        session_date = requested_session_date or None
        before = self.pending_trade_finalization_diagnostics(session_date)
        symbols = sorted({str(row.get("symbol") or "").upper() for row in before if str(row.get("symbol") or "").strip()})
        if symbols:
            self.rebuild_positions_from_executions(symbols, allow_historical_open_lots=False, broker_net_positions=self._broker_net_positions)
        after = self.pending_trade_finalization_diagnostics(session_date)
        return {
            "session_date": session_date or "",
            "full_rebuild": full_rebuild,
            "symbols_processed": symbols,
            "pending_before": len(before),
            "pending_after": len(after),
            "resolved": max(0, len(before) - len(after)),
            "before": before,
            "after": after,
        }

    def pending_buy_commission_executions(self, session_date: str | None = None) -> list[dict[str, Any]]:
        diagnostics = self.pending_trade_finalization_diagnostics(session_date)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in diagnostics:
            blockers = {part.strip() for part in str(row.get("blockers") or "").split(",") if part.strip()}
            execution_id = str(row.get("buy_execution_id") or "").strip()
            if execution_id and "BUY_COMMISSION_NOT_IBKR" in blockers and execution_id not in seen:
                out.append({
                    "execution_id": execution_id,
                    "session_date": row.get("session_date"),
                    "symbol": row.get("symbol"),
                    "trade_id": row.get("trade_id"),
                })
                seen.add(execution_id)
        return out

    def pending_buy_commission_execution_ids(self, session_date: str | None = None) -> list[str]:
        return [str(row.get("execution_id") or "") for row in self.pending_buy_commission_executions(session_date)]

    def _reconcile_trade_state_after_execution(self, execution: dict[str, Any]) -> None:
        try:
            symbol = str(execution.get("symbol") or "").upper().strip()
            if not symbol:
                return
            side = normalized_execution_side(execution.get("side"))
            broker_net_positions = self._broker_net_positions if side == "SELL" else None
            self.rebuild_symbol_trade_state(
                symbol,
                allow_historical_open_lots=False,
                broker_net_positions=broker_net_positions,
            )
        except Exception as exc:
            line = f"{utc_now_iso()} SQLITE_WRITE_FAILED method=rebuild_symbol_trade_state error={exc!r}"
            print(line, flush=True)
            if not unified_logger_installed():
                append_unified_log(line)

    def _active_quantity_for_symbol(self, symbol: str) -> float:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return 0.0
        rows = self.query(
            """
            SELECT SUM(COALESCE(quantity, 0)) AS quantity
            FROM positions
            WHERE UPPER(symbol) = ?
              AND COALESCE(active, 0) = 1
              AND UPPER(COALESCE(status, '')) IN ('OPEN', 'EXIT_ORDER')
            """,
            [symbol],
        )
        return safe_float(rows[0].get("quantity") if rows else 0.0) or 0.0

    def _equivalent_execution_needs_position_reconcile(self, execution: dict[str, Any]) -> bool:
        """Return True when a duplicate execution should repair missing active state.

        Duplicate/equivalent execution rows usually skip the reducer to avoid
        churn. During live broker reconciliation, though, the broker snapshot can
        prove a symbol is open while SQLite has no matching active row. In that
        case a re-seen execution is useful repair input and should rebuild the
        symbol state.
        """
        if self._broker_net_positions is None:
            return False
        symbol = str(execution.get("symbol") or "").upper().strip()
        if not symbol:
            return False
        broker_qty = safe_float(self._broker_net_positions.get(symbol))
        if broker_qty is None:
            broker_qty = 0.0
        sqlite_qty = self._active_quantity_for_symbol(symbol)
        return abs(sqlite_qty - broker_qty) > 1e-9

    def _closed_trade_id_for_pair(self, buy: dict[str, Any], sell: dict[str, Any], matched_qty: float) -> str:
        symbol = str(buy.get("symbol") or sell.get("symbol") or "").upper()
        buy_time = execution_sort_time(buy)
        sell_time = execution_sort_time(sell)
        entry_date = iso_date_part(buy_time or buy.get("session_date"))
        exit_date = iso_date_part(sell_time or sell.get("session_date"))
        buy_exec_id = str(buy.get("execution_id") or "")
        sell_exec_id = str(sell.get("execution_id") or "")
        return reconstructed_trade_id(entry_date, exit_date, symbol, buy_exec_id, sell_exec_id)

    def _clear_reconstructed_trades_for_symbol(self, symbol: str) -> int:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return 0
        rows = self.query(
            """
            SELECT trade_id
            FROM trades
            WHERE UPPER(symbol) = ?
              AND (
                trade_id LIKE 'reconstructed:%'
                OR raw_json LIKE '%sqlite_execution_reducer%'
                OR raw_json LIKE '%executions_pair_repair%'
              )
            """,
            [symbol],
        )
        trade_ids = [str(row.get("trade_id") or "") for row in rows if row.get("trade_id")]
        if not trade_ids:
            return 0
        placeholders = ",".join("?" for _ in trade_ids)
        self.execute(f"UPDATE executions SET trade_id = NULL WHERE trade_id IN ({placeholders})", trade_ids)
        self.execute(f"DELETE FROM trades WHERE trade_id IN ({placeholders})", trade_ids)
        return len(trade_ids)

    def _delete_duplicate_reconstructed_execution_pair(self, trade_id: str, symbol: str, raw_json: Any, quantity: float | None) -> int:
        raw = parse_jsonish(raw_json)
        buy_exec = str(raw.get("buy_execution_id") or "")
        sell_exec = str(raw.get("sell_execution_id") or "")
        if not buy_exec or not sell_exec:
            return 0
        rows = self.query(
            """
            SELECT trade_id, quantity, raw_json
            FROM trades
            WHERE trade_id != ?
              AND UPPER(symbol) = ?
              AND UPPER(COALESCE(status, '')) IN ('CLOSED', 'DONE', 'EXIT_FILLED', 'FLAT')
            """,
            [trade_id, str(symbol or "").upper()],
        )
        duplicate_ids: list[str] = []
        qty = safe_float(quantity) or 0.0
        for row in rows:
            other_raw = parse_jsonish(row.get("raw_json"))
            if str(other_raw.get("buy_execution_id") or "") != buy_exec:
                continue
            if str(other_raw.get("sell_execution_id") or "") != sell_exec:
                continue
            other_qty = safe_float(row.get("quantity")) or 0.0
            if abs(other_qty - qty) > 1e-9:
                continue
            other_trade_id = str(row.get("trade_id") or "")
            if other_trade_id.startswith("reconstructed:"):
                duplicate_ids.append(other_trade_id)
        if not duplicate_ids:
            return 0
        placeholders = ",".join("?" for _ in duplicate_ids)
        self.execute(f"UPDATE executions SET trade_id = NULL WHERE trade_id IN ({placeholders})", duplicate_ids)
        self.execute(f"DELETE FROM trades WHERE trade_id IN ({placeholders})", duplicate_ids)
        return len(duplicate_ids)

    def _latest_position_excursion_raw(self, symbol: str, buy_execution_id: Any = None) -> dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return {}
        rows = self.query(
            """
            SELECT raw_json
            FROM positions
            WHERE UPPER(symbol) = ?
            ORDER BY COALESCE(active, 0) DESC, COALESCE(updated_at, '') DESC, rowid DESC
            LIMIT 10
            """,
            [symbol],
        )
        buy_execution_id = str(buy_execution_id or "")
        fallback: dict[str, Any] = {}
        for row in rows:
            raw = parse_jsonish(row.get("raw_json"))
            if not raw:
                continue
            if not fallback:
                fallback = raw
            open_ids = {str(value) for value in raw.get("open_lot_execution_ids") or []}
            if buy_execution_id and buy_execution_id in open_ids:
                return raw
        return fallback

    @staticmethod
    def _trade_excursion_stats(
        *,
        entry_price: float | None,
        exit_price: float | None,
        quantity: float | None,
        net_pnl: float | None,
        raw_sources: Iterable[dict[str, Any]],
    ) -> dict[str, float | None]:
        if entry_price is None or entry_price <= 0:
            return {
                "mfe_pct": None,
                "mae_pct": None,
                "peak_price": None,
                "low_price": None,
                "peak_unrealized_pnl": None,
                "max_adverse_unrealized_pnl": None,
                "giveback_from_peak": None,
            }
        peak_candidates = [entry_price]
        low_candidates = [entry_price]
        if exit_price is not None and exit_price > 0:
            peak_candidates.append(exit_price)
            low_candidates.append(exit_price)
        raw_peak_pnl: float | None = None
        raw_adverse_pnl: float | None = None
        raw_mfe_pct: float | None = None
        raw_mae_pct: float | None = None
        for raw in raw_sources:
            if not raw:
                continue
            peak = raw_float_any(raw, "peak_price", "peak_price_since_entry", "high_watermark", "mfe_price")
            low = raw_float_any(raw, "low_price", "low_price_since_entry", "low_watermark", "mae_price")
            raw_mfe_pct = first_safe_float(raw_mfe_pct, raw_float_any(raw, "mfe_pct", "peak_pct", "peak_unrealized_pct", "peak_gain_pct", "max_gain_pct"))
            raw_mae_pct = first_safe_float(raw_mae_pct, raw_float_any(raw, "mae_pct", "low_pct", "max_adverse_pct", "max_adverse_unrealized_pct"))
            raw_peak_pnl = first_safe_float(raw_peak_pnl, raw_float_any(raw, "peak_unrealized_pnl", "max_unrealized_pnl"))
            raw_adverse_pnl = first_safe_float(raw_adverse_pnl, raw_float_any(raw, "max_adverse_unrealized_pnl", "mae_unrealized_pnl", "min_unrealized_pnl"))
            if peak is not None and peak > 0:
                peak_candidates.append(peak)
            if low is not None and low > 0:
                low_candidates.append(low)
        if raw_mfe_pct is not None:
            peak_candidates.append(entry_price * (1.0 + raw_mfe_pct / 100.0))
        if raw_mae_pct is not None:
            low_candidates.append(entry_price * (1.0 + raw_mae_pct / 100.0))
        peak_price = max(peak_candidates) if peak_candidates else None
        low_price = min(low_candidates) if low_candidates else None
        mfe_pct = pct_from_prices(peak_price, entry_price)
        mae_pct = pct_from_prices(low_price, entry_price)
        qty = quantity or 0.0
        price_peak_pnl = (peak_price - entry_price) * qty if peak_price is not None else None
        price_adverse_pnl = (low_price - entry_price) * qty if low_price is not None else None
        peak_unrealized_pnl = max(value for value in (raw_peak_pnl, price_peak_pnl) if value is not None) if raw_peak_pnl is not None or price_peak_pnl is not None else None
        max_adverse_unrealized_pnl = min(value for value in (raw_adverse_pnl, price_adverse_pnl) if value is not None) if raw_adverse_pnl is not None or price_adverse_pnl is not None else None
        giveback_from_peak = None
        if peak_unrealized_pnl is not None and net_pnl is not None:
            giveback_from_peak = peak_unrealized_pnl - net_pnl
        return {
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "peak_price": peak_price,
            "low_price": low_price,
            "peak_unrealized_pnl": peak_unrealized_pnl,
            "max_adverse_unrealized_pnl": max_adverse_unrealized_pnl,
            "giveback_from_peak": giveback_from_peak,
        }

    def _broker_flat_for_symbol(self, symbol: str, broker_net_positions: dict[str, float] | None) -> bool:
        if broker_net_positions is None:
            return False
        target = safe_float(broker_net_positions.get(str(symbol or "").upper().strip()))
        return abs(target or 0.0) <= 1e-9

    def _mark_buy_commission_unavailable_after_eod(self, execution_id: Any) -> None:
        execution_id = str(execution_id or "").strip()
        if not execution_id:
            return
        rows = self.query("SELECT commission, commission_source, raw_json FROM executions WHERE execution_id = ?", [execution_id])
        if not rows:
            return
        current = rows[0]
        source = str(current.get("commission_source") or "").lower()
        if source in TERMINAL_COMMISSION_SOURCES and safe_float(current.get("commission")) is not None:
            return
        raw = parse_jsonish(current.get("raw_json"))
        raw["commission_fallback_source"] = FALLBACK_BUY_COMMISSION_SOURCE
        raw["commission_fallback_time"] = utc_now_iso()
        raw["commission_fallback_reason"] = "broker_flat_sell_realized_buy_commission_unavailable"
        self.execute(
            """
            UPDATE executions
            SET commission = COALESCE(commission, 0),
                commission_source = ?,
                raw_json = ?
            WHERE execution_id = ?
            """,
            [FALLBACK_BUY_COMMISSION_SOURCE, safe_json(raw), execution_id],
        )

    def rebuild_symbol_trade_state(
        self,
        symbol: str,
        strategy_name: str | None = None,
        *,
        allow_historical_open_lots: bool = False,
        broker_net_positions: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if self._transaction_depth == 0:
            with self.transaction():
                return self.rebuild_symbol_trade_state(
                    symbol,
                    strategy_name,
                    allow_historical_open_lots=allow_historical_open_lots,
                    broker_net_positions=broker_net_positions,
                )
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return {"symbol": symbol, "closed_trades": 0, "open_quantity": 0.0}
        strategy_clause = ""
        params: list[Any] = [symbol]
        if strategy_name:
            strategy_clause = "AND COALESCE(strategy_name, 'unknown') = ?"
            params.append(strategy_name)
        rows = self.query(
            f"""
            SELECT execution_id, trade_id, order_key, order_id, perm_id,
                   COALESCE(strategy_name, 'unknown') AS strategy_name,
                   session_date, symbol, side, quantity, price, exchange, liquidity,
                   executed_at, recorded_at, commission, commission_currency,
                   realized_pnl, commission_source, exit_reason, exit_reason_source, raw_json
            FROM executions
            WHERE UPPER(symbol) = ? {strategy_clause}
            ORDER BY COALESCE(executed_at, recorded_at, ''), execution_id
            """,
            params,
        )
        reducer_started_at = utc_now_iso()
        self._clear_reconstructed_trades_for_symbol(symbol)
        open_lots: list[dict[str, Any]] = []
        closed_count = 0
        last_event_time = utc_now_iso()
        latest_strategy = strategy_name or "unknown"
        latest_session = session_date_utc()
        latest_price: float | None = None

        for row in rows:
            side = normalized_execution_side(row.get("side"))
            qty = safe_float(row.get("quantity")) or 0.0
            price = safe_float(row.get("price")) or 0.0
            if qty <= 0 or price <= 0:
                continue
            event_time = execution_sort_time(row) or utc_now_iso()
            last_event_time = event_time
            latest_strategy = str(row.get("strategy_name") or latest_strategy or "unknown")
            latest_session = str(row.get("session_date") or "") or iso_date_part(event_time) or latest_session or session_date_utc()
            latest_price = price
            if side == "BUY":
                lot = dict(row)
                lot["remaining_qty"] = qty
                lot["original_qty"] = qty
                open_lots.append(lot)
                continue
            if side != "SELL":
                continue

            remaining = qty
            while remaining > 1e-9 and open_lots:
                lot = open_lots[0]
                lot_remaining = safe_float(lot.get("remaining_qty")) or 0.0
                if lot_remaining <= 1e-9:
                    open_lots.pop(0)
                    continue
                matched_qty = min(remaining, lot_remaining)
                buy_price = safe_float(lot.get("price")) or 0.0
                sell_price = price
                buy_time = execution_sort_time(lot)
                sell_time = event_time
                entry_date = str(lot.get("session_date") or "") or iso_date_part(buy_time) or latest_session
                exit_date = str(row.get("session_date") or "") or iso_date_part(sell_time) or latest_session
                buy_fraction = matched_qty / (safe_float(lot.get("original_qty")) or matched_qty or 1.0)
                sell_fraction = matched_qty / qty if qty else 1.0
                buy_commission_source = str(lot.get("commission_source") or "").lower()
                sell_commission_source = str(row.get("commission_source") or "").lower()
                buy_commission_confirmed = buy_commission_source == "ibkr"
                sell_commission_confirmed = sell_commission_source == "ibkr"
                buy_commission_fallback = buy_commission_source in (TERMINAL_COMMISSION_SOURCES - {"ibkr"})
                buy_commission_missing = buy_commission_source == "missing"
                sell_commission_missing = sell_commission_source == "missing"
                commission = execution_commission(lot, buy_fraction) + execution_commission(row, sell_fraction)
                gross = (sell_price - buy_price) * matched_qty
                net_pnl = gross - commission
                trade_id = self._closed_trade_id_for_pair(lot, row, matched_qty)
                strategy = str(lot.get("strategy_name") or row.get("strategy_name") or latest_strategy or "unknown")
                buy_raw = parse_jsonish(lot.get("raw_json"))
                sell_raw = parse_jsonish(row.get("raw_json"))
                position_raw = self._latest_position_excursion_raw(symbol, lot.get("execution_id"))
                excursion = self._trade_excursion_stats(
                    entry_price=buy_price,
                    exit_price=sell_price,
                    quantity=matched_qty,
                    net_pnl=net_pnl,
                    raw_sources=(position_raw, buy_raw, sell_raw),
                )
                sell_realized_expected = sell_commission_confirmed or sell_commission_missing
                sell_realized_ready = safe_float(row.get("realized_pnl")) is not None
                broker_flat_for_symbol = self._broker_flat_for_symbol(symbol, broker_net_positions)
                buy_commission_unavailable_after_eod = (
                    buy_commission_missing
                    and sell_commission_confirmed
                    and sell_realized_ready
                    and broker_flat_for_symbol
                )
                if buy_commission_unavailable_after_eod:
                    self._mark_buy_commission_unavailable_after_eod(lot.get("execution_id"))
                    lot["commission"] = 0.0
                    lot["commission_source"] = FALLBACK_BUY_COMMISSION_SOURCE
                    buy_commission_missing = False
                    buy_commission_fallback = True
                    commission = execution_commission(lot, buy_fraction) + execution_commission(row, sell_fraction)
                    net_pnl = gross - commission
                pending_commission_count = int(buy_commission_missing) + int(sell_commission_missing)
                pending_realized_pnl_count = 1 if sell_realized_expected and not sell_realized_ready else 0
                if pending_commission_count:
                    trade_status = "COMMISSION_PENDING"
                elif pending_realized_pnl_count:
                    trade_status = "PNL_PENDING"
                else:
                    trade_status = "CLOSED"
                pnl_status = "REALIZED_PNL_READY" if sell_realized_ready else "PNL_PENDING"
                if pending_commission_count == 0 and buy_commission_confirmed and sell_commission_confirmed:
                    commission_status = "OK"
                elif pending_commission_count == 0 and buy_commission_fallback and sell_commission_confirmed:
                    commission_status = "BUY_FALLBACK"
                elif pending_commission_count == 0:
                    commission_status = "UNKNOWN"
                else:
                    commission_status = "PARTIAL" if pending_commission_count == 1 else "MISSING"
                commission_report_times = [
                    str(value)
                    for value in (buy_raw.get("commission_report_time"), sell_raw.get("commission_report_time"))
                    if value not in (None, "")
                ]
                realized_pnl_ready_times = [
                    str(value)
                    for value in (buy_raw.get("realized_pnl_ready_time"), sell_raw.get("realized_pnl_ready_time"))
                    if value not in (None, "")
                ]
                raw = {
                    "reconstruction_source": "sqlite_execution_reducer",
                    "buy_execution_id": lot.get("execution_id"),
                    "sell_execution_id": row.get("execution_id"),
                    "sell_order_id": row.get("order_id"),
                    "sell_perm_id": row.get("perm_id"),
                    "exit_reason": row.get("exit_reason") or sell_raw.get("exit_reason") or "",
                    "exit_reason_source": row.get("exit_reason_source") or sell_raw.get("exit_reason_source") or "",
                    "matched_quantity": matched_qty,
                    "buy_original_quantity": safe_float(lot.get("original_qty")) or matched_qty,
                    "sell_original_quantity": qty,
                    "entry_executed_at": buy_time,
                    "exit_executed_at": sell_time,
                    "execution_insert_time": sell_raw.get("execution_insert_time") or row.get("recorded_at"),
                    "entry_execution_insert_time": buy_raw.get("execution_insert_time") or lot.get("recorded_at"),
                    "exit_execution_insert_time": sell_raw.get("execution_insert_time") or row.get("recorded_at"),
                    "commission_report_time": max(commission_report_times) if commission_report_times else "",
                    "entry_commission_report_time": buy_raw.get("commission_report_time") or "",
                    "exit_commission_report_time": sell_raw.get("commission_report_time") or "",
                    "realized_pnl_ready_time": max(realized_pnl_ready_times) if realized_pnl_ready_times else "",
                    "closed_trade_finalized_time": reducer_started_at if trade_status == "CLOSED" else "",
                    "broker_realized_execution_count": int(sell_realized_ready),
                    "sqlite_realized_execution_count": int(sell_realized_ready),
                    "pending_commission_count": pending_commission_count,
                    "pending_realized_pnl_count": pending_realized_pnl_count,
                    "pnl_status": pnl_status,
                    "commission_status": commission_status,
                    "buy_commission_fallback_source": FALLBACK_BUY_COMMISSION_SOURCE if buy_commission_fallback else "",
                    "buy_commission_unavailable_after_eod": bool(buy_commission_fallback),
                    "broker_flat_at_fallback": bool(broker_flat_for_symbol),
                    "mfe_pct": excursion.get("mfe_pct"),
                    "mae_pct": excursion.get("mae_pct"),
                    "peak_price": excursion.get("peak_price"),
                    "low_price": excursion.get("low_price"),
                    "peak_unrealized_pnl": excursion.get("peak_unrealized_pnl"),
                    "max_adverse_unrealized_pnl": excursion.get("max_adverse_unrealized_pnl"),
                    "giveback_from_peak": excursion.get("giveback_from_peak"),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "buy_commission_confirmed": buy_commission_confirmed,
                    "sell_commission_confirmed": sell_commission_confirmed,
                }
                existing_trade = self.query("SELECT * FROM trades WHERE trade_id = ?", [trade_id])
                existing_raw = parse_jsonish(existing_trade[0].get("raw_json")) if existing_trade else {}
                if existing_raw:
                    existing_raw.update(raw)
                    raw = existing_raw
                preserved: dict[str, Any] = {}
                if existing_trade:
                    current = existing_trade[0]
                    for key in (
                        "mfe_pct",
                        "mae_pct",
                        "peak_price",
                        "low_price",
                        "peak_unrealized_pnl",
                        "max_adverse_unrealized_pnl",
                        "giveback_from_peak",
                        "entry_signal_time",
                        "entry_order_time",
                        "exit_signal_time",
                        "exit_order_time",
                    ):
                        if current.get(key) not in (None, ""):
                            preserved[key] = current.get(key)
                    if not raw.get("exit_reason") and current.get("exit_reason") not in (None, ""):
                        preserved["exit_reason"] = current.get("exit_reason")
                self.upsert_trade(
                    {
                        "trade_id": trade_id,
                        "strategy_name": strategy,
                        "session_date": entry_date,
                        "symbol": symbol,
                        "status": trade_status,
                        "entry_fill_time": buy_time,
                        "exit_fill_time": sell_time,
                        "closed_at": sell_time,
                        "entry_price": buy_price,
                        "exit_price": sell_price,
                        "quantity": matched_qty,
                        "remaining_quantity": 0,
                        "gross_pnl": gross,
                        "commission": commission,
                        "net_pnl": net_pnl,
                        "mfe_pct": excursion.get("mfe_pct"),
                        "mae_pct": excursion.get("mae_pct"),
                        "peak_price": excursion.get("peak_price"),
                        "low_price": excursion.get("low_price"),
                        "peak_unrealized_pnl": excursion.get("peak_unrealized_pnl"),
                        "max_adverse_unrealized_pnl": excursion.get("max_adverse_unrealized_pnl"),
                        "giveback_from_peak": excursion.get("giveback_from_peak"),
                        "exit_reason": raw.get("exit_reason") or "",
                        "ibkr_entry_confirmed": True,
                        "ibkr_exit_confirmed": True,
                        "ibkr_position_flat_confirmed": True,
                        "ibkr_position_flat_confirmed_at": sell_time,
                        "updated_at": reducer_started_at,
                        "raw_json": raw,
                        **preserved,
                    }
                )
                buy_fully_consumed_by_trade = abs(matched_qty - (safe_float(lot.get("original_qty")) or matched_qty)) <= 1e-9
                sell_fully_consumed_by_trade = abs(matched_qty - qty) <= 1e-9
                if buy_fully_consumed_by_trade and sell_fully_consumed_by_trade:
                    self.execute(
                        "UPDATE executions SET trade_id = ? WHERE execution_id IN (?, ?)",
                        (trade_id, lot.get("execution_id"), row.get("execution_id")),
                    )
                closed_count += 1
                lot["remaining_qty"] = lot_remaining - matched_qty
                remaining -= matched_qty
                if (safe_float(lot.get("remaining_qty")) or 0.0) <= 1e-9:
                    open_lots.pop(0)

        today = session_date_utc()
        suppressed_historical_lots: list[dict[str, Any]] = []
        if not allow_historical_open_lots:
            current_open_lots: list[dict[str, Any]] = []
            for lot in open_lots:
                entry_time = execution_sort_time(lot)
                entry_date = str(lot.get("session_date") or "") or iso_date_part(entry_time) or latest_session
                if entry_date and entry_date < today:
                    suppressed_historical_lots.append(lot)
                else:
                    current_open_lots.append(lot)
            open_lots = current_open_lots

        open_qty = sum(safe_float(lot.get("remaining_qty")) or 0.0 for lot in open_lots)
        broker_target_qty: float | None = None
        broker_suppressed_lots: list[dict[str, Any]] = []
        if broker_net_positions is not None:
            broker_target_qty = max(safe_float(broker_net_positions.get(symbol, 0.0)) or 0.0, 0.0)
            if broker_target_qty <= 1e-9:
                broker_suppressed_lots = open_lots
                open_lots = []
                open_qty = 0.0
            elif open_qty > broker_target_qty + 1e-9:
                qty_to_keep = broker_target_qty
                kept_reversed: list[dict[str, Any]] = []
                suppressed_reversed: list[dict[str, Any]] = []
                for lot in reversed(open_lots):
                    lot_qty = safe_float(lot.get("remaining_qty")) or 0.0
                    if lot_qty <= 1e-9:
                        continue
                    if qty_to_keep <= 1e-9:
                        suppressed_reversed.append(lot)
                        continue
                    take_qty = min(lot_qty, qty_to_keep)
                    if take_qty < lot_qty - 1e-9:
                        kept = dict(lot)
                        kept["remaining_qty"] = take_qty
                        kept_reversed.append(kept)
                        suppressed = dict(lot)
                        suppressed["remaining_qty"] = lot_qty - take_qty
                        suppressed_reversed.append(suppressed)
                    else:
                        kept_reversed.append(lot)
                    qty_to_keep -= take_qty
                open_lots = list(reversed(kept_reversed))
                broker_suppressed_lots = list(reversed(suppressed_reversed))
                open_qty = sum(safe_float(lot.get("remaining_qty")) or 0.0 for lot in open_lots)

        suppressed_qty = sum(safe_float(lot.get("remaining_qty")) or 0.0 for lot in suppressed_historical_lots)
        if suppressed_qty > 1e-9:
            stale_cost = sum((safe_float(lot.get("remaining_qty")) or 0.0) * (safe_float(lot.get("price")) or 0.0) for lot in suppressed_historical_lots)
            stale_avg_price = stale_cost / suppressed_qty if suppressed_qty else None
            stale_first = suppressed_historical_lots[0]
            stale_entry_time = execution_sort_time(stale_first)
            stale_entry_date = str(stale_first.get("session_date") or "") or iso_date_part(stale_entry_time) or latest_session
            stale_strategy = str(stale_first.get("strategy_name") or latest_strategy or "unknown")
            self.upsert_position(
                {
                    "position_key": f"{stale_strategy}:{stale_entry_date}:{symbol}:stale_open_lot",
                    "strategy_name": stale_strategy,
                    "session_date": stale_entry_date,
                    "symbol": symbol,
                    "status": "STALE_CARRY_OPEN",
                    "quantity": suppressed_qty,
                    "avg_price": stale_avg_price,
                    "source": "sqlite_execution_reducer",
                    "ibkr_quantity": None,
                    "active": 0,
                    "exit_sent": 0,
                    "updated_at": last_event_time,
                    "raw_json": {
                        "active": False,
                        "entry_fill_verified": True,
                        "entry_time": stale_entry_time,
                        "entry_price": stale_avg_price,
                        "stale_open_lot_suppressed": True,
                        "requires_ibkr_confirmation": True,
                        "open_lot_execution_ids": [lot.get("execution_id") for lot in suppressed_historical_lots],
                    },
                }
            )
        broker_suppressed_qty = sum(safe_float(lot.get("remaining_qty")) or 0.0 for lot in broker_suppressed_lots)
        if broker_suppressed_qty > 1e-9:
            broker_suppressed_cost = sum((safe_float(lot.get("remaining_qty")) or 0.0) * (safe_float(lot.get("price")) or 0.0) for lot in broker_suppressed_lots)
            broker_suppressed_avg = broker_suppressed_cost / broker_suppressed_qty if broker_suppressed_qty else None
            broker_suppressed_first = broker_suppressed_lots[0]
            broker_suppressed_entry_time = execution_sort_time(broker_suppressed_first)
            broker_suppressed_entry_date = str(broker_suppressed_first.get("session_date") or "") or iso_date_part(broker_suppressed_entry_time) or latest_session
            broker_suppressed_strategy = str(broker_suppressed_first.get("strategy_name") or latest_strategy or "unknown")
            self.upsert_position(
                {
                    "position_key": f"{broker_suppressed_strategy}:{broker_suppressed_entry_date}:{symbol}:broker_unconfirmed_open_lot",
                    "strategy_name": broker_suppressed_strategy,
                    "session_date": broker_suppressed_entry_date,
                    "symbol": symbol,
                    "status": "BROKER_UNCONFIRMED_OPEN_LOT",
                    "quantity": broker_suppressed_qty,
                    "avg_price": broker_suppressed_avg,
                    "source": "sqlite_execution_reducer",
                    "ibkr_quantity": broker_target_qty,
                    "active": 0,
                    "exit_sent": 0,
                    "updated_at": last_event_time,
                    "raw_json": {
                        "active": False,
                        "entry_fill_verified": True,
                        "entry_time": broker_suppressed_entry_time,
                        "entry_price": broker_suppressed_avg,
                        "broker_position_reducer_suppressed": True,
                        "broker_target_quantity": broker_target_qty,
                        "suppressed_quantity": broker_suppressed_qty,
                        "open_lot_execution_ids": [lot.get("execution_id") for lot in broker_suppressed_lots],
                    },
                }
            )
        if open_qty > 1e-9:
            weighted_cost = sum((safe_float(lot.get("remaining_qty")) or 0.0) * (safe_float(lot.get("price")) or 0.0) for lot in open_lots)
            avg_price = weighted_cost / open_qty if open_qty else None
            first_lot = open_lots[0]
            entry_time = execution_sort_time(first_lot)
            entry_date = str(first_lot.get("session_date") or "") or iso_date_part(entry_time) or latest_session
            first_lot_raw = parse_jsonish(first_lot.get("raw_json"))
            entry_metadata = self._entry_metadata_for_execution(first_lot, first_lot_raw)
            strategy = str(entry_metadata.get("strategy_name") or first_lot.get("strategy_name") or latest_strategy or "unknown")
            open_excursion = self._trade_excursion_stats(
                entry_price=avg_price,
                exit_price=latest_price,
                quantity=open_qty,
                net_pnl=(latest_price - avg_price) * open_qty if latest_price is not None and avg_price is not None else None,
                raw_sources=[parse_jsonish(lot.get("raw_json")) for lot in open_lots],
            )
            self.mark_position_flat(symbol=symbol, reason="sqlite_execution_reducer_superseded_open_lot", status="CLOSED")
            self.upsert_position(
                {
                    "position_key": f"{strategy}:{entry_date}:{symbol}",
                    "strategy_name": strategy,
                    "session_date": entry_date,
                    "symbol": symbol,
                    "status": "OPEN",
                    "quantity": open_qty,
                    "avg_price": avg_price,
                    "source": "sqlite_execution_reducer",
                    "ibkr_quantity": open_qty,
                    "active": 1,
                    "exit_sent": 0,
                    "top100_rank": safe_int(entry_metadata.get("top100_rank")),
                    "top100_score": safe_float(entry_metadata.get("top100_score")),
                    "top100_source_date": entry_metadata.get("top100_source_date"),
                    "top100_features_json": entry_metadata.get("top100_features_json"),
                    "live_entry_score": safe_float(entry_metadata.get("live_entry_score")),
                    "live_entry_rank": safe_int(entry_metadata.get("live_entry_rank")),
                    "live_entry_features_json": entry_metadata.get("live_entry_features_json"),
                    "signal_source": entry_metadata.get("signal_source"),
                    "signal_time": entry_metadata.get("signal_time"),
                    "ready_since": entry_metadata.get("ready_since"),
                    "entry_order_id": entry_metadata.get("entry_order_id"),
                    "entry_perm_id": entry_metadata.get("entry_perm_id"),
                    "updated_at": last_event_time,
                    "raw_json": self._merge_entry_metadata_into_raw({
                        "active": True,
                        "entry_fill_verified": True,
                        "entry_time": entry_time,
                        "entry_price": avg_price,
                        "market_price": latest_price,
                        "market_price_at": last_event_time,
                        "market_price_source": "execution_reducer",
                        "peak_price": open_excursion.get("peak_price"),
                        "low_price": open_excursion.get("low_price"),
                        "mfe_pct": open_excursion.get("mfe_pct"),
                        "mae_pct": open_excursion.get("mae_pct"),
                        "peak_pct": open_excursion.get("mfe_pct"),
                        "peak_unrealized_pct": open_excursion.get("mfe_pct"),
                        "peak_unrealized_pnl": open_excursion.get("peak_unrealized_pnl"),
                        "max_adverse_unrealized_pnl": open_excursion.get("max_adverse_unrealized_pnl"),
                        "last_update_time": last_event_time,
                        "broker_target_quantity": broker_target_qty,
                        "open_lot_execution_ids": [lot.get("execution_id") for lot in open_lots],
                    }, entry_metadata),
                }
            )
        else:
            self.mark_position_flat(symbol=symbol, strategy_name=strategy_name, reason="sqlite_execution_reducer_flat", status="CLOSED", updated_at=last_event_time)
            if rows:
                self.upsert_position(
                    {
                        "position_key": f"{latest_strategy}:{latest_session}:{symbol}",
                        "strategy_name": latest_strategy,
                        "session_date": latest_session,
                        "symbol": symbol,
                        "status": "CLOSED",
                        "quantity": 0,
                        "avg_price": latest_price,
                        "source": "sqlite_execution_reducer",
                        "ibkr_quantity": 0,
                        "active": 0,
                        "exit_sent": 0,
                        "updated_at": last_event_time,
                        "raw_json": {
                            "active": False,
                            "ibkr_position_flat_confirmed": True,
                            "flat_confirmed_reason": "sqlite_execution_reducer_flat",
                            "flat_confirmed_at": last_event_time,
                            "market_price": latest_price,
                            "market_price_at": last_event_time,
                            "market_price_source": "execution_reducer",
                        },
                    }
                )
        return {
            "symbol": symbol,
            "closed_trades": closed_count,
            "open_quantity": open_qty,
            "suppressed_historical_open_quantity": suppressed_qty,
            "broker_target_quantity": broker_target_qty,
            "broker_suppressed_open_quantity": broker_suppressed_qty,
            "open_lot_suppressed": suppressed_qty > 1e-9 or broker_suppressed_qty > 1e-9,
        }

    def rebuild_positions_from_executions(
        self,
        symbols: list[str] | tuple[str, ...] | None = None,
        *,
        allow_historical_open_lots: bool = False,
        broker_net_positions: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if self._transaction_depth == 0:
            with self.transaction():
                return self.rebuild_positions_from_executions(
                    symbols,
                    allow_historical_open_lots=allow_historical_open_lots,
                    broker_net_positions=broker_net_positions,
                )
        if symbols is None:
            rows = self.query(
                """
                SELECT symbol FROM executions WHERE COALESCE(symbol, '') != ''
                UNION
                SELECT symbol FROM positions WHERE COALESCE(active, 0) = 1 AND COALESCE(symbol, '') != ''
                ORDER BY symbol
                """
            )
            symbols_to_rebuild = [str(row.get("symbol") or "").upper() for row in rows]
        else:
            symbols_to_rebuild = sorted({str(symbol or "").upper().strip() for symbol in symbols if str(symbol or "").strip()})

        results = [
            self.rebuild_symbol_trade_state(
                symbol,
                allow_historical_open_lots=allow_historical_open_lots,
                broker_net_positions=broker_net_positions,
            )
            for symbol in symbols_to_rebuild
        ]
        open_symbols = [row["symbol"] for row in results if (safe_float(row.get("open_quantity")) or 0.0) > 1e-9]
        suppressed_symbols = [row["symbol"] for row in results if row.get("open_lot_suppressed")]
        return {
            "symbols_processed": len(symbols_to_rebuild),
            "open_symbols": open_symbols,
            "open_symbols_count": len(open_symbols),
            "suppressed_historical_open_symbols": suppressed_symbols,
            "suppressed_historical_open_symbols_count": len(suppressed_symbols),
            "broker_constrained": broker_net_positions is not None,
            "closed_trades_rebuilt": sum(safe_int(row.get("closed_trades")) or 0 for row in results),
        }

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
            "position_key": row.get("position_key"),
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
            "exit_reason": row.get("exit_reason"),
            "exit_reason_source": row.get("exit_reason_source"),
            "raw_json": row.get("raw_json") or row,
        }
        self._upsert("orders", data, ["order_key"], list(data.keys()))
        return order_key

    def upsert_trade(self, row: dict[str, Any]) -> str:
        if self._transaction_depth == 0:
            with self.transaction():
                return self.upsert_trade(row)
        trade_id = str(row.get("trade_id") or uuid.uuid4().hex)
        previous = self.query(
            """
            SELECT
                trade_reduction_version,
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
                entry_perm_id
            FROM trades
            WHERE trade_id = ?
            """,
            [trade_id],
        )
        previous_row = previous[0] if previous else {}
        if not previous_row:
            symbol_for_meta = str(row.get("symbol") or "").upper()
            session_for_meta = row.get("session_date") or session_date_utc()
            if symbol_for_meta and session_for_meta:
                entry_rows = self.query(
                    """
                    SELECT
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
                        entry_perm_id
                    FROM trades
                    WHERE UPPER(symbol) = ?
                      AND session_date = ?
                      AND (
                        trade_id LIKE 'entry:%'
                        OR UPPER(COALESCE(status, '')) IN ('ENTRY_PENDING', 'ENTRY_PARTIAL', 'OPEN')
                      )
                    ORDER BY COALESCE(entry_order_time, entry_signal_time, entry_fill_time, updated_at, '') DESC
                    LIMIT 1
                    """,
                    [symbol_for_meta, session_for_meta],
                )
                previous_row = entry_rows[0] if entry_rows else {}
        previous_version = safe_int(previous[0].get("trade_reduction_version")) if previous else None
        raw = parse_jsonish(row.get("raw_json"))
        def entry_meta_value(name: str) -> Any:
            value = row.get(name)
            if value not in (None, ""):
                return value
            value = raw.get(name)
            if value not in (None, ""):
                return value
            return previous_row.get(name) if previous_row else None
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
            "peak_price": safe_float(row.get("peak_price")),
            "low_price": safe_float(row.get("low_price")),
            "peak_unrealized_pnl": safe_float(row.get("peak_unrealized_pnl")),
            "max_adverse_unrealized_pnl": safe_float(row.get("max_adverse_unrealized_pnl")),
            "giveback_from_peak": safe_float(row.get("giveback_from_peak")),
            "exit_reason": row.get("exit_reason") or parse_jsonish(row.get("raw_json")).get("exit_reason"),
            "top100_rank": safe_int(entry_meta_value("top100_rank")),
            "top100_score": safe_float(entry_meta_value("top100_score")),
            "top100_source_date": entry_meta_value("top100_source_date"),
            "top100_features_json": entry_meta_value("top100_features_json"),
            "live_entry_score": safe_float(entry_meta_value("live_entry_score")),
            "live_entry_rank": safe_int(entry_meta_value("live_entry_rank")),
            "live_entry_features_json": entry_meta_value("live_entry_features_json"),
            "signal_source": entry_meta_value("signal_source"),
            "signal_time": entry_meta_value("signal_time"),
            "ready_since": entry_meta_value("ready_since"),
            "entry_order_id": entry_meta_value("entry_order_id"),
            "entry_perm_id": entry_meta_value("entry_perm_id"),
            "ibkr_entry_confirmed": bool_int(row.get("ibkr_entry_confirmed")),
            "ibkr_exit_confirmed": bool_int(row.get("ibkr_exit_confirmed")),
            "ibkr_position_flat_confirmed": bool_int(row.get("ibkr_position_flat_confirmed")),
            "ibkr_position_flat_confirmed_at": row.get("ibkr_position_flat_confirmed_at"),
            "raw_json": row.get("raw_json") or row,
            "updated_at": row.get("updated_at") or utc_now_iso(),
            "trade_reduction_version": safe_int(row.get("trade_reduction_version")) or ((previous_version or 0) + 1),
        }
        status = str(data.get("status") or "").upper()
        if status in {"CLOSED", "DONE", "EXIT_FILLED", "FLAT"}:
            self._delete_duplicate_reconstructed_execution_pair(trade_id, data["symbol"], data.get("raw_json"), data.get("quantity"))
            duplicates = self.query(
                """
                SELECT trade_id, raw_json
                FROM trades
                WHERE trade_id != ?
                  AND UPPER(symbol) = ?
                  AND COALESCE(entry_fill_time, '') = COALESCE(?, '')
                  AND COALESCE(exit_fill_time, '') = COALESCE(?, '')
                  AND ABS(COALESCE(quantity, 0) - COALESCE(?, 0)) < 0.000001
                  AND ABS(COALESCE(entry_price, 0) - COALESCE(?, 0)) < 0.000001
                  AND ABS(COALESCE(exit_price, 0) - COALESCE(?, 0)) < 0.000001
                """,
                [
                    trade_id,
                    data["symbol"],
                    data.get("entry_fill_time"),
                    data.get("exit_fill_time"),
                    data.get("quantity") or 0.0,
                    data.get("entry_price") or 0.0,
                    data.get("exit_price") or 0.0,
                ],
            )
            current_raw = parse_jsonish(data.get("raw_json"))
            current_buy_exec = str(current_raw.get("buy_execution_id") or "")
            current_sell_exec = str(current_raw.get("sell_execution_id") or "")
            for duplicate in duplicates:
                duplicate_id = str(duplicate.get("trade_id") or "")
                duplicate_raw = parse_jsonish(duplicate.get("raw_json"))
                source = str(duplicate_raw.get("reconstruction_source") or "").lower()
                if not duplicate_id.startswith("reconstructed:") and "execution" not in source:
                    continue
                duplicate_buy_exec = str(duplicate_raw.get("buy_execution_id") or "")
                duplicate_sell_exec = str(duplicate_raw.get("sell_execution_id") or "")
                if (
                    current_buy_exec
                    and current_sell_exec
                    and duplicate_buy_exec
                    and duplicate_sell_exec
                    and (duplicate_buy_exec, duplicate_sell_exec) != (current_buy_exec, current_sell_exec)
                ):
                    continue
                self.execute("UPDATE executions SET trade_id = ? WHERE trade_id = ?", (trade_id, duplicate_id))
                self.execute("DELETE FROM trades WHERE trade_id = ?", (duplicate_id,))
        self._upsert("trades", data, ["trade_id"], list(data.keys()))
        return trade_id

    def upsert_position(self, row: dict[str, Any]) -> str:
        strategy = row.get("strategy_name") or "unknown"
        session = row.get("session_date") or session_date_utc()
        symbol = str(row.get("symbol") or "").upper()
        position_key = str(row.get("position_key") or f"{strategy}:{session}:{symbol}")
        previous = self.query(
            """
            SELECT
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
                entry_perm_id
            FROM positions
            WHERE position_key = ?
            """,
            [position_key],
        )
        previous_row = previous[0] if previous else {}
        raw = parse_jsonish(row.get("raw_json"))
        def entry_meta_value(name: str) -> Any:
            value = row.get(name)
            if value not in (None, ""):
                return value
            value = raw.get(name)
            if value not in (None, ""):
                return value
            return previous_row.get(name) if previous_row else None
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
            "top100_rank": safe_int(entry_meta_value("top100_rank")),
            "top100_score": safe_float(entry_meta_value("top100_score")),
            "top100_source_date": entry_meta_value("top100_source_date"),
            "top100_features_json": entry_meta_value("top100_features_json"),
            "live_entry_score": safe_float(entry_meta_value("live_entry_score")),
            "live_entry_rank": safe_int(entry_meta_value("live_entry_rank")),
            "live_entry_features_json": entry_meta_value("live_entry_features_json"),
            "signal_source": entry_meta_value("signal_source"),
            "signal_time": entry_meta_value("signal_time"),
            "ready_since": entry_meta_value("ready_since"),
            "entry_order_id": entry_meta_value("entry_order_id"),
            "entry_perm_id": entry_meta_value("entry_perm_id"),
            "updated_at": row.get("updated_at") or utc_now_iso(),
            "raw_json": row.get("raw_json") or row,
        }
        active_statuses = {"OPEN", "EXIT_ORDER"}
        if data["active"] and str(data.get("status") or "").upper() in active_statuses and symbol:
            incoming_priority = position_source_priority(data.get("source"), data.get("raw_json"), data.get("ibkr_quantity"))
            existing_rows = self.query(
                """
                SELECT position_key, source, ibkr_quantity, updated_at, raw_json
                FROM positions
                WHERE UPPER(symbol) = ?
                  AND position_key != ?
                  AND COALESCE(active, 0) = 1
                  AND UPPER(COALESCE(status, '')) IN ('OPEN', 'EXIT_ORDER')
                """,
                [symbol, position_key],
            )
            stronger_existing = False
            for existing in existing_rows:
                existing_priority = position_source_priority(existing.get("source"), existing.get("raw_json"), existing.get("ibkr_quantity"))
                if existing_priority > incoming_priority:
                    stronger_existing = True
                    break
            if stronger_existing:
                raw = parse_jsonish(data.get("raw_json"))
                live_metric_keys = {
                    "market_price",
                    "market_price_at",
                    "market_price_source",
                    "unrealized_pnl",
                    "unrealized_pct",
                    "peak_pct",
                    "peak_unrealized_pct",
                    "peak_price",
                    "drop_from_peak_pct",
                    "last_update",
                    "data_quality",
                }
                metric_update = {key: raw.get(key) for key in live_metric_keys if raw.get(key) is not None}
                if metric_update:
                    for existing in existing_rows:
                        existing_priority = position_source_priority(existing.get("source"), existing.get("raw_json"), existing.get("ibkr_quantity"))
                        if existing_priority <= incoming_priority:
                            continue
                        existing_raw = parse_jsonish(existing.get("raw_json"))
                        existing_raw.update(metric_update)
                        existing_raw["market_metrics_merged_from"] = data.get("source") or "lower_priority_position"
                        existing_raw["market_metrics_merged_at"] = data["updated_at"]
                        self.execute(
                            """
                            UPDATE positions
                            SET updated_at = ?,
                                raw_json = ?
                            WHERE position_key = ?
                            """,
                            (data["updated_at"], safe_json(existing_raw), existing["position_key"]),
                        )
                raw.update({
                    "active": False,
                    "stale_duplicate_suppressed": True,
                    "duplicate_suppressed_reason": "stronger_active_position_exists",
                    "duplicate_suppressed_at": data["updated_at"],
                })
                data["active"] = 0
                data["status"] = "STALE_DUPLICATE_SUPPRESSED"
                data["raw_json"] = raw
            else:
                for existing in existing_rows:
                    raw = parse_jsonish(existing.get("raw_json"))
                    raw.update({
                        "active": False,
                        "stale_duplicate_suppressed": True,
                        "duplicate_suppressed_reason": "canonical_position_replaced",
                        "duplicate_suppressed_by": position_key,
                        "duplicate_suppressed_at": data["updated_at"],
                    })
                    self.execute(
                        """
                        UPDATE positions
                        SET active = 0,
                            status = 'STALE_DUPLICATE_SUPPRESSED',
                            exit_sent = 0,
                            updated_at = ?,
                            raw_json = ?
                        WHERE position_key = ?
                        """,
                        (data["updated_at"], safe_json(raw), existing["position_key"]),
                    )
        self._upsert("positions", data, ["position_key"], list(data.keys()))
        return position_key

    def mark_position_flat(
        self,
        *,
        symbol: str,
        strategy_name: str | None = None,
        session_date: str | None = None,
        reason: str = "reconciliation_clean",
        status: str = "CLOSED",
        updated_at: str | None = None,
    ) -> int:
        updated_at = updated_at or utc_now_iso()
        clauses = ["UPPER(symbol) = ?", "COALESCE(active, 0) = 1"]
        params: list[Any] = [str(symbol).upper()]
        if strategy_name:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        if session_date:
            clauses.append("session_date = ?")
            params.append(session_date)
        rows = self.query(f"SELECT position_key, raw_json FROM positions WHERE {' AND '.join(clauses)}", params)
        for row in rows:
            raw = parse_jsonish(row.get("raw_json"))
            raw.update({
                "ibkr_position_flat_confirmed": True,
                "flat_confirmed_reason": reason,
                "flat_confirmed_at": updated_at,
            })
            self.execute(
                """
                UPDATE positions
                SET active = 0,
                    status = ?,
                    exit_sent = 0,
                    updated_at = ?,
                    raw_json = ?
                WHERE position_key = ?
                """,
                (status, updated_at, safe_json(raw), row["position_key"]),
            )
        return len(rows)

    def mark_all_positions_flat(
        self,
        *,
        reason: str = "reconciliation_clean",
        strategy_name: str | None = None,
        session_date: str | None = None,
        status: str = "FLAT_CONFIRMED",
        updated_at: str | None = None,
    ) -> int:
        updated_at = updated_at or utc_now_iso()
        clauses = ["COALESCE(active, 0) = 1"]
        params: list[Any] = []
        if strategy_name:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        if session_date:
            clauses.append("session_date = ?")
            params.append(session_date)
        rows = self.query(f"SELECT position_key, raw_json FROM positions WHERE {' AND '.join(clauses)}", params)
        for row in rows:
            raw = parse_jsonish(row.get("raw_json"))
            raw.update({
                "ibkr_position_flat_confirmed": True,
                "flat_confirmed_reason": reason,
                "flat_confirmed_at": updated_at,
            })
            self.execute(
                """
                UPDATE positions
                SET active = 0,
                    status = ?,
                    exit_sent = 0,
                    updated_at = ?,
                    raw_json = ?
                WHERE position_key = ?
                """,
                (status, updated_at, safe_json(raw), row["position_key"]),
            )
        return len(rows)

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


@dataclass
class SQLiteWriteRequest:
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    priority: str
    wait_for_ack: bool
    timeout_seconds: float
    coalesce_key: str = ""
    detail: str = ""
    table: str = ""
    enqueued_at: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    exception: BaseException | None = None


@dataclass(order=True)
class SQLiteWriteQueueItem:
    priority_rank: int
    sequence: int
    request: SQLiteWriteRequest | None = field(compare=False)


class SQLiteWriteQueue:
    """Single-writer proxy for runtime SQLite writes.

    The live trading path should not contend with collector/dashboard/top100
    writes using multiple write connections. This proxy owns one write
    connection in a dedicated thread. Critical ledger calls wait briefly for an
    acknowledgement; diagnostics/market-data calls are best-effort and can be
    dropped when the queue is overloaded.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        maxsize: int = SQLITE_WRITER_QUEUE_MAXSIZE,
        critical_timeout_seconds: float = SQLITE_WRITER_CRITICAL_TIMEOUT_SECONDS,
        best_effort_timeout_seconds: float = SQLITE_WRITER_BEST_EFFORT_TIMEOUT_SECONDS,
    ) -> None:
        self.path = Path(resolve_sqlite_path(path))
        self.maxsize = int(maxsize)
        self.critical_timeout_seconds = float(critical_timeout_seconds)
        self.best_effort_timeout_seconds = float(best_effort_timeout_seconds)
        self._queue: queue.PriorityQueue[SQLiteWriteQueueItem] = queue.PriorityQueue(maxsize=max(1, self.maxsize))
        self._sequence = 0
        self._timeout_log_last: dict[str, float] = {}
        self._coalesced_keys: set[str] = set()
        self._coalesced_lock = threading.Lock()
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._status_lock = threading.Lock()
        self._status: dict[str, Any] = {
            "queue_depth": 0,
            "max_queue_depth": 0,
            "max_queue_size": self.maxsize,
            "dropped_writes": 0,
            "ack_timeouts_total": 0,
            "ack_timeouts_by_method": {},
            "coalesced_writes": 0,
            "coalesced_writes_by_method": {},
            "last_ack_timeout_method": "",
            "last_ack_timeout_at": "",
            "write_count": 0,
            "failed_writes": 0,
            "last_write_latency_ms": None,
            "last_write_method": "",
            "last_write_error": "",
            "last_write_at": "",
            "current_write_method": "",
            "current_write_detail": "",
            "current_write_table": "",
            "current_write_started_at": "",
            "current_write_duration_seconds": 0.0,
            "oldest_queued_age_seconds": 0.0,
            "writer_alive": 0,
        }
        self._thread = threading.Thread(target=self._run, name="sqlite-runtime-writer", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=critical_timeout_seconds):
            raise TimeoutError("SQLite writer queue did not initialize")

    def _set_status(self, **updates: Any) -> None:
        with self._status_lock:
            self._status.update(updates)
            queue_depth = self._queue.qsize()
            self._status["queue_depth"] = queue_depth
            self._status["max_queue_depth"] = max(int(self._status.get("max_queue_depth", 0) or 0), queue_depth)
            self._status["writer_alive"] = int(self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            out = dict(self._status)
        queue_depth = self._queue.qsize()
        out["queue_depth"] = queue_depth
        out["max_queue_depth"] = max(int(out.get("max_queue_depth", 0) or 0), queue_depth)
        out["writer_alive"] = int(self._thread.is_alive())
        current_started = out.get("current_write_monotonic")
        if isinstance(current_started, (int, float)) and current_started > 0:
            out["current_write_duration_seconds"] = round(time.monotonic() - float(current_started), 3)
        out.pop("current_write_monotonic", None)
        out["oldest_queued_age_seconds"] = self._oldest_queued_age_seconds()
        return out

    def _oldest_queued_age_seconds(self) -> float:
        try:
            with self._queue.mutex:
                enqueued = [
                    item.request.enqueued_at
                    for item in list(self._queue.queue)
                    if item.request is not None
                ]
            if not enqueued:
                return 0.0
            return round(max(0.0, time.monotonic() - min(enqueued)), 3)
        except Exception:
            return 0.0

    def _priority_rank(self, priority: str) -> int:
        if priority == "critical":
            return 0
        if priority == "normal":
            return 5
        return 10

    def _run(self) -> None:
        store: SQLiteRuntimeStore | None = None
        try:
            store = SQLiteRuntimeStore(self.path)
            self._ready.set()
            while True:
                item = self._queue.get()
                request = item.request
                if request is None:
                    self._queue.task_done()
                    break
                started = time.monotonic()
                self._set_status(
                    current_write_method=request.method,
                    current_write_detail=request.detail,
                    current_write_table=request.table,
                    current_write_started_at=utc_now_iso(),
                    current_write_monotonic=started,
                    current_write_duration_seconds=0.0,
                    oldest_queued_age_seconds=self._oldest_queued_age_seconds(),
                )
                try:
                    request.result = safe_sqlite_call(store, request.method, *request.args, **request.kwargs)
                except BaseException as exc:
                    request.exception = exc
                finally:
                    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
                    if (latency_ms / 1000.0) >= SQLITE_WRITER_SLOW_WRITE_SECONDS:
                        log_sqlite_line(
                            f"{utc_now_iso()} SQLITE_SLOW_WRITE method={request.method} "
                            f"duration_seconds={round(latency_ms / 1000.0, 3)} "
                            f"table={str(request.table or '').replace(' ', '_')} "
                            f"detail={str(request.detail or '').replace(' ', '_')} "
                            f"queue_depth={self._queue.qsize()} "
                            f"oldest_queued_age_seconds={self._oldest_queued_age_seconds()}"
                        )
                    with self._status_lock:
                        if request.exception is None:
                            self._status["write_count"] = int(self._status.get("write_count", 0) or 0) + 1
                            self._status["last_write_error"] = ""
                        else:
                            self._status["failed_writes"] = int(self._status.get("failed_writes", 0) or 0) + 1
                        self._status["last_write_latency_ms"] = latency_ms
                        self._status["last_write_method"] = request.method
                        self._status["last_write_at"] = utc_now_iso()
                        self._status["current_write_method"] = ""
                        self._status["current_write_detail"] = ""
                        self._status["current_write_table"] = ""
                        self._status["current_write_started_at"] = ""
                        self._status["current_write_monotonic"] = 0.0
                        self._status["current_write_duration_seconds"] = 0.0
                        self._status["oldest_queued_age_seconds"] = self._oldest_queued_age_seconds()
                        queue_depth = self._queue.qsize()
                        self._status["queue_depth"] = queue_depth
                        self._status["max_queue_depth"] = max(int(self._status.get("max_queue_depth", 0) or 0), queue_depth)
                        self._status["writer_alive"] = int(self._thread.is_alive())
                    self._release_coalesced_key(request.coalesce_key)
                    request.done.set()
                    self._queue.task_done()
        except BaseException as exc:
            self._set_status(last_write_error=repr(exc), failed_writes=int(self._status.get("failed_writes", 0) or 0) + 1, writer_alive=0)
            self._ready.set()
            log_sqlite_line(f"{utc_now_iso()} SQLITE_WRITER_CRASH error={exc!r}")
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception:
                    pass
            self._closed.set()

    def _priority_for(self, method: str, priority: str | None = None) -> str:
        if priority:
            return priority
        if method in CRITICAL_SQLITE_WRITE_METHODS:
            return "critical"
        if method in BEST_EFFORT_SQLITE_WRITE_METHODS:
            return "best_effort"
        return "critical"

    def _request_metadata(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, str, str]:
        if method == "mark_operation_status":
            name = str(args[0] if args else kwargs.get("name") or "").strip()
            status = str(args[1] if len(args) > 1 else kwargs.get("status") or "").strip()
            key = f"{method}:{name}:{status}"
            return key, f"operation={name} status={status}", "runtime_state"
        if method == "runtime_pending_counts":
            session_date = str(args[0] if args else kwargs.get("session_date") or "").strip()
            key = f"{method}:{session_date or 'all'}"
            return key, f"session_date={session_date or 'all'} sql=pending_counts_aggregate", "executions,trades"
        if method == "finalize_pending_trades":
            session_date = str(args[0] if args else kwargs.get("session_date") or "").strip()
            if not session_date and os.environ.get("TRADING_BOT_ALLOW_FULL_PENDING_TRADE_REBUILD") != "1":
                effective = datetime.now(timezone.utc).date().isoformat()
                key = f"{method}:{effective}"
                return (
                    key,
                    f"session_date={effective} sql=pending_trade_rebuild full_rebuild=0 default_scoped=1",
                    "trades,executions,positions",
                )
            key = f"{method}:{session_date or 'all'}"
            return key, f"session_date={session_date or 'all'} sql=pending_trade_rebuild full_rebuild={0 if session_date else 1}", "trades,executions,positions"
        if method == "rebuild_positions_from_executions":
            symbols = args[0] if args else kwargs.get("symbols")
            symbol_count = len(symbols) if isinstance(symbols, (list, tuple, set)) else "all"
            return "", f"symbols={symbol_count} sql=reducer_rebuild", "executions,positions,trades"
        return "", f"method={method}", ""

    def _release_coalesced_key(self, key: str) -> None:
        if not key:
            return
        with self._coalesced_lock:
            self._coalesced_keys.discard(key)

    def _coalesce_duplicate(self, method: str, key: str) -> bool:
        if method not in COALESCED_SQLITE_WRITE_METHODS or not key:
            return False
        with self._coalesced_lock:
            if key in self._coalesced_keys:
                with self._status_lock:
                    self._status["coalesced_writes"] = int(self._status.get("coalesced_writes", 0) or 0) + 1
                    by_method = dict(self._status.get("coalesced_writes_by_method") or {})
                    by_method[method] = int(by_method.get(method, 0) or 0) + 1
                    self._status["coalesced_writes_by_method"] = by_method
                    queue_depth = self._queue.qsize()
                    self._status["queue_depth"] = queue_depth
                    self._status["max_queue_depth"] = max(int(self._status.get("max_queue_depth", 0) or 0), queue_depth)
                return True
            self._coalesced_keys.add(key)
            return False

    def call(
        self,
        method: str,
        *args: Any,
        priority: str | None = None,
        wait_for_ack: bool | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._closed.is_set():
            log_sqlite_line(f"{utc_now_iso()} SQLITE_WRITE_FAILED method={method} error=RuntimeError('writer queue closed')")
            return None
        effective_priority = self._priority_for(method, priority)
        if wait_for_ack is None:
            wait_for_ack = effective_priority == "critical"
        if timeout_seconds is None:
            timeout_seconds = (
                SQLITE_WRITER_METHOD_ACK_TIMEOUT_SECONDS.get(method, self.critical_timeout_seconds)
                if wait_for_ack
                else self.best_effort_timeout_seconds
            )
        coalesce_key, detail, table = self._request_metadata(method, tuple(args), dict(kwargs))
        if self._coalesce_duplicate(method, coalesce_key):
            return None
        request = SQLiteWriteRequest(
            method=method,
            args=tuple(args),
            kwargs=dict(kwargs),
            priority=effective_priority,
            wait_for_ack=bool(wait_for_ack),
            timeout_seconds=float(timeout_seconds),
            coalesce_key=coalesce_key,
            detail=detail,
            table=table,
        )
        try:
            self._sequence += 1
            item = SQLiteWriteQueueItem(self._priority_rank(effective_priority), self._sequence, request)
            self._queue.put(item, timeout=max(0.0, float(timeout_seconds)))
        except queue.Full:
            self._release_coalesced_key(coalesce_key)
            with self._status_lock:
                self._status["dropped_writes"] = int(self._status.get("dropped_writes", 0) or 0) + 1
                queue_depth = self._queue.qsize()
                self._status["queue_depth"] = queue_depth
                self._status["max_queue_depth"] = max(int(self._status.get("max_queue_depth", 0) or 0), queue_depth)
                self._status["last_write_error"] = f"queue_full:{method}"
            log_sqlite_line(f"{utc_now_iso()} SQLITE_WRITE_DROPPED method={method} priority={effective_priority} reason=queue_full")
            return None
        self._set_status(queue_depth=self._queue.qsize())
        if not wait_for_ack:
            return "queued"
        if not request.done.wait(timeout=max(0.0, float(timeout_seconds))):
            with self._status_lock:
                total = int(self._status.get("ack_timeouts_total", 0) or 0) + 1
                by_method = dict(self._status.get("ack_timeouts_by_method") or {})
                by_method[method] = int(by_method.get(method, 0) or 0) + 1
                self._status["ack_timeouts_total"] = total
                self._status["ack_timeouts_by_method"] = by_method
                self._status["last_ack_timeout_method"] = method
                self._status["last_ack_timeout_at"] = utc_now_iso()
            now = time.monotonic()
            last = float(self._timeout_log_last.get(method) or 0.0)
            if now - last >= SQLITE_WRITER_TIMEOUT_LOG_INTERVAL_SECONDS:
                status = self.status()
                queue_depth = int(status.get("queue_depth") or 0)
                dropped_writes = int(status.get("dropped_writes") or 0)
                current_duration = float(status.get("current_write_duration_seconds") or 0.0)
                should_log = queue_depth > 0 or dropped_writes > 0
                if should_log:
                    self._timeout_log_last[method] = now
                    log_sqlite_line(
                        f"{utc_now_iso()} SQLITE_WRITE_ACK_TIMEOUT method={method} "
                        f"priority={effective_priority} timeout_seconds={timeout_seconds} "
                        f"queue_depth={queue_depth} max_queue_depth={status.get('max_queue_depth') or 0} "
                        f"oldest_queued_age_seconds={status.get('oldest_queued_age_seconds')} "
                        f"current_write_method={status.get('current_write_method') or ''} "
                        f"current_write_duration_seconds={current_duration} "
                        f"current_write_table={str(status.get('current_write_table') or '').replace(' ', '_')} "
                        f"current_write_detail={str(status.get('current_write_detail') or '').replace(' ', '_')} "
                        f"dropped_writes={dropped_writes} "
                        f"coalesced_writes={status.get('coalesced_writes') or 0} "
                        f"ack_timeouts_total={status.get('ack_timeouts_total') or 0} "
                        f"ack_timeouts_by_method={json.dumps(status.get('ack_timeouts_by_method') or {}, sort_keys=True, separators=(',', ':'))} "
                        f"last_write_method={status.get('last_write_method') or ''} "
                        f"last_write_latency_ms={status.get('last_write_latency_ms') or ''} "
                        f"last_error={str(status.get('last_write_error') or '').replace(' ', '_')}"
                    )
            return None
        if request.exception is not None:
            raise request.exception
        return request.result

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)

        def _call(*args: Any, **kwargs: Any) -> Any:
            return self.call(method, *args, **kwargs)

        return _call

    def query(self, sql: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]:
        with connect_sqlite(self.path, read_only=True) as conn:
            cur = conn.execute(sql, tuple(params or ()))
            return [dict(row) for row in cur.fetchall()]

    def execute(self, sql: str, params: Iterable[Any] | None = None) -> sqlite3.Cursor:
        return self.call("execute", sql, tuple(params or ()), priority="critical", wait_for_ack=True)

    def close(self, *, drain: bool = True, timeout_seconds: float = SQLITE_WRITER_JOIN_TIMEOUT_SECONDS) -> None:
        if drain:
            deadline = time.monotonic() + max(0.0, timeout_seconds)
            while self._queue.unfinished_tasks > 0 and time.monotonic() < deadline:
                time.sleep(0.01)
        try:
            self._sequence += 1
            self._queue.put_nowait(SQLiteWriteQueueItem(99, self._sequence, None))
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.0, timeout_seconds))
        self._closed.set()


def safe_sqlite_call(
    store: SQLiteRuntimeStore | SQLiteWriteQueue | None,
    method: str,
    *args: Any,
    priority: str | None = None,
    wait_for_ack: bool | None = None,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> Any:
    if store is None:
        return None
    if isinstance(store, SQLiteWriteQueue):
        try:
            return store.call(
                method,
                *args,
                priority=priority,
                wait_for_ack=wait_for_ack,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )
        except (KeyboardInterrupt, SystemExit):
            log_sqlite_line(f"{utc_now_iso()} SQLITE_CALL_INTERRUPTED method={method}")
            raise
        except Exception as exc:
            log_sqlite_line(f"{utc_now_iso()} SQLITE_WRITE_FAILED method={method} error={exc!r}")
            return None
    attempts = max(1, SQLITE_LOCK_RETRY_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            return getattr(store, method)(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            log_sqlite_line(f"{utc_now_iso()} SQLITE_CALL_INTERRUPTED method={method}")
            raise
        except Exception as exc:
            if is_sqlite_busy_error(exc) and attempt < attempts:
                delay = min(SQLITE_LOCK_RETRY_MAX_SECONDS, SQLITE_LOCK_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
                log_sqlite_line(
                    f"{utc_now_iso()} SQLITE_BUSY_RETRY method={method} attempt={attempt}/{attempts} "
                    f"delay_seconds={delay:.3f} error={exc!r}"
                )
                time.sleep(delay)
                continue
            log_sqlite_line(f"{utc_now_iso()} SQLITE_WRITE_FAILED method={method} error={exc!r}")
            return None
    return None
