from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.live_trading.unified_logger import append_unified_log, unified_logger_installed


DEFAULT_SQLITE_PATH = "data/runtime/trading_runtime.sqlite"
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


class SQLiteRuntimeStore:
    def __init__(self, path: str | Path | None = None, *, init: bool = True) -> None:
        self.path = Path(resolve_sqlite_path(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._broker_net_positions: dict[str, float] | None = None
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
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
        symbols = [str(row.get("symbol") or "").upper() for row in rows if row.get("symbol")]
        if not symbols:
            return {
                "broker_constrained": True,
                "symbols_processed": 0,
                "open_symbols_count": 0,
                "suppressed_historical_open_symbols_count": 0,
            }
        return self.rebuild_positions_from_executions(symbols, broker_net_positions=self._broker_net_positions)

    @contextmanager
    def transaction(self):
        outermost = self._transaction_depth == 0
        if outermost:
            self.conn.execute("BEGIN IMMEDIATE")
        self._transaction_depth += 1
        try:
            yield
        except Exception:
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
                exit_reason TEXT,
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
        self._ensure_column("trades", "updated_at", "TEXT")
        self._ensure_column("trades", "trade_reduction_version", "INTEGER DEFAULT 1")
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
        raw = parse_jsonish(data.get("raw_json"))
        if not raw.get("execution_insert_time") and data.get("recorded_at"):
            raw["execution_insert_time"] = data.get("recorded_at")
        if data.get("commission_source") == "ibkr" and not raw.get("commission_report_time"):
            raw["commission_report_time"] = row.get("commission_report_time") or data.get("recorded_at")
        if data.get("realized_pnl") is not None and not raw.get("realized_pnl_ready_time"):
            raw["realized_pnl_ready_time"] = row.get("realized_pnl_ready_time") or raw.get("commission_report_time") or data.get("recorded_at")
        data["raw_json"] = raw
        columns = list(data.keys())
        with self.transaction():
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

    def runtime_pending_counts(self, session_date: str | None = None) -> dict[str, Any]:
        where = ""
        params: list[Any] = []
        if session_date:
            where = "WHERE COALESCE(substr(executed_at, 1, 10), substr(recorded_at, 1, 10), session_date) = ?"
            params.append(session_date)
        execution_rows = self.query(
            f"""
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
            "pending_execution_count": int(exec_row.get("execution_count") or 0),
            "pending_commission_count": int(exec_row.get("pending_commission_count") or 0),
            "pending_realized_pnl_count": int(exec_row.get("pending_realized_pnl_count") or 0),
            "pending_trade_finalization_count": int(trade_row.get("pending_trade_finalization_count") or 0),
        }

    def _reconcile_trade_state_after_execution(self, execution: dict[str, Any]) -> None:
        try:
            symbol = str(execution.get("symbol") or "").upper().strip()
            if not symbol:
                return
            self.rebuild_symbol_trade_state(
                symbol,
                allow_historical_open_lots=False,
                broker_net_positions=self._broker_net_positions,
            )
        except Exception as exc:
            line = f"{utc_now_iso()} SQLITE_WRITE_FAILED method=rebuild_symbol_trade_state error={exc!r}"
            print(line, flush=True)
            if not unified_logger_installed():
                append_unified_log(line)

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
                   realized_pnl, commission_source, raw_json
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
                buy_commission_confirmed = str(lot.get("commission_source") or "").lower() == "ibkr"
                sell_commission_confirmed = str(row.get("commission_source") or "").lower() == "ibkr"
                buy_commission_missing = str(lot.get("commission_source") or "").lower() == "missing"
                sell_commission_missing = str(row.get("commission_source") or "").lower() == "missing"
                commission = execution_commission(lot, buy_fraction) + execution_commission(row, sell_fraction)
                gross = (sell_price - buy_price) * matched_qty
                trade_id = self._closed_trade_id_for_pair(lot, row, matched_qty)
                strategy = str(lot.get("strategy_name") or row.get("strategy_name") or latest_strategy or "unknown")
                buy_raw = parse_jsonish(lot.get("raw_json"))
                sell_raw = parse_jsonish(row.get("raw_json"))
                pending_commission_count = int(buy_commission_missing) + int(sell_commission_missing)
                sell_realized_expected = sell_commission_confirmed or sell_commission_missing
                sell_realized_ready = safe_float(row.get("realized_pnl")) is not None
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
                    for key in ("mfe_pct", "mae_pct", "exit_reason", "entry_signal_time", "entry_order_time", "exit_signal_time", "exit_order_time"):
                        if current.get(key) not in (None, ""):
                            preserved[key] = current.get(key)
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
                        "net_pnl": gross - commission,
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
            strategy = str(first_lot.get("strategy_name") or latest_strategy or "unknown")
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
                    "updated_at": last_event_time,
                    "raw_json": {
                        "active": True,
                        "entry_fill_verified": True,
                        "entry_time": entry_time,
                        "entry_price": avg_price,
                        "market_price": latest_price,
                        "market_price_at": last_event_time,
                        "market_price_source": "execution_reducer",
                        "broker_target_quantity": broker_target_qty,
                        "open_lot_execution_ids": [lot.get("execution_id") for lot in open_lots],
                    },
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
        if self._transaction_depth == 0:
            with self.transaction():
                return self.upsert_trade(row)
        trade_id = str(row.get("trade_id") or uuid.uuid4().hex)
        previous = self.query("SELECT trade_reduction_version FROM trades WHERE trade_id = ?", [trade_id])
        previous_version = safe_int(previous[0].get("trade_reduction_version")) if previous else None
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
