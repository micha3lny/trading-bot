from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard.runtime_queries import DateWindow, list_sessions, list_strategies, load_dashboard_snapshot, load_diagnostics, utc_today
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


class RuntimeDashboardQueriesTests(unittest.TestCase):
    def insert_execution_direct(self, store: SQLiteRuntimeStore, row: dict) -> None:
        payload = dict(row)
        store.conn.execute(
            """
            INSERT INTO executions (
                execution_id, trade_id, strategy_name, session_date, symbol, side, quantity, price,
                executed_at, recorded_at, commission, commission_currency, commission_source, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("execution_id"),
                payload.get("trade_id"),
                payload.get("strategy_name"),
                payload.get("session_date"),
                payload.get("symbol"),
                payload.get("side"),
                payload.get("quantity"),
                payload.get("price") or payload.get("fill_price"),
                payload.get("executed_at"),
                payload.get("recorded_at"),
                payload.get("commission"),
                payload.get("commission_currency"),
                payload.get("commission_source"),
                json.dumps(payload.get("raw_json") or payload),
            ),
        )
        store.conn.commit()

    def test_runtime_open_positions_use_sqlite_positions_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_date = utc_today()
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_position({
                "session_date": session_date,
                "strategy_name": "v67",
                "symbol": "OPEN1",
                "quantity": 5,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "source": "live_buy",
                "exit_sent": 1,
                "updated_at": f"{session_date}T14:01:00+00:00",
                "raw_json": {
                    "entry_time": f"{session_date}T13:31:00+00:00",
                    "market_price": 10.5,
                    "market_price_at": f"{session_date}T14:00:00+00:00",
                    "peak_pct": 8.0,
                    "data_quality": "OK",
                    "open_lot_execution_ids": ["B1", "B2"],
                },
            })
            store.upsert_position({
                "session_date": session_date,
                "strategy_name": "v67",
                "symbol": "OLDORPHAN",
                "quantity": 1,
                "avg_price": 10,
                "active": 1,
                "status": "ORPHAN_STALE_POSITION",
                "updated_at": f"{session_date}T14:01:00+00:00",
                "raw_json": {"market_price": 11},
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow(session_date, session_date), "v67")
            open_positions = snapshot["open_positions"]

            self.assertEqual(open_positions["symbol"].tolist(), ["OPEN1"])
            row = open_positions.iloc[0]
            self.assertAlmostEqual(row["now"], 10.5)
            self.assertAlmostEqual(row["now_dollars"], 2.5)
            self.assertAlmostEqual(row["now_pct"], 5.0)
            self.assertAlmostEqual(row["peak_pct"], 8.0)
            self.assertAlmostEqual(row["giveback_pct"], 3.0)
            self.assertEqual(row["ibkr_confirmed"], "UNKNOWN")
            self.assertEqual(row["source"], "live_buy")
            self.assertEqual(row["exit_sent"], 1)
            self.assertEqual(row["execution_ids"], "B1, B2")

    def test_runtime_executions_use_sqlite_and_sort_descending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_date = utc_today()
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "E1",
                "session_date": session_date,
                "strategy_name": "v67",
                "symbol": "AAA",
                "side": "BOT",
                "quantity": 2,
                "price": 10,
                "executed_at": f"{session_date}T13:30:00+00:00",
                "recorded_at": f"{session_date}T13:30:01+00:00",
            })
            store.upsert_execution({
                "execution_id": "E2",
                "session_date": session_date,
                "strategy_name": "v67",
                "symbol": "BBB",
                "side": "SLD",
                "quantity": 3,
                "price": 11,
                "executed_at": f"{session_date}T13:31:00+00:00",
                "recorded_at": f"{session_date}T13:31:01+00:00",
                "commission": 0.5,
                "commission_source": "ibkr",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow(session_date, session_date), "v67")
            executions = snapshot["executions"]

            self.assertEqual(executions["execution_id"].tolist()[:2], ["E2", "E1"])
            latest = executions.iloc[0]
            self.assertEqual(latest["side"], "SLD")
            self.assertEqual(latest["qty"], 3)
            self.assertEqual(latest["price"], 11)
            self.assertEqual(latest["gross_value"], 33)
            self.assertIn(latest["data_quality"], {"OK", "PNL_PENDING"})
            pending = executions[executions["execution_id"] == "E1"].iloc[0]
            self.assertEqual(pending["data_quality"], "COMMISSION_PENDING")

    def test_snapshot_reconstructs_closed_open_summary_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_date = utc_today()
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B1",
                "session_date": session_date,
                "strategy_name": "v67",
                "symbol": "AAA",
                "side": "BOT",
                "quantity": 10,
                "price": 10,
                "commission": 0.25,
                "commission_source": "ibkr",
                "recorded_at": f"{session_date}T13:30:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S1",
                "session_date": session_date,
                "strategy_name": "v67",
                "symbol": "AAA",
                "side": "SLD",
                "quantity": 10,
                "price": 10.5,
                "commission": 0.25,
                "commission_source": "ibkr",
                "recorded_at": f"{session_date}T13:45:00+00:00",
            })
            store.upsert_position({
                "session_date": session_date,
                "strategy_name": "v67",
                "symbol": "BBB",
                "quantity": 5,
                "avg_price": 20,
                "active": 1,
                "status": "OPEN",
                "updated_at": f"{session_date}T14:00:00+00:00",
                "raw_json": {"market_price": 21, "peak_price": 22, "entry_time": f"{session_date}T13:35:00+00:00"},
            })
            store.record_runtime_event(
                session_date=session_date,
                strategy_name="v67",
                event_type="DELAYED_FILL_AFTER_CANCEL",
                symbol="AAA",
            )
            store.record_risk_event(
                session_date=session_date,
                strategy_name="v67",
                event_type="RISK_GUARD_BLOCK_ENTRY",
                symbol="CCC",
                blocked=1,
                reason="max_daily_loss",
            )
            store.close()

            default_snapshot = load_dashboard_snapshot(db, DateWindow(session_date, session_date), "v67")

            self.assertEqual(default_snapshot["summary"]["closed_trades"], 1)
            self.assertFalse(default_snapshot["closed_positions"].empty)
            self.assertEqual(default_snapshot["diagnostics"]["persisted_closed_trades_count"], 1)
            self.assertEqual(default_snapshot["diagnostics"]["reconstructed_execution_pairs_count"], 0)
            self.assertEqual(default_snapshot["diagnostics"]["displayed_closed_trades_count"], 1)
            self.assertEqual(default_snapshot["diagnostics"]["execution_reconstruction_disabled"], 0)

            snapshot = load_dashboard_snapshot(db, DateWindow(session_date, session_date), "v67", include_reconstructed=True)

            self.assertEqual(snapshot["summary"]["closed_trades"], 1)
            self.assertEqual(snapshot["summary"]["open_trades"], 1)
            self.assertAlmostEqual(snapshot["summary"]["gross_pnl"], 5.0)
            self.assertAlmostEqual(snapshot["summary"]["net_actual_pnl"], 4.5)
            self.assertEqual(snapshot["closed_positions"].iloc[0]["symbol"], "AAA")
            self.assertEqual(snapshot["open_positions"].iloc[0]["symbol"], "BBB")
            self.assertEqual(snapshot["diagnostics"]["delayed_fills"], 1)
            self.assertEqual(snapshot["diagnostics"]["risk_guard_blocks"], 1)

    def test_execution_pair_without_trade_row_reconstructs_closed_trade_with_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B_RECON",
                "session_date": "2026-05-27",
                "strategy_name": "unknown",
                "symbol": "MRAM",
                "side": "BOT",
                "quantity": 2,
                "price": 10,
                "commission": 0.2,
                "commission_source": "ibkr",
                "executed_at": "2026-05-27T13:31:00+00:00",
                "recorded_at": "2026-05-27T13:32:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S_RECON",
                "session_date": "2026-05-27",
                "strategy_name": "unknown",
                "symbol": "MRAM",
                "side": "SLD",
                "quantity": 2,
                "price": 11,
                "commission": 0.2,
                "commission_source": "ibkr",
                "executed_at": "2026-05-27T13:41:00+00:00",
                "recorded_at": "2026-05-27T13:42:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "All", include_reconstructed=True)
            closed = snapshot["closed_positions"].iloc[0]

            self.assertEqual(closed["symbol"], "MRAM")
            self.assertAlmostEqual(closed["gross"], 2.0)
            self.assertEqual(closed["entry_time"], "2026-05-27T13:31:00+00:00")
            self.assertEqual(closed["exit_time"], "2026-05-27T13:41:00+00:00")
            self.assertEqual(closed["commission_status"], "OK")
            self.assertEqual(closed["closed_source"], "trades")

    def test_execution_pair_without_executed_at_uses_recorded_at_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B_MISSING_TIME",
                "session_date": "2026-05-27",
                "symbol": "GRRR",
                "side": "BOT",
                "quantity": 1,
                "price": 10,
                "recorded_at": "2026-05-27T13:32:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S_MISSING_TIME",
                "session_date": "2026-05-27",
                "symbol": "GRRR",
                "side": "SLD",
                "quantity": 1,
                "price": 11,
                "recorded_at": "2026-05-27T13:42:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "All", include_reconstructed=True)
            closed = snapshot["closed_positions"].iloc[0]

            self.assertEqual(closed["symbol"], "GRRR")
            self.assertEqual(closed["entry_time"], "2026-05-27T13:32:00+00:00")
            self.assertEqual(closed["exit_time"], "2026-05-27T13:42:00+00:00")
            self.assertAlmostEqual(closed["hold_minutes"], 10.0)
            self.assertEqual(closed["commission_status"], "MISSING")
            self.assertNotIn("MISSING_EXECUTION_TIME", closed["data_quality"])
            self.assertEqual(closed["closed_source"], "trades")
            self.assertIn("COMMISSION_MISSING", closed["data_quality"])

    def test_execution_pair_raw_json_execution_time_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B_RAW_TIME",
                "session_date": "2026-05-27",
                "symbol": "RAWTS",
                "side": "BOT",
                "quantity": 1,
                "price": 10,
                "raw_json": {"execution": {"time": "2026-05-27T13:31:00+00:00"}},
            })
            store.upsert_execution({
                "execution_id": "S_RAW_TIME",
                "session_date": "2026-05-27",
                "symbol": "RAWTS",
                "side": "SLD",
                "quantity": 1,
                "price": 11,
                "raw_json": {"execution": {"time": "2026-05-27T13:41:00+00:00"}},
            })
            rows = store.query("SELECT execution_id, executed_at FROM executions ORDER BY execution_id")
            store.close()

            self.assertEqual(rows[0]["executed_at"], "2026-05-27T13:31:00+00:00")
            self.assertEqual(rows[1]["executed_at"], "2026-05-27T13:41:00+00:00")

    def test_reconstructed_trade_peak_matches_runtime_symbol_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "reconstructed:2026-05-27:AKTX:B:S",
                "session_date": "2026-05-27",
                "strategy_name": "unknown",
                "symbol": "AKTX",
                "status": "CLOSED",
                "entry_price": 17.0,
                "exit_price": 17.3778,
                "quantity": 1,
                "gross_pnl": 0.3778,
                "raw_json": {"reconstruction_source": "executions_pair"},
            })
            store.record_runtime_event(
                session_date="2026-05-27",
                strategy_name="unknown",
                event_type="PEAK_UPDATED",
                symbol="AKTX",
                raw_json={"entry_price": 17.0, "peak_price": 17.97},
            )
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "All")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["peak_pct"], 5.7059, places=3)
            self.assertAlmostEqual(closed["drop_from_peak_pct"], -3.2955, places=3)
            self.assertEqual(closed["peak_source"], "runtime_events_symbol_session")
            self.assertEqual(closed["peak_match_quality"], "symbol_session_unique")

    def test_runtime_peak_event_with_null_session_date_matches_event_time_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "reconstructed:2026-05-27:AKTX:B:S",
                "session_date": "2026-05-27",
                "strategy_name": "unknown",
                "symbol": "AKTX",
                "status": "CLOSED",
                "entry_price": 17.0,
                "exit_price": 17.3778,
                "quantity": 1,
                "gross_pnl": 0.3778,
                "raw_json": {"reconstruction_source": "executions_pair"},
            })
            store.execute(
                """
                INSERT INTO runtime_events (
                    event_time, severity, event_type, strategy_name, session_date, symbol, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-05-27T13:39:00+00:00",
                    "INFO",
                    "PEAK_UPDATED",
                    "unknown",
                    None,
                    "AKTX",
                    '{"entry_price": 17.0, "peak_price": 17.97}',
                ),
            )
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "All")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["peak_pct"], 5.7059, places=3)
            self.assertAlmostEqual(closed["drop_from_peak_pct"], -3.2955, places=3)
            self.assertEqual(closed["peak_source"], "runtime_events_symbol_session")
            self.assertEqual(closed["peak_match_quality"], "symbol_session_unique")

    def test_closed_trade_commission_uses_confirmed_executions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T1",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "AAA",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T13:30:00+00:00",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 10,
                "gross_pnl": 10,
                "commission": 99,
                "net_pnl": -89,
                "mfe_pct": 12,
            })
            self.insert_execution_direct(store, {
                "execution_id": "B1",
                "trade_id": "T1",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "AAA",
                "side": "BOT",
                "quantity": 10,
                "price": 10,
                "commission": 0.35,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:30:00+00:00",
            })
            self.insert_execution_direct(store, {
                "execution_id": "S1",
                "trade_id": "T1",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "AAA",
                "side": "SLD",
                "quantity": 10,
                "price": 11,
                "commission": 0.40,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:40:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["ibkr_commission"], 99.0)
            self.assertAlmostEqual(closed["net_actual"], -89.0)
            self.assertAlmostEqual(closed["net_pct"], -89.0)
            self.assertEqual(closed["commission_status"], "OK")
            self.assertEqual(closed["data_quality"], "OK")
            self.assertEqual(closed["entry_execution_count"], 0)
            self.assertEqual(closed["exit_execution_count"], 0)
            self.assertEqual(closed["confirmed_commission_execution_count"], 2)
            self.assertEqual(closed["expected_commission_execution_count"], 0)
            self.assertEqual(closed["peak_source"], "trades.mfe_pct")
            self.assertEqual(snapshot["data_quality_summary"]["commission_ok"], 1)
            self.assertEqual(snapshot["data_quality_summary"]["peak_ok"], 1)

    def test_closed_trade_commission_matches_symbol_time_without_trade_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_NO_EXEC_ID",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "MATCH",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T13:30:00+00:00",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 1,
                "gross_pnl": 1,
                "mfe_pct": 10,
            })
            store.upsert_execution({
                "execution_id": "B_SYMBOL_TIME",
                "session_date": "2026-05-27",
                "symbol": "MATCH",
                "side": "BOT",
                "quantity": 1,
                "price": 10,
                "commission": 0.11,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:31:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S_SYMBOL_TIME",
                "session_date": "2026-05-27",
                "symbol": "MATCH",
                "side": "SLD",
                "quantity": 1,
                "price": 11,
                "commission": 0.12,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:39:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["ibkr_commission"], 0.0)
            self.assertEqual(closed["commission_status"], "MISSING")
            self.assertEqual(closed["entry_execution_count"], 0)
            self.assertEqual(closed["exit_execution_count"], 0)

    def test_reconstructed_trade_commission_matches_execution_ids_without_trade_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "reconstructed:2026-05-27:AKTX:B_AKTX:S_AKTX",
                "session_date": "2026-05-27",
                "strategy_name": "unknown",
                "symbol": "AKTX",
                "status": "CLOSED",
                "entry_price": 17.01,
                "exit_price": 17.26,
                "quantity": 5,
                "gross_pnl": 1.25,
                "mfe_pct": 5.0,
                "raw_json": {
                    "reconstruction_source": "executions_pair",
                    "buy_execution_id": "B_AKTX",
                    "sell_execution_id": "S_AKTX",
                },
            })
            self.insert_execution_direct(store, {
                "execution_id": "B_AKTX",
                "session_date": "2026-05-27",
                "strategy_name": "unknown",
                "symbol": "AKTX",
                "side": "BOT",
                "quantity": 5,
                "price": 17.01,
                "commission": 0.865515,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:37:14+00:00",
            })
            self.insert_execution_direct(store, {
                "execution_id": "S_AKTX",
                "session_date": "2026-05-27",
                "strategy_name": "unknown",
                "symbol": "AKTX",
                "side": "SLD",
                "quantity": 5,
                "price": 17.26,
                "commission": 0.880768,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:44:07+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "All")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["ibkr_commission"], 0.0)
            self.assertEqual(closed["commission_status"], "MISSING")
            self.assertEqual(closed["confirmed_commission_execution_count"], 0)
            self.assertIn("matched_by=trades_table", closed["commission_source_detail"])

    def test_fifo_lots_with_same_prices_are_not_deduped_by_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            for trade_id, qty, buy_exec, sell_exec in [
                ("reconstructed:2026-06-16:2026-06-16:RXT:B_RXT_1:S_RXT_1", 58, "B_RXT_1", "S_RXT_1"),
                ("reconstructed:2026-06-16:2026-06-16:RXT:B_RXT_2:S_RXT_1", 42, "B_RXT_2", "S_RXT_1"),
                ("reconstructed:2026-06-16:2026-06-16:RXT:B_RXT_3:S_RXT_2", 58, "B_RXT_3", "S_RXT_2"),
            ]:
                store.upsert_trade({
                    "trade_id": trade_id,
                    "session_date": "2026-06-16",
                    "strategy_name": "unknown",
                    "symbol": "RXT",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-06-16T13:35:00+00:00",
                    "exit_fill_time": "2026-06-16T14:10:00+00:00",
                    "entry_price": 1.25,
                    "exit_price": 1.30,
                    "quantity": qty,
                    "gross_pnl": round((1.30 - 1.25) * qty, 6),
                    "commission": 0.0,
                    "net_pnl": round((1.30 - 1.25) * qty, 6),
                    "raw_json": {
                        "reconstruction_source": "sqlite_execution_reducer",
                        "buy_execution_id": buy_exec,
                        "sell_execution_id": sell_exec,
                    },
                })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-16", "2026-06-16"), "All")
            closed = snapshot["closed_positions"]

            self.assertEqual(len(closed), 3)
            self.assertEqual(closed["symbol"].tolist(), ["RXT", "RXT", "RXT"])
            self.assertAlmostEqual(closed["qty"].sum(), 158.0)
            self.assertEqual(set(closed["entry_execution_id"]), {"B_RXT_1", "B_RXT_2", "B_RXT_3"})
            self.assertEqual(set(closed["exit_execution_id"]), {"S_RXT_1", "S_RXT_2"})
            self.assertEqual(snapshot["summary"]["closed_trades"], 3)

    def test_closed_trade_times_fall_back_to_execution_recorded_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_RECORDED_AT",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "TIMEFB",
                "status": "CLOSED",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 1,
                "gross_pnl": 1,
                "mfe_pct": 10,
            })
            store.upsert_execution({
                "execution_id": "B_RECORDED_AT",
                "trade_id": "T_RECORDED_AT",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "TIMEFB",
                "side": "BOT",
                "quantity": 1,
                "price": 10,
                "commission": 0.11,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:31:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S_RECORDED_AT",
                "trade_id": "T_RECORDED_AT",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "TIMEFB",
                "side": "SLD",
                "quantity": 1,
                "price": 11,
                "commission": 0.12,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:39:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertIsNone(closed["entry_time"])
            self.assertIsNone(closed["exit_time"])
            self.assertIn("MISSING_ENTRY", closed["data_quality"])
            self.assertIn("MISSING_EXIT", closed["data_quality"])

    def test_closed_trade_peak_zero_remains_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T0",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "FLAT",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T13:30:00+00:00",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 9.9,
                "quantity": 10,
                "gross_pnl": -1,
                "mfe_pct": 0,
            })
            for execution_id, side, price, ts in [
                ("B0", "BOT", 10, "2026-05-27T13:30:00+00:00"),
                ("S0", "SLD", 9.9, "2026-05-27T13:40:00+00:00"),
            ]:
                store.upsert_execution({
                    "execution_id": execution_id,
                    "trade_id": "T0",
                    "session_date": "2026-05-27",
                    "strategy_name": "v67",
                    "symbol": "FLAT",
                    "side": side,
                    "quantity": 10,
                    "price": price,
                    "commission": 0.0,
                    "commission_source": "ibkr",
                    "recorded_at": ts,
                })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertEqual(closed["peak_pct"], 0)
            self.assertEqual(closed["peak_source"], "trades.mfe_pct")
            self.assertEqual(closed["commission_status"], "MISSING")
            self.assertIn("COMMISSION_MISSING", closed["data_quality"])

    def test_peak_from_trade_raw_json_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_RAW_PEAK",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "RAWPK",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T13:30:00+00:00",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 10.5,
                "quantity": 1,
                "gross_pnl": 0.5,
                "raw_json": {"peak_gain_pct": 8.5},
            })
            for execution_id, side, price, ts in [
                ("B_RAW_PEAK", "BOT", 10, "2026-05-27T13:30:00+00:00"),
                ("S_RAW_PEAK", "SLD", 10.5, "2026-05-27T13:40:00+00:00"),
            ]:
                store.upsert_execution({
                    "execution_id": execution_id,
                    "trade_id": "T_RAW_PEAK",
                    "session_date": "2026-05-27",
                    "strategy_name": "v67",
                    "symbol": "RAWPK",
                    "side": side,
                    "quantity": 1,
                    "price": price,
                    "commission": 0.1,
                    "commission_source": "ibkr",
                    "recorded_at": ts,
                })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["peak_pct"], 8.5)
            self.assertEqual(closed["peak_source"], "trades.raw_json")

    def test_peak_from_lifecycle_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            recorder_root = Path(tmp) / "recorder"
            session_dir = recorder_root / "2026-05-27"
            session_dir.mkdir(parents=True)
            lifecycle_path = session_dir / "trade_lifecycle.csv"
            with lifecycle_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["event", "symbol", "peak_gain_pct", "recorded_at"])
                writer.writeheader()
                writer.writerow({
                    "event": "SELL_ORDER_SENT",
                    "symbol": "LIFE",
                    "peak_gain_pct": "6.25",
                    "recorded_at": "2026-05-27T13:40:00+00:00",
                })
            previous = os.environ.get("TRADING_BOT_RECORDER_DIR")
            os.environ["TRADING_BOT_RECORDER_DIR"] = str(recorder_root)
            try:
                store = SQLiteRuntimeStore(db)
                store.upsert_trade({
                    "trade_id": "T_LIFECYCLE_PEAK",
                    "session_date": "2026-05-27",
                    "strategy_name": "v67",
                    "symbol": "LIFE",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-05-27T13:30:00+00:00",
                    "exit_fill_time": "2026-05-27T13:40:00+00:00",
                    "entry_price": 10,
                    "exit_price": 10.5,
                    "quantity": 1,
                    "gross_pnl": 0.5,
                })
                for execution_id, side, price, ts in [
                    ("B_LIFECYCLE_PEAK", "BOT", 10, "2026-05-27T13:30:00+00:00"),
                    ("S_LIFECYCLE_PEAK", "SLD", 10.5, "2026-05-27T13:40:00+00:00"),
                ]:
                    store.upsert_execution({
                        "execution_id": execution_id,
                        "trade_id": "T_LIFECYCLE_PEAK",
                        "session_date": "2026-05-27",
                        "strategy_name": "v67",
                        "symbol": "LIFE",
                        "side": side,
                        "quantity": 1,
                        "price": price,
                        "commission": 0.1,
                        "commission_source": "ibkr",
                        "recorded_at": ts,
                    })
                store.close()

                snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
                closed = snapshot["closed_positions"].iloc[0]

                self.assertAlmostEqual(closed["peak_pct"], 6.25)
                self.assertEqual(closed["peak_source"], "trade_lifecycle_symbol_session")
            finally:
                if previous is None:
                    os.environ.pop("TRADING_BOT_RECORDER_DIR", None)
                else:
                    os.environ["TRADING_BOT_RECORDER_DIR"] = previous

    def test_no_peak_source_remains_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_NO_PEAK",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "NOPEAK",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T13:30:00+00:00",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 10.5,
                "quantity": 1,
                "gross_pnl": 0.5,
            })
            for execution_id, side, price, ts in [
                ("B_NO_PEAK", "BOT", 10, "2026-05-27T13:30:00+00:00"),
                ("S_NO_PEAK", "SLD", 10.5, "2026-05-27T13:40:00+00:00"),
            ]:
                self.insert_execution_direct(store, {
                    "execution_id": execution_id,
                    "trade_id": "T_NO_PEAK",
                    "session_date": "2026-05-27",
                    "strategy_name": "v67",
                    "symbol": "NOPEAK",
                    "side": side,
                    "quantity": 1,
                    "price": price,
                    "commission": 0.1,
                    "commission_source": "ibkr",
                    "recorded_at": ts,
                })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertTrue(closed["peak_pct"] != closed["peak_pct"])
            self.assertEqual(closed["peak_source"], "missing")
            self.assertEqual(snapshot["data_quality_summary"]["peak_missing"], 1)

    def test_missing_entry_time_does_not_reuse_exit_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_MISSING_ENTRY",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "TIME",
                "status": "CLOSED",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 1,
                "gross_pnl": 1,
                "mfe_pct": 10,
            })
            store.upsert_execution({
                "execution_id": "S_MISSING_ENTRY",
                "trade_id": "T_MISSING_ENTRY",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "TIME",
                "side": "SLD",
                "quantity": 1,
                "price": 11,
                "commission": 0.2,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:40:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertIsNone(closed["entry_time"])
            self.assertEqual(closed["exit_time"], "2026-05-27T13:40:00+00:00")
            self.assertNotEqual(closed["entry_time"], closed["exit_time"])
            self.assertIn("MISSING_ENTRY", closed["data_quality"])

    def test_missing_commission_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_NO_COMM",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "NOCOMM",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T13:30:00+00:00",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 1,
                "gross_pnl": 1,
                "mfe_pct": 10,
            })
            for execution_id, side, price, ts in [
                ("B_NO_COMM", "BOT", 10, "2026-05-27T13:30:00+00:00"),
                ("S_NO_COMM", "SLD", 11, "2026-05-27T13:40:00+00:00"),
            ]:
                store.upsert_execution({
                    "execution_id": execution_id,
                    "trade_id": "T_NO_COMM",
                    "session_date": "2026-05-27",
                    "strategy_name": "v67",
                    "symbol": "NOCOMM",
                    "side": side,
                    "quantity": 1,
                    "price": price,
                    "recorded_at": ts,
                })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertEqual(closed["ibkr_commission"], 0)
            self.assertEqual(closed["commission_status"], "MISSING")
            self.assertIn("COMMISSION_MISSING", closed["data_quality"])

    def test_partial_commission_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_PARTIAL_COMM",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "PART",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T13:30:00+00:00",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 1,
                "gross_pnl": 1,
                "mfe_pct": 10,
            })
            store.upsert_execution({
                "execution_id": "B_PARTIAL_COMM",
                "trade_id": "T_PARTIAL_COMM",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "PART",
                "side": "BOT",
                "quantity": 1,
                "price": 10,
                "commission": 0.3,
                "commission_source": "ibkr",
                "recorded_at": "2026-05-27T13:30:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S_PARTIAL_COMM",
                "trade_id": "T_PARTIAL_COMM",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "PART",
                "side": "SLD",
                "quantity": 1,
                "price": 11,
                "recorded_at": "2026-05-27T13:40:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["ibkr_commission"], 0.0)
            self.assertEqual(closed["commission_status"], "MISSING")
            self.assertIn("COMMISSION_MISSING", closed["data_quality"])

    def test_partial_fill_missing_one_execution_commission_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_MULTI_PARTIAL_COMM",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "MULTI",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T13:30:00+00:00",
                "exit_fill_time": "2026-05-27T13:40:00+00:00",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 2,
                "gross_pnl": 2,
                "mfe_pct": 10,
            })
            for execution_id, side, qty, price, commission in [
                ("B_MULTI_1", "BOT", 1, 10, 0.1),
                ("B_MULTI_2", "BOT", 1, 10, None),
                ("S_MULTI_1", "SLD", 2, 11, 0.2),
            ]:
                row = {
                    "execution_id": execution_id,
                    "trade_id": "T_MULTI_PARTIAL_COMM",
                    "session_date": "2026-05-27",
                    "strategy_name": "v67",
                    "symbol": "MULTI",
                    "side": side,
                    "quantity": qty,
                    "price": price,
                    "recorded_at": "2026-05-27T13:35:00+00:00",
                }
                if commission is not None:
                    row["commission"] = commission
                    row["commission_source"] = "ibkr"
                self.insert_execution_direct(store, row)
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["ibkr_commission"], 0.0)
            self.assertEqual(closed["commission_status"], "MISSING")
            self.assertEqual(closed["expected_commission_execution_count"], 0)
            self.assertEqual(closed["confirmed_commission_execution_count"], 0)

    def test_same_second_true_roundtrip_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            same_second = "2026-05-27T13:30:00+00:00"
            store.upsert_trade({
                "trade_id": "T_FAST",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "FAST",
                "status": "CLOSED",
                "entry_fill_time": same_second,
                "exit_fill_time": same_second,
                "entry_price": 10,
                "exit_price": 10.1,
                "quantity": 1,
                "gross_pnl": 0.1,
                "mfe_pct": 1,
            })
            for execution_id, side, price in [
                ("B_FAST", "BOT", 10),
                ("S_FAST", "SLD", 10.1),
            ]:
                store.upsert_execution({
                    "execution_id": execution_id,
                    "trade_id": "T_FAST",
                    "session_date": "2026-05-27",
                    "strategy_name": "v67",
                    "symbol": "FAST",
                    "side": side,
                    "quantity": 1,
                    "price": price,
                    "commission": 0.1,
                    "commission_source": "ibkr",
                    "executed_at": same_second,
                    "recorded_at": same_second,
                })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertEqual(closed["entry_time"], same_second)
            self.assertEqual(closed["exit_time"], same_second)
            self.assertNotIn("SUSPECT_TIME_MATCH", closed["data_quality"])

    def test_sessions_and_strategy_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B1",
                "session_date": "2026-05-25",
                "strategy_name": "alpha",
                "symbol": "AAA",
                "side": "BOT",
                "quantity": 1,
                "price": 10,
            })
            store.upsert_execution({
                "execution_id": "B2",
                "session_date": "2026-05-26",
                "strategy_name": "beta",
                "symbol": "BBB",
                "side": "BOT",
                "quantity": 1,
                "price": 20,
            })
            store.close()

            self.assertEqual(list_sessions(db), ["2026-05-26", "2026-05-25"])
            self.assertEqual(list_strategies(db, DateWindow("2026-05-26", "2026-05-26")), ["beta"])

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-25", "2026-05-26"), "alpha")
            self.assertEqual(set(snapshot["executions"]["strategy"].unique()), {"alpha"})

    def test_flat_execution_symbol_is_not_shown_as_stale_open_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B1",
                "session_date": "2026-05-26",
                "strategy_name": "v67",
                "symbol": "STALE",
                "side": "BOT",
                "quantity": 4,
                "price": 10,
                "recorded_at": "2026-05-26T13:30:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S1",
                "session_date": "2026-05-26",
                "strategy_name": "v67",
                "symbol": "STALE",
                "side": "SLD",
                "quantity": 4,
                "price": 11,
                "recorded_at": "2026-05-26T13:45:00+00:00",
            })
            store.upsert_position({
                "session_date": "2026-05-26",
                "strategy_name": "v67",
                "symbol": "STALE",
                "quantity": 4,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "updated_at": "2026-05-26T14:00:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-26", "2026-05-26"), "v67", include_reconstructed=True)

            self.assertEqual(snapshot["summary"]["closed_trades"], 1)
            self.assertEqual(snapshot["summary"]["open_trades"], 0)
            self.assertTrue(snapshot["open_positions"].empty)

    def test_historical_active_position_without_execution_net_is_flagged_as_carry_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_position({
                "session_date": "2026-05-26",
                "strategy_name": "v67",
                "symbol": "STALE",
                "quantity": 4,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "updated_at": "2026-05-26T14:00:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-26", "2026-05-26"), "v67")

            self.assertEqual(snapshot["summary"]["open_trades"], 1)
            self.assertFalse(snapshot["open_positions"].empty)
            self.assertEqual(snapshot["open_positions"].iloc[0]["position_bucket"], "today")

    def test_dashboard_uses_latest_position_row_and_ignores_stale_active_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_position({
                "position_key": "v67:2026-05-26:STALE:old",
                "session_date": "2026-05-26",
                "strategy_name": "v67",
                "symbol": "STALE",
                "quantity": 4,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "updated_at": "2026-05-26T14:00:00+00:00",
                "raw_json": {"market_price": 11},
            })
            store.upsert_position({
                "position_key": "v67:2026-05-26:STALE:new",
                "session_date": "2026-05-26",
                "strategy_name": "v67",
                "symbol": "STALE",
                "quantity": 4,
                "avg_price": 10,
                "active": 0,
                "status": "FLAT_CONFIRMED",
                "updated_at": "2026-05-26T20:00:00+00:00",
                "raw_json": {"ibkr_position_flat_confirmed": True},
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-26", "2026-05-26"), "v67")

            self.assertTrue(snapshot["open_positions"].empty)
            self.assertEqual(snapshot["diagnostics"]["sqlite_active_positions_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["latest_active_positions_count"], 0)
            self.assertEqual(snapshot["diagnostics"]["stale_active_positions_count"], 1)

    def test_carried_trade_closed_next_day_appears_on_exit_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_CARRIED",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "DUOT",
                "status": "CLOSED",
                "entry_fill_time": "2026-05-27T19:55:00+00:00",
                "exit_fill_time": "2026-05-28T13:35:00+00:00",
                "entry_price": 10,
                "exit_price": 11,
                "quantity": 2,
                "gross_pnl": 2,
            })
            store.upsert_execution({
                "execution_id": "B_DUOT",
                "trade_id": "T_CARRIED",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "DUOT",
                "side": "BOT",
                "quantity": 2,
                "price": 10,
                "commission": 0.5,
                "commission_source": "ibkr",
                "executed_at": "2026-05-27T19:55:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S_DUOT",
                "trade_id": "T_CARRIED",
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "DUOT",
                "side": "SLD",
                "quantity": 2,
                "price": 11,
                "commission": 0.6,
                "commission_source": "ibkr",
                "executed_at": "2026-05-28T13:35:00+00:00",
            })
            store.close()

            today = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67")
            previous = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            combined = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-28"), "v67")

            self.assertEqual(today["closed_positions"].iloc[0]["symbol"], "DUOT")
            self.assertEqual(today["closed_positions"].iloc[0]["entry_date"], "2026-05-27")
            self.assertEqual(today["closed_positions"].iloc[0]["exit_date"], "2026-05-28")
            self.assertAlmostEqual(today["closed_positions"].iloc[0]["ibkr_commission"], 0.0)
            self.assertTrue(previous["closed_positions"].empty)
            self.assertEqual(combined["closed_positions"].iloc[0]["symbol"], "DUOT")
            self.assertIn("2026-05-28", list_sessions(db))

    def test_carried_reconstructed_trade_uses_original_entry_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_position({
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "DUOT",
                "quantity": 2,
                "avg_price": 10,
                "active": 0,
                "status": "CLOSED",
                "updated_at": "2026-05-28T13:36:00+00:00",
                "raw_json": {"entry_time": "2026-05-27T19:55:00+00:00"},
            })
            store.upsert_execution({
                "execution_id": "B_DUOT_RECON",
                "trade_id": None,
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "DUOT",
                "side": "BOT",
                "quantity": 2,
                "price": 10,
                "commission": 0.5,
                "commission_source": "ibkr",
                "executed_at": "2026-05-27T19:55:00+00:00",
                "recorded_at": "2026-05-27T19:55:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S_DUOT_RECON",
                "trade_id": None,
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "DUOT",
                "side": "SLD",
                "quantity": 2,
                "price": 11,
                "commission": 0.6,
                "commission_source": "ibkr",
                "executed_at": "2026-05-28T13:35:00+00:00",
                "recorded_at": "2026-05-28T13:35:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67", include_reconstructed=True)
            closed = snapshot["closed_positions"]

            self.assertEqual(len(closed), 1)
            self.assertEqual(closed.iloc[0]["symbol"], "DUOT")
            self.assertEqual(closed.iloc[0]["entry_date"], "2026-05-27")
            self.assertEqual(closed.iloc[0]["exit_date"], "2026-05-28")
            self.assertEqual(closed.iloc[0]["closed_source"], "trades")
            self.assertNotIn("CARRIED_ENTRY_TIME_MISSING", closed.iloc[0]["data_quality"])

    def test_sell_only_carried_execution_does_not_fake_same_day_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_position({
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "DUOT",
                "quantity": 2,
                "avg_price": 10,
                "active": 0,
                "status": "CLOSED",
                "updated_at": "2026-05-28T13:36:00+00:00",
                "raw_json": {"entry_time": "2026-05-27T19:55:00+00:00"},
            })
            store.upsert_execution({
                "execution_id": "S_DUOT_ONLY",
                "trade_id": None,
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "DUOT",
                "side": "SLD",
                "quantity": 2,
                "price": 11,
                "commission": 0.6,
                "commission_source": "ibkr",
                "executed_at": "2026-05-28T13:35:00+00:00",
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67")

            self.assertTrue(snapshot["closed_positions"].empty)

    def test_open_positions_include_entry_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B_OPEN",
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "OPENX",
                "side": "BOT",
                "quantity": 3,
                "price": 10,
                "executed_at": "2026-05-28T13:31:00+00:00",
            })
            store.upsert_position({
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "OPENX",
                "quantity": 3,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "updated_at": "2026-05-28T13:40:00+00:00",
                "raw_json": {"entry_time": "2026-05-28T13:31:00+00:00", "market_price": 11},
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67")

            self.assertEqual(snapshot["open_positions"].iloc[0]["symbol"], "OPENX")
            self.assertEqual(snapshot["open_positions"].iloc[0]["entry_time"], "2026-05-28T13:31:00+00:00")

    def test_open_position_calculates_upnl_and_now_pct_from_current_price(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B_OPEN",
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "OPENX",
                "side": "BOT",
                "quantity": 5,
                "price": 10,
                "commission": 0.25,
                "commission_source": "ibkr",
                "executed_at": "2026-05-28T13:31:00+00:00",
            })
            store.upsert_position({
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "OPENX",
                "quantity": 5,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "updated_at": "2026-05-28T13:40:00+00:00",
                "raw_json": {"entry_time": "2026-05-28T13:31:00+00:00", "market_price": 10.5},
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67")
            open_row = snapshot["open_positions"].iloc[0]

            self.assertEqual(open_row["now"], 10.5)
            self.assertAlmostEqual(open_row["upnl"], 2.25)
            self.assertAlmostEqual(open_row["now_pct"], 5.0)
            self.assertEqual(open_row["price_status"], "OK")
            self.assertEqual(open_row["now_price_source"], "live_quote")

    def test_open_position_missing_current_price_does_not_fake_now_as_buy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B_OPEN",
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "MISSX",
                "side": "BOT",
                "quantity": 5,
                "price": 10,
            })
            store.upsert_position({
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "MISSX",
                "quantity": 5,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "updated_at": "2026-05-28T13:40:00+00:00",
                "raw_json": {"entry_time": "2026-05-28T13:31:00+00:00"},
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67")
            open_row = snapshot["open_positions"].iloc[0]

            self.assertIsNone(open_row["now"])
            self.assertIsNone(open_row["upnl"])
            self.assertIsNone(open_row["now_pct"])
            self.assertEqual(open_row["price_status"], "MISSING_PRICE")
            self.assertEqual(open_row["now_price_source"], "missing")

    def test_adopted_position_entry_time_uses_adopted_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B_ADOPT",
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "ADOPT",
                "side": "BOT",
                "quantity": 2,
                "price": 10,
            })
            store.upsert_position({
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "ADOPT",
                "quantity": 2,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "updated_at": "2026-05-28T06:30:00+00:00",
                "raw_json": {"entry_time": "adopted_on_restart:2026-05-28T06:20:06+00:00", "market_price": 10.2},
            })
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67")
            open_row = snapshot["open_positions"].iloc[0]

            self.assertEqual(open_row["entry_time"], "2026-05-28T06:20:06+00:00")
            self.assertEqual(open_row["entry_source"], "ADOPTED")
            self.assertGreater(open_row["hold_minutes"], 0)

    def test_carried_open_position_entry_time_falls_back_to_prior_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_trade({
                "trade_id": "T_CARRY_OPEN",
                "session_date": "2026-05-27",
                "strategy_name": "v67",
                "symbol": "CARRY",
                "status": "OPEN",
                "entry_fill_time": "2026-05-27T19:55:00+00:00",
                "entry_price": 10,
                "quantity": 2,
            })
            store.upsert_position({
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "CARRY",
                "quantity": 2,
                "avg_price": 10,
                "active": 1,
                "status": "OPEN",
                "updated_at": "2026-05-28T06:30:00+00:00",
                "raw_json": {"market_price": 10.4},
            })
            store.conn.execute(
                """
                INSERT INTO executions (
                    execution_id, strategy_name, session_date, symbol, side, quantity, price,
                    executed_at, recorded_at, commission_source, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "B_CARRY",
                    "v67",
                    "2026-05-28",
                    "CARRY",
                    "BOT",
                    2,
                    10,
                    "2026-05-28T06:30:00+00:00",
                    "2026-05-28T06:30:00+00:00",
                    "missing",
                    "{}",
                ),
            )
            store.conn.commit()
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67")
            open_row = snapshot["open_positions"].iloc[0]

            self.assertEqual(open_row["entry_time"], "2026-05-27T19:55:00+00:00")
            self.assertEqual(open_row["entry_source"], "trade")

    def test_rejected_entry_is_excluded_from_open_positions_and_listed_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_position({
                "session_date": "2026-05-28",
                "strategy_name": "v67",
                "symbol": "CONL",
                "quantity": 2,
                "avg_price": 50,
                "active": 0,
                "status": "ENTRY_REJECTED",
                "updated_at": "2026-05-28T13:36:00+00:00",
                "raw_json": {
                    "reject_reason": "no_trading_permission_kid",
                    "ibkr_error_code": 201,
                    "order_id": "123",
                },
            })
            store.record_runtime_event(
                event_time="2026-05-28T13:36:00+00:00",
                event_type="ENTRY_ORDER_REJECTED",
                severity="WARN",
                strategy_name="v67",
                session_date="2026-05-28",
                symbol="CONL",
                order_id="123",
                reason="no_trading_permission_kid",
                raw_json={
                    "quantity": 2,
                    "price": 42.5,
                    "ibkr_error_code": 201,
                    "reject_reason": "no_trading_permission_kid",
                },
            )
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-28", "2026-05-28"), "v67")

            self.assertTrue(snapshot["open_positions"].empty)
            self.assertEqual(len(snapshot["rejected_entries"]), 1)
            self.assertEqual(snapshot["rejected_entries"].iloc[0]["symbol"], "CONL")
            self.assertEqual(snapshot["rejected_entries"].iloc[0]["reason"], "no_trading_permission_kid")
            self.assertEqual(snapshot["rejected_entries"].iloc[0]["price"], 42.5)
            self.assertEqual(snapshot["rejected_entries"].iloc[0]["rejected_at"], "2026-05-28T13:36:00+00:00")

    def test_repeated_dashboard_snapshot_metrics_are_stable_when_db_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.upsert_execution({
                "execution_id": "B_STABLE_DASH",
                "session_date": "2026-05-29",
                "strategy_name": "v67",
                "symbol": "DASH",
                "side": "BOT",
                "quantity": 2,
                "price": 10,
                "commission": 0.35,
                "commission_source": "ibkr",
                "executed_at": "2026-05-29T13:30:00+00:00",
            })
            store.upsert_execution({
                "execution_id": "S_STABLE_DASH",
                "session_date": "2026-05-29",
                "strategy_name": "v67",
                "symbol": "DASH",
                "side": "SLD",
                "quantity": 2,
                "price": 11,
                "commission": 0.36,
                "commission_source": "ibkr",
                "executed_at": "2026-05-29T13:40:00+00:00",
            })
            store.close()

            def metrics() -> tuple:
                snap = load_dashboard_snapshot(db, DateWindow("2026-05-29", "2026-05-29"), "v67")
                summary = snap["summary"]
                quality = snap["data_quality_summary"]
                diagnostics = snap["diagnostics"]
                return (
                    summary["closed_trades"],
                    round(summary["gross_pnl"], 6),
                    round(summary["commissions"], 6),
                    round(summary["net_actual_pnl"], 6),
                    quality["commission_ok"],
                    quality["commission_partial"],
                    quality["commission_missing"],
                    diagnostics["trades_count"],
                    diagnostics["reconstructed_trades_count"],
                    diagnostics["displayed_closed_trades_count"],
                )

            first = metrics()
            second = metrics()
            self.assertEqual(first, second)

    def test_diagnostics_tolerates_legacy_trades_schema_without_updated_at(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE runtime_events (
                event_time TEXT,
                event_type TEXT,
                reason TEXT,
                session_date TEXT,
                strategy_name TEXT
            );
            CREATE TABLE risk_events (
                event_time TEXT,
                event_type TEXT,
                reason TEXT,
                session_date TEXT,
                strategy_name TEXT,
                repeat_count INTEGER
            );
            CREATE TABLE reconciliation_runs (
                started_at TEXT,
                finished_at TEXT,
                orphan_count INTEGER,
                drift_count INTEGER
            );
            CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                strategy_name TEXT,
                session_date TEXT,
                exit_fill_time TEXT,
                closed_at TEXT,
                raw_json TEXT
            );
            INSERT INTO trades (trade_id, strategy_name, session_date, exit_fill_time, closed_at, raw_json)
            VALUES ('T1', 'v67', '2026-05-29', '2026-05-29T13:45:00+00:00', '2026-05-29T13:45:00+00:00', '{}');
            """
        )
        try:
            diagnostics = load_diagnostics(conn, DateWindow("2026-05-29", "2026-05-29"), "v67")
        finally:
            conn.close()

        self.assertEqual(diagnostics["trades_count"], 1)
        self.assertEqual(diagnostics["trades_updated_last_60s"], 0)
        self.assertEqual(diagnostics["last_reducer_run_at"], "")

    def test_dashboard_snapshot_migrates_existing_db_before_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            store.close()

            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            try:
                cols = [
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(trades)").fetchall()
                    if row["name"] not in {"updated_at", "trade_reduction_version"}
                ]
                col_sql = ", ".join(cols)
                conn.execute(f"CREATE TABLE trades_legacy AS SELECT {col_sql} FROM trades")
                conn.execute("DROP TABLE trades")
                conn.execute("ALTER TABLE trades_legacy RENAME TO trades")
                conn.commit()
                before = {row["name"] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
                self.assertNotIn("updated_at", before)
                self.assertNotIn("trade_reduction_version", before)
            finally:
                conn.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-29", "2026-05-29"), "All")
            self.assertEqual(snapshot["diagnostics"]["trades_count"], 0)

            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            try:
                after = {row["name"] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
            finally:
                conn.close()
            self.assertIn("updated_at", after)
            self.assertIn("trade_reduction_version", after)

    def test_duplicate_active_symbol_prefers_execution_reducer_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-15:ELMT:live",
                    "session_date": "2026-06-15",
                    "strategy_name": "v67",
                    "symbol": "ELMT",
                    "quantity": 2,
                    "avg_price": 10,
                    "active": 1,
                    "status": "OPEN",
                    "source": "live_buy",
                    "updated_at": "2026-06-15T13:30:00+00:00",
                    "raw_json": {"entry_price": 10, "market_price": 10.4, "entry_time": "2026-06-15T13:30:00+00:00"},
                })
                store.upsert_position({
                    "position_key": "v67:2026-06-15:ELMT:reducer",
                    "session_date": "2026-06-15",
                    "strategy_name": "v67",
                    "symbol": "ELMT",
                    "quantity": 2,
                    "avg_price": 10,
                    "ibkr_quantity": 2,
                    "active": 1,
                    "status": "OPEN",
                    "source": "sqlite_execution_reducer",
                    "updated_at": "2026-06-15T13:29:00+00:00",
                    "raw_json": {"entry_fill_verified": True, "entry_price": 10, "market_price": 10.5, "entry_time": "2026-06-15T13:29:00+00:00"},
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-15", "2026-06-15"), "v67")
            open_positions = snapshot["open_positions"]
            self.assertEqual(len(open_positions), 1)
            self.assertEqual(open_positions.iloc[0]["symbol"], "ELMT")
            self.assertEqual(open_positions.iloc[0]["source"], "sqlite_execution_reducer")

    def test_latest_closed_row_suppresses_older_active_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-15:BRCB:open",
                    "session_date": "2026-06-15",
                    "strategy_name": "v67",
                    "symbol": "BRCB",
                    "quantity": 1,
                    "avg_price": 20,
                    "active": 1,
                    "status": "OPEN",
                    "updated_at": "2026-06-15T13:30:00+00:00",
                    "raw_json": {"entry_price": 20, "market_price": 21, "entry_time": "2026-06-15T13:30:00+00:00"},
                })
                store.upsert_position({
                    "position_key": "v67:2026-06-15:BRCB:closed",
                    "session_date": "2026-06-15",
                    "strategy_name": "v67",
                    "symbol": "BRCB",
                    "quantity": 0,
                    "avg_price": 21,
                    "ibkr_quantity": 0,
                    "active": 0,
                    "status": "CLOSED",
                    "updated_at": "2026-06-15T13:40:00+00:00",
                    "raw_json": {"active": False, "ibkr_position_flat_confirmed": True},
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-15", "2026-06-15"), "v67")
            self.assertTrue(snapshot["open_positions"].empty)

    def test_stale_unconfirmed_carry_open_remains_visible_unless_status_is_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-03:ACMR",
                    "session_date": "2026-06-03",
                    "strategy_name": "v67",
                    "symbol": "ACMR",
                    "quantity": 1,
                    "avg_price": 30,
                    "active": 1,
                    "status": "OPEN",
                    "updated_at": "2026-06-03T13:30:00+00:00",
                    "raw_json": {"entry_price": 30, "market_price": 31, "entry_time": "2026-06-03T13:30:00+00:00"},
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-15", "2026-06-15"), "v67")
            self.assertEqual(len(snapshot["open_positions"]), 1)
            self.assertEqual(snapshot["open_positions"].iloc[0]["symbol"], "ACMR")
            self.assertEqual(snapshot["summary"]["open_trades"], 1)
            self.assertFalse(snapshot["orphan_stale_positions"].empty)
            row = snapshot["orphan_stale_positions"].iloc[0]
            self.assertEqual(row["symbol"], "ACMR")
            self.assertIn("ORPHAN_STALE_POSITION", row["data_quality"])
            self.assertEqual(snapshot["diagnostics"]["orphan_stale_position_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["oldest_orphan_stale_position"], "ACMR")
            self.assertEqual(snapshot["diagnostics"]["today_open_positions_count"], 0)
            self.assertEqual(snapshot["diagnostics"]["stale_carry_open_count"], 1)

    def test_explicit_orphan_stale_status_is_excluded_from_main_open_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-03:MRAM",
                    "session_date": "2026-06-03",
                    "strategy_name": "v67",
                    "symbol": "MRAM",
                    "quantity": 1,
                    "avg_price": 30,
                    "active": 1,
                    "status": "ORPHAN_STALE_POSITION",
                    "updated_at": "2026-06-15T13:30:00+00:00",
                    "raw_json": {"entry_price": 30, "market_price": 31, "entry_time": "2026-06-03T13:30:00+00:00"},
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-15", "2026-06-15"), "v67")

            self.assertTrue(snapshot["open_positions"].empty)
            self.assertEqual(snapshot["summary"]["open_trades"], 0)
            self.assertEqual(snapshot["diagnostics"]["active_positions_raw_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["displayed_open_positions_count"], 0)

    def test_today_active_position_with_zero_ibkr_quantity_is_still_displayed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-16:ARTV",
                    "session_date": "2026-06-16",
                    "strategy_name": "v67",
                    "symbol": "ARTV",
                    "quantity": 8,
                    "avg_price": 4.20,
                    "ibkr_quantity": 0,
                    "active": 1,
                    "status": "OPEN",
                    "updated_at": "2026-06-16T14:09:04+00:00",
                    "raw_json": {
                        "entry_price": 4.20,
                        "market_price": 4.35,
                        "market_price_at": "2026-06-16T14:09:05+00:00",
                        "entry_time": "2026-06-16T14:09:04+00:00",
                    },
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-16", "2026-06-16"), "v67")

            self.assertEqual(len(snapshot["open_positions"]), 1)
            row = snapshot["open_positions"].iloc[0]
            self.assertEqual(row["symbol"], "ARTV")
            self.assertEqual(row["position_bucket"], "today")
            self.assertEqual(snapshot["summary"]["open_trades"], 1)
            self.assertEqual(snapshot["diagnostics"]["active_positions_raw_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["active_positions_today_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["displayed_open_positions_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["orphan_stale_position_count"], 0)
            self.assertTrue(snapshot["excluded_open_positions"].empty)

    def test_active_position_is_not_hidden_by_flat_execution_net(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-16:ASYS",
                    "session_date": "2026-06-16",
                    "strategy_name": "v67",
                    "symbol": "ASYS",
                    "quantity": 5,
                    "avg_price": 7.10,
                    "active": 1,
                    "status": "OPEN",
                    "updated_at": "2026-06-16T14:06:25+00:00",
                    "raw_json": {
                        "entry_price": 7.10,
                        "market_price": 7.25,
                        "market_price_at": "2026-06-16T14:06:30+00:00",
                        "entry_time": "2026-06-16T14:06:25+00:00",
                    },
                })
                self.insert_execution_direct(store, {
                    "execution_id": "ASYS_B1",
                    "session_date": "2026-06-16",
                    "strategy_name": "v67",
                    "symbol": "ASYS",
                    "side": "BOT",
                    "quantity": 5,
                    "price": 7.10,
                    "recorded_at": "2026-06-16T13:50:00+00:00",
                })
                self.insert_execution_direct(store, {
                    "execution_id": "ASYS_S1",
                    "session_date": "2026-06-16",
                    "strategy_name": "v67",
                    "symbol": "ASYS",
                    "side": "SLD",
                    "quantity": 5,
                    "price": 7.20,
                    "recorded_at": "2026-06-16T14:00:00+00:00",
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-16", "2026-06-16"), "v67")

            self.assertEqual(len(snapshot["open_positions"]), 1)
            self.assertEqual(snapshot["open_positions"].iloc[0]["symbol"], "ASYS")
            self.assertEqual(snapshot["summary"]["open_trades"], 1)
            self.assertEqual(snapshot["diagnostics"]["active_positions_after_orphan_filter_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["displayed_open_positions_count"], 1)

    def test_confirmed_stale_active_open_is_still_flagged_as_carry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-03:ACMR",
                    "session_date": "2026-06-03",
                    "strategy_name": "v67",
                    "symbol": "ACMR",
                    "quantity": 1,
                    "avg_price": 30,
                    "active": 1,
                    "status": "OPEN",
                    "updated_at": "2026-06-03T13:30:00+00:00",
                    "raw_json": {
                        "entry_price": 30,
                        "market_price": 31,
                        "entry_time": "2026-06-03T13:30:00+00:00",
                        "entry_fill_verified": True,
                    },
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-15", "2026-06-15"), "v67")
            row = snapshot["open_positions"].iloc[0]
            self.assertEqual(row["position_bucket"], "carry_stale")
            self.assertIn("STALE_CARRY_OPEN", row["data_quality"])
            self.assertTrue(snapshot["orphan_stale_positions"].empty)
            self.assertEqual(snapshot["diagnostics"]["stale_carry_open_count"], 1)

    def test_unconfirmed_stale_active_open_is_not_hidden_without_orphan_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-05-11:MRAM",
                    "session_date": "2026-05-11",
                    "strategy_name": "v67",
                    "symbol": "MRAM",
                    "quantity": 3,
                    "avg_price": 31.65,
                    "ibkr_quantity": 0,
                    "active": 1,
                    "status": "OPEN",
                    "updated_at": "2026-06-16T14:09:04+00:00",
                    "raw_json": {
                        "entry_price": 31.65,
                        "market_price": 30.50,
                        "market_price_at": "2026-06-16T14:09:05+00:00",
                        "entry_time": "2026-05-11T13:30:00+00:00",
                    },
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-16", "2026-06-16"), "v67")

            self.assertEqual(len(snapshot["raw_active_positions"]), 1)
            self.assertEqual(len(snapshot["open_positions"]), 1)
            self.assertEqual(snapshot["open_positions"].iloc[0]["symbol"], "MRAM")
            self.assertEqual(snapshot["open_positions"].iloc[0]["position_bucket"], "carry_stale")
            self.assertEqual(snapshot["diagnostics"]["raw_active_sqlite_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["displayed_open_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["dropped_open_count"], 0)
            self.assertEqual(snapshot["diagnostics"]["dropped_symbols"], "")

    def test_dropped_open_diagnostics_report_explicit_orphan_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-16:FGMC",
                    "session_date": "2026-06-16",
                    "strategy_name": "v67",
                    "symbol": "FGMC",
                    "quantity": 104,
                    "avg_price": 1.50,
                    "active": 1,
                    "status": "OPEN",
                    "updated_at": "2026-06-16T14:49:36+00:00",
                    "raw_json": {"entry_time": "2026-06-16T14:48:43+00:00", "market_price": 1.55},
                })
                store.upsert_position({
                    "position_key": "v67:2026-05-11:MRAM",
                    "session_date": "2026-05-11",
                    "strategy_name": "v67",
                    "symbol": "MRAM",
                    "quantity": 3,
                    "avg_price": 31.65,
                    "active": 1,
                    "status": "ORPHAN_STALE_POSITION",
                    "updated_at": "2026-06-16T14:09:04+00:00",
                    "raw_json": {"entry_time": "2026-05-11T13:30:00+00:00", "market_price": 30.50},
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-16", "2026-06-16"), "v67")

            self.assertEqual(len(snapshot["raw_active_positions"]), 2)
            self.assertEqual(len(snapshot["open_positions"]), 1)
            self.assertEqual(snapshot["open_positions"].iloc[0]["symbol"], "FGMC")
            self.assertEqual(snapshot["diagnostics"]["raw_active_sqlite_count"], 2)
            self.assertEqual(snapshot["diagnostics"]["displayed_open_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["dropped_open_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["dropped_symbols"], "MRAM")

    def test_carried_closed_trade_is_flagged_on_exit_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_trade({
                    "trade_id": "T_CARRY_CLOSE",
                    "strategy_name": "v67",
                    "session_date": "2026-05-13",
                    "symbol": "OUST",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-05-13T13:30:00+00:00",
                    "exit_fill_time": "2026-06-15T13:45:00+00:00",
                    "closed_at": "2026-06-15T13:45:00+00:00",
                    "entry_price": 10,
                    "exit_price": 11,
                    "quantity": 1,
                    "gross_pnl": 1,
                    "commission": 0,
                    "net_pnl": 1,
                    "raw_json": {},
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-15", "2026-06-15"), "v67")
            row = snapshot["closed_positions"].iloc[0]
            self.assertTrue(bool(row["carried_closed_today"]))
            self.assertIn("CARRIED_POSITION_CLOSED_TODAY", row["data_quality"])
            self.assertEqual(snapshot["diagnostics"]["carried_closed_today_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["raw_closed_trade_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["displayed_closed_trade_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["dropped_closed_trade_count"], 0)
            self.assertEqual(snapshot["diagnostics"]["dropped_closed_trade_ids"], "")

    def test_reconstructed_carry_closed_trade_does_not_inflate_runtime_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_trade({
                    "trade_id": "reconstructed:2026-05-13:2026-06-16:OUST:B_OLD:S_TODAY",
                    "strategy_name": "v67",
                    "session_date": "2026-05-13",
                    "symbol": "OUST",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-05-13T13:30:00+00:00",
                    "exit_fill_time": "2026-06-16T14:45:00+00:00",
                    "closed_at": "2026-06-16T14:45:00+00:00",
                    "entry_price": 1.00,
                    "exit_price": 12.487727,
                    "quantity": 22,
                    "gross_pnl": 252.73,
                    "commission": 1.025094,
                    "net_pnl": 251.704906,
                    "raw_json": {
                        "reconstruction_source": "sqlite_execution_reducer",
                        "buy_execution_id": "B_OLD",
                        "sell_execution_id": "S_TODAY",
                    },
                })
            finally:
                store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-06-16", "2026-06-16"), "v67")

            self.assertEqual(snapshot["summary"]["closed_trades"], 1)
            self.assertAlmostEqual(snapshot["summary"]["gross_pnl"], 0.0)
            self.assertAlmostEqual(snapshot["summary"]["net_actual_pnl"], 0.0)
            row = snapshot["closed_positions"].iloc[0]
            self.assertFalse(bool(row["runtime_pnl_trusted"]))
            self.assertIn("CARRY_BASIS_UNVERIFIED", row["data_quality"])
            self.assertTrue(pd.isna(row["gross"]))
            self.assertTrue(pd.isna(row["net_actual"]))
            self.assertEqual(snapshot["diagnostics"]["untrusted_carry_closed_count"], 1)
            self.assertEqual(snapshot["diagnostics"]["untrusted_carry_closed_symbols"], "OUST")


if __name__ == "__main__":
    unittest.main()
