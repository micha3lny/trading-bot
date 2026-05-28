from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.dashboard.runtime_queries import DateWindow, list_sessions, list_strategies, load_dashboard_snapshot, utc_today
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


class RuntimeDashboardQueriesTests(unittest.TestCase):
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

            snapshot = load_dashboard_snapshot(db, DateWindow(session_date, session_date), "v67")

            self.assertEqual(snapshot["summary"]["closed_trades"], 1)
            self.assertEqual(snapshot["summary"]["open_trades"], 1)
            self.assertAlmostEqual(snapshot["summary"]["gross_pnl"], 5.0)
            self.assertAlmostEqual(snapshot["summary"]["net_actual_pnl"], 4.5)
            self.assertEqual(snapshot["closed_positions"].iloc[0]["symbol"], "AAA")
            self.assertIn("RECONSTRUCTED_FROM_EXECUTIONS", snapshot["closed_positions"].iloc[0]["data_quality"])
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

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "All")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertEqual(closed["symbol"], "MRAM")
            self.assertAlmostEqual(closed["gross"], 2.0)
            self.assertEqual(closed["entry_time"], "2026-05-27T13:31:00+00:00")
            self.assertEqual(closed["exit_time"], "2026-05-27T13:41:00+00:00")
            self.assertEqual(closed["commission_status"], "OK")
            self.assertIn("RECONSTRUCTED_FROM_EXECUTIONS", closed["data_quality"])

    def test_execution_pair_without_executed_at_keeps_missing_times(self) -> None:
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

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "All")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertEqual(closed["symbol"], "GRRR")
            self.assertIsNone(closed["entry_time"])
            self.assertIsNone(closed["exit_time"])
            self.assertTrue(closed["hold_minutes"] is None)
            self.assertEqual(closed["commission_status"], "MISSING")
            self.assertIn("MISSING_EXECUTION_TIME", closed["data_quality"])
            self.assertIn("RECONSTRUCTED_FROM_EXECUTIONS", closed["data_quality"])
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
            store.upsert_execution({
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
            store.upsert_execution({
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

            self.assertAlmostEqual(closed["ibkr_commission"], 0.75)
            self.assertAlmostEqual(closed["net_actual"], 9.25)
            self.assertAlmostEqual(closed["net_pct"], 9.25)
            self.assertEqual(closed["commission_status"], "OK")
            self.assertEqual(closed["data_quality"], "OK")
            self.assertEqual(closed["entry_execution_count"], 1)
            self.assertEqual(closed["exit_execution_count"], 1)
            self.assertEqual(closed["confirmed_commission_execution_count"], 2)
            self.assertEqual(closed["expected_commission_execution_count"], 2)
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

            self.assertAlmostEqual(closed["ibkr_commission"], 0.23)
            self.assertEqual(closed["commission_status"], "OK")
            self.assertEqual(closed["entry_execution_count"], 1)
            self.assertEqual(closed["exit_execution_count"], 1)

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
            self.assertEqual(closed["commission_status"], "OK")
            self.assertEqual(closed["data_quality"], "OK")

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
                store.upsert_execution({
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

            self.assertAlmostEqual(closed["ibkr_commission"], 0.3)
            self.assertEqual(closed["commission_status"], "PARTIAL")
            self.assertIn("COMMISSION_PARTIAL", closed["data_quality"])

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
                store.upsert_execution(row)
            store.close()

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-27", "2026-05-27"), "v67")
            closed = snapshot["closed_positions"].iloc[0]

            self.assertAlmostEqual(closed["ibkr_commission"], 0.3)
            self.assertEqual(closed["commission_status"], "PARTIAL")
            self.assertEqual(closed["expected_commission_execution_count"], 3)
            self.assertEqual(closed["confirmed_commission_execution_count"], 2)

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

            snapshot = load_dashboard_snapshot(db, DateWindow("2026-05-26", "2026-05-26"), "v67")

            self.assertEqual(snapshot["summary"]["closed_trades"], 1)
            self.assertEqual(snapshot["summary"]["open_trades"], 0)
            self.assertTrue(snapshot["open_positions"].empty)

    def test_historical_active_position_without_execution_net_is_hidden(self) -> None:
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

            self.assertEqual(snapshot["summary"]["open_trades"], 0)
            self.assertTrue(snapshot["open_positions"].empty)


if __name__ == "__main__":
    unittest.main()
