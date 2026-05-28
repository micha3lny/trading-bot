from __future__ import annotations

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
            self.assertEqual(snapshot["open_positions"].iloc[0]["symbol"], "BBB")
            self.assertEqual(snapshot["diagnostics"]["delayed_fills"], 1)
            self.assertEqual(snapshot["diagnostics"]["risk_guard_blocks"], 1)

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
