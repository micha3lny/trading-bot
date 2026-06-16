from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from scripts.cleanup_runtime_events import cleanup_runtime_events
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore
from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v66_ibkr_account_recorder import record_recent_fills


class FakeIB:
    def __init__(self, fills):
        self._fills = fills

    def reqExecutions(self, _filter):
        return list(self._fills)


class FailingStore:
    def upsert_execution(self, _row):
        raise RuntimeError("sqlite down")


def fake_fill(exec_id: str, commission: float | None = None):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol="RKLB"),
        execution=SimpleNamespace(
            execId=exec_id,
            side="BOT",
            shares=10,
            price=10.0,
            orderId=101,
            permId=202,
            exchange="SMART",
            lastLiquidity=1,
            time="2026-05-22T13:30:00+00:00",
        ),
        commissionReport=SimpleNamespace(
            execId=exec_id,
            commission=commission,
            currency="USD",
            realizedPNL=0.0,
        ),
    )


class SQLiteRuntimeStoreTests(unittest.TestCase):
    def test_schema_initializes_and_wal_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                tables = {row["name"] for row in store.query("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertIn("runtime_events", tables)
                self.assertIn("executions", tables)
                self.assertEqual(store.query("PRAGMA journal_mode")[0]["journal_mode"].lower(), "wal")
            finally:
                store.close()

    def test_upsert_execution_idempotent_and_commission_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "E1", "symbol": "RKLB", "side": "BOT", "quantity": 10, "price": 10, "commission_source": "missing"})
                store.upsert_execution({"execution_id": "E1", "symbol": "RKLB", "side": "BOT", "quantity": 10, "price": 10, "commission": 0.35, "commission_source": "ibkr"})
                rows = store.query("SELECT * FROM executions WHERE execution_id = 'E1'")
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["commission"], 0.35)
                self.assertEqual(rows[0]["commission_source"], "ibkr")
            finally:
                store.close()

    def test_bot_fill_creates_open_position(self) -> None:
        today = date.today().isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({
                    "execution_id": "B_OPEN",
                    "strategy_name": "v67",
                    "session_date": today,
                    "symbol": "AKTX",
                    "side": "BOT",
                    "quantity": 5,
                    "price": 10.0,
                    "executed_at": f"{today}T13:35:00+00:00",
                    "recorded_at": f"{today}T13:35:00+00:00",
                    "commission": 0.35,
                    "commission_source": "ibkr",
                })

                positions = store.query("SELECT * FROM positions WHERE symbol = 'AKTX'")
                trades = store.query("SELECT * FROM trades")

                self.assertEqual(len(positions), 1)
                self.assertEqual(positions[0]["active"], 1)
                self.assertEqual(positions[0]["status"], "OPEN")
                self.assertEqual(positions[0]["ibkr_quantity"], 5)
                self.assertEqual(len(trades), 0)
            finally:
                store.close()

    def test_sld_fill_closes_position_and_creates_closed_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({
                    "execution_id": "B1",
                    "strategy_name": "v67",
                    "session_date": "2026-05-29",
                    "symbol": "AKTX",
                    "side": "BOT",
                    "quantity": 5,
                    "price": 10.0,
                    "executed_at": "2026-05-29T13:35:00+00:00",
                    "recorded_at": "2026-05-29T13:35:00+00:00",
                    "commission": 0.35,
                    "commission_source": "ibkr",
                })
                store.upsert_execution({
                    "execution_id": "S1",
                    "strategy_name": "v67",
                    "session_date": "2026-05-29",
                    "symbol": "AKTX",
                    "side": "SLD",
                    "quantity": 5,
                    "price": 11.0,
                    "executed_at": "2026-05-29T14:05:00+00:00",
                    "recorded_at": "2026-05-29T14:05:00+00:00",
                    "commission": 0.36,
                    "commission_source": "ibkr",
                })

                latest = store.get_latest_position("AKTX")
                trades = store.query("SELECT * FROM trades WHERE symbol = 'AKTX'")

                self.assertIsNotNone(latest)
                self.assertEqual(latest["active"], 0)
                self.assertEqual(latest["status"], "CLOSED")
                self.assertEqual(len(trades), 1)
                trade = trades[0]
                self.assertEqual(trade["status"], "CLOSED")
                self.assertEqual(trade["entry_fill_time"], "2026-05-29T13:35:00+00:00")
                self.assertEqual(trade["exit_fill_time"], "2026-05-29T14:05:00+00:00")
                self.assertAlmostEqual(trade["gross_pnl"], 5.0)
                self.assertAlmostEqual(trade["commission"], 0.71)
                self.assertAlmostEqual(trade["net_pnl"], 4.29)
                self.assertEqual(trade["ibkr_entry_confirmed"], 1)
                self.assertEqual(trade["ibkr_exit_confirmed"], 1)
            finally:
                store.close()

    def test_commission_report_update_refreshes_closed_trade_commission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B1", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "RKLB", "side": "BOT", "quantity": 2, "price": 10, "executed_at": "2026-05-29T13:30:00+00:00", "commission_source": "missing"})
                store.upsert_execution({"execution_id": "S1", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "RKLB", "side": "SLD", "quantity": 2, "price": 12, "executed_at": "2026-05-29T13:40:00+00:00", "commission_source": "missing"})
                self.assertEqual(store.query("SELECT commission FROM trades")[0]["commission"], 0.0)

                store.upsert_execution({"execution_id": "B1", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "RKLB", "side": "BOT", "quantity": 2, "price": 10, "executed_at": "2026-05-29T13:30:00+00:00", "commission": 0.35, "commission_source": "ibkr"})
                store.upsert_execution({"execution_id": "S1", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "RKLB", "side": "SLD", "quantity": 2, "price": 12, "executed_at": "2026-05-29T13:40:00+00:00", "commission": 0.36, "commission_source": "ibkr"})

                trade = store.query("SELECT gross_pnl, commission, net_pnl FROM trades")[0]
                self.assertAlmostEqual(trade["gross_pnl"], 4.0)
                self.assertAlmostEqual(trade["commission"], 0.71)
                self.assertAlmostEqual(trade["net_pnl"], 3.29)
            finally:
                store.close()

    def test_commission_update_keeps_trade_count_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B_STABLE", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "STBL", "side": "BOT", "quantity": 3, "price": 10, "executed_at": "2026-05-29T13:30:00+00:00", "commission_source": "missing"})
                store.upsert_execution({"execution_id": "S_STABLE", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "STBL", "side": "SLD", "quantity": 3, "price": 11, "executed_at": "2026-05-29T13:40:00+00:00", "commission_source": "missing"})
                before = store.query("SELECT COUNT(*) AS n, SUM(gross_pnl) AS gross, SUM(commission) AS commission, SUM(net_pnl) AS net FROM trades")[0]

                store.upsert_execution({"execution_id": "B_STABLE", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "STBL", "side": "BOT", "quantity": 3, "price": 10, "executed_at": "2026-05-29T13:30:00+00:00", "commission": 0.35, "commission_source": "ibkr"})
                store.upsert_execution({"execution_id": "S_STABLE", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "STBL", "side": "SLD", "quantity": 3, "price": 11, "executed_at": "2026-05-29T13:40:00+00:00", "commission": 0.36, "commission_source": "ibkr"})
                after = store.query("SELECT COUNT(*) AS n, SUM(gross_pnl) AS gross, SUM(commission) AS commission, SUM(net_pnl) AS net FROM trades")[0]

                self.assertEqual(before["n"], 1)
                self.assertEqual(after["n"], 1)
                self.assertAlmostEqual(before["gross"], after["gross"])
                self.assertAlmostEqual(after["commission"], 0.71)
                self.assertAlmostEqual(after["net"], (before["gross"] or 0) - 0.71)
            finally:
                store.close()

    def test_persisted_trade_dedupe_is_atomic_to_final_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B_DEDUPE", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "DDUP", "side": "BOT", "quantity": 1, "price": 20, "executed_at": "2026-05-29T13:30:00+00:00"})
                store.upsert_execution({"execution_id": "S_DEDUPE", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "DDUP", "side": "SLD", "quantity": 1, "price": 21, "executed_at": "2026-05-29T13:40:00+00:00"})
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM trades")[0]["n"], 1)

                store.upsert_trade({
                    "trade_id": "PERSISTED-DDUP",
                    "strategy_name": "v67",
                    "session_date": "2026-05-29",
                    "symbol": "DDUP",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-05-29T13:30:00+00:00",
                    "exit_fill_time": "2026-05-29T13:40:00+00:00",
                    "entry_price": 20,
                    "exit_price": 21,
                    "quantity": 1,
                    "gross_pnl": 1,
                    "net_pnl": 1,
                    "raw_json": {"source": "persisted_test"},
                })

                rows = store.query("SELECT trade_id FROM trades WHERE symbol = 'DDUP'")
                execution_links = store.query("SELECT DISTINCT trade_id FROM executions WHERE symbol = 'DDUP'")
                self.assertEqual([row["trade_id"] for row in rows], ["PERSISTED-DDUP"])
                self.assertEqual([row["trade_id"] for row in execution_links], ["PERSISTED-DDUP"])
            finally:
                store.close()

    def test_entry_rejected_position_creates_no_trade_or_open_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_position({
                    "position_key": "v67:2026-05-29:CONL",
                    "strategy_name": "v67",
                    "session_date": "2026-05-29",
                    "symbol": "CONL",
                    "status": "ENTRY_REJECTED",
                    "quantity": 3,
                    "avg_price": 25,
                    "active": 0,
                    "raw_json": {"entry_fill_verified": False, "reject_reason": "no_trading_permission_kid"},
                })

                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM trades")[0]["n"], 0)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM positions WHERE active = 1")[0]["n"], 0)
            finally:
                store.close()

    def test_eod_sell_creates_closed_trade_from_existing_buy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B_EOD", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "EODX", "side": "BOT", "quantity": 4, "price": 8, "executed_at": "2026-05-29T13:35:00+00:00"})
                store.upsert_execution({"execution_id": "S_EOD", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "EODX", "side": "SLD", "quantity": 4, "price": 7.5, "executed_at": "2026-05-29T19:59:00+00:00", "raw_json": {"reason": "eod_flatten"}})

                trade = store.query("SELECT * FROM trades WHERE symbol = 'EODX'")[0]
                position = store.get_latest_position("EODX")
                self.assertEqual(trade["status"], "CLOSED")
                self.assertEqual(trade["exit_fill_time"], "2026-05-29T19:59:00+00:00")
                self.assertAlmostEqual(trade["gross_pnl"], -2.0)
                self.assertEqual(position["active"], 0)
            finally:
                store.close()

    def test_fifo_one_buy_two_sells_consumes_quantity_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B100", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "FIFO", "side": "BOT", "quantity": 100, "price": 10, "executed_at": "2026-05-29T13:30:00+00:00"})
                store.upsert_execution({"execution_id": "S40", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "FIFO", "side": "SLD", "quantity": 40, "price": 11, "executed_at": "2026-05-29T13:40:00+00:00"})
                store.upsert_execution({"execution_id": "S60", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "FIFO", "side": "SLD", "quantity": 60, "price": 12, "executed_at": "2026-05-29T13:50:00+00:00"})

                rows = store.query("SELECT trade_id, quantity, gross_pnl, raw_json FROM trades WHERE symbol = 'FIFO' ORDER BY exit_fill_time")

                self.assertEqual(len(rows), 2)
                self.assertAlmostEqual(sum(row["quantity"] for row in rows), 100)
                self.assertAlmostEqual(rows[0]["quantity"], 40)
                self.assertAlmostEqual(rows[0]["gross_pnl"], 40)
                self.assertAlmostEqual(rows[1]["quantity"], 60)
                self.assertAlmostEqual(rows[1]["gross_pnl"], 120)
                self.assertEqual(len({row["trade_id"] for row in rows}), 2)
                for row in rows:
                    raw = json.loads(row["raw_json"])
                    self.assertEqual(raw["buy_execution_id"], "B100")
                    self.assertIn(raw["sell_execution_id"], {"S40", "S60"})
                    self.assertEqual(raw["buy_original_quantity"], 100)
            finally:
                store.close()

    def test_fifo_two_buys_one_sell_consumes_lots_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B15", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "LOTX", "side": "BOT", "quantity": 15, "price": 10, "executed_at": "2026-05-29T13:30:00+00:00"})
                store.upsert_execution({"execution_id": "B101", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "LOTX", "side": "BOT", "quantity": 101, "price": 11, "executed_at": "2026-05-29T13:35:00+00:00"})
                store.upsert_execution({"execution_id": "S116", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "LOTX", "side": "SLD", "quantity": 116, "price": 12, "executed_at": "2026-05-29T13:40:00+00:00"})

                rows = store.query("SELECT quantity, entry_price, exit_price, gross_pnl, raw_json FROM trades WHERE symbol = 'LOTX' ORDER BY entry_fill_time")

                self.assertEqual(len(rows), 2)
                self.assertAlmostEqual(sum(row["quantity"] for row in rows), 116)
                self.assertAlmostEqual(rows[0]["quantity"], 15)
                self.assertAlmostEqual(rows[0]["gross_pnl"], 30)
                self.assertAlmostEqual(rows[1]["quantity"], 101)
                self.assertAlmostEqual(rows[1]["gross_pnl"], 101)
                self.assertEqual(json.loads(rows[0]["raw_json"])["buy_execution_id"], "B15")
                self.assertEqual(json.loads(rows[1]["raw_json"])["buy_execution_id"], "B101")
            finally:
                store.close()

    def test_reducer_rerun_rebuilds_same_trades_without_reusing_execution_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B_RERUN", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "RERUN", "side": "BOT", "quantity": 100, "price": 10, "executed_at": "2026-05-29T13:30:00+00:00"})
                store.upsert_execution({"execution_id": "S_RERUN_1", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "RERUN", "side": "SLD", "quantity": 40, "price": 11, "executed_at": "2026-05-29T13:40:00+00:00"})
                store.upsert_execution({"execution_id": "S_RERUN_2", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "RERUN", "side": "SLD", "quantity": 60, "price": 12, "executed_at": "2026-05-29T13:50:00+00:00"})
                before = store.query("SELECT trade_id, quantity, gross_pnl FROM trades WHERE symbol = 'RERUN' ORDER BY trade_id")

                store.rebuild_symbol_trade_state("RERUN")
                after = store.query("SELECT trade_id, quantity, gross_pnl FROM trades WHERE symbol = 'RERUN' ORDER BY trade_id")

                self.assertEqual(before, after)
                self.assertEqual(len(after), 2)
                self.assertAlmostEqual(sum(row["quantity"] for row in after), 100)
            finally:
                store.close()

    def test_old_carried_buy_today_sell_is_consumed_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B_CARRY", "strategy_name": "v67", "session_date": "2026-05-28", "symbol": "CARRY", "side": "BOT", "quantity": 3, "price": 20, "executed_at": "2026-05-28T18:00:00+00:00"})
                store.upsert_execution({"execution_id": "S_CARRY", "strategy_name": "v67", "session_date": "2026-05-29", "symbol": "CARRY", "side": "SLD", "quantity": 3, "price": 21, "executed_at": "2026-05-29T13:40:00+00:00"})
                store.rebuild_symbol_trade_state("CARRY")

                rows = store.query("SELECT trade_id, session_date, entry_fill_time, exit_fill_time, quantity FROM trades WHERE symbol = 'CARRY'")

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["session_date"], "2026-05-28")
                self.assertEqual(rows[0]["entry_fill_time"], "2026-05-28T18:00:00+00:00")
                self.assertEqual(rows[0]["exit_fill_time"], "2026-05-29T13:40:00+00:00")
                self.assertAlmostEqual(rows[0]["quantity"], 3)
            finally:
                store.close()

    def test_reducer_does_not_resurrect_historical_open_lot_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.conn.execute(
                    """
                    INSERT INTO executions (
                        execution_id, strategy_name, session_date, symbol, side,
                        quantity, price, executed_at, recorded_at, commission_source, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "B_OLD_OPEN", "v67", "2026-05-13", "OLDOPEN", "BOT",
                        5, 10, "2026-05-13T13:30:00+00:00", "2026-05-13T13:30:00+00:00", "missing", "{}",
                    ),
                )
                store.conn.commit()
                store.rebuild_symbol_trade_state("OLDOPEN")

                active = store.query("SELECT * FROM positions WHERE symbol = 'OLDOPEN' AND COALESCE(active, 0) = 1")
                stale = store.query("SELECT * FROM positions WHERE symbol = 'OLDOPEN' AND status = 'STALE_CARRY_OPEN'")

                self.assertEqual(active, [])
                self.assertEqual(len(stale), 1)
                self.assertEqual(stale[0]["active"], 0)
            finally:
                store.close()

    def test_full_close_sets_position_inactive(self) -> None:
        today = date.today().isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B_FULL", "strategy_name": "v67", "session_date": today, "symbol": "FULL", "side": "BOT", "quantity": 10, "price": 5, "executed_at": f"{today}T13:30:00+00:00"})
                store.upsert_execution({"execution_id": "S_FULL", "strategy_name": "v67", "session_date": today, "symbol": "FULL", "side": "SLD", "quantity": 10, "price": 6, "executed_at": f"{today}T13:45:00+00:00"})

                active = store.query("SELECT * FROM positions WHERE symbol = 'FULL' AND COALESCE(active, 0) = 1")
                closed = store.query("SELECT * FROM trades WHERE symbol = 'FULL' AND status = 'CLOSED'")

                self.assertEqual(active, [])
                self.assertEqual(len(closed), 1)
                self.assertAlmostEqual(closed[0]["quantity"], 10)
            finally:
                store.close()

    def test_partial_close_reduces_active_quantity(self) -> None:
        today = date.today().isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B_PART", "strategy_name": "v67", "session_date": today, "symbol": "PART", "side": "BOT", "quantity": 100, "price": 10, "executed_at": f"{today}T13:30:00+00:00"})
                store.upsert_execution({"execution_id": "S_PART", "strategy_name": "v67", "session_date": today, "symbol": "PART", "side": "SLD", "quantity": 40, "price": 11, "executed_at": f"{today}T13:45:00+00:00"})

                active = store.query("SELECT * FROM positions WHERE symbol = 'PART' AND COALESCE(active, 0) = 1")
                closed = store.query("SELECT * FROM trades WHERE symbol = 'PART' AND status = 'CLOSED'")

                self.assertEqual(len(active), 1)
                self.assertAlmostEqual(active[0]["quantity"], 60)
                self.assertAlmostEqual(active[0]["ibkr_quantity"], 60)
                self.assertEqual(len(closed), 1)
                self.assertAlmostEqual(closed[0]["quantity"], 40)
            finally:
                store.close()

    def test_reducer_keeps_current_lot_and_suppresses_old_unmatched_lot(self) -> None:
        today = date.today().isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_execution({"execution_id": "B_OLD_MIX", "strategy_name": "v67", "session_date": "2026-05-13", "symbol": "MIXED", "side": "BOT", "quantity": 5, "price": 10, "executed_at": "2026-05-13T13:30:00+00:00"})
                store.upsert_execution({"execution_id": "B_NEW_MIX", "strategy_name": "v67", "session_date": today, "symbol": "MIXED", "side": "BOT", "quantity": 2, "price": 20, "executed_at": f"{today}T13:31:00+00:00"})
                result = store.rebuild_symbol_trade_state("MIXED")

                active = store.query("SELECT * FROM positions WHERE symbol = 'MIXED' AND COALESCE(active, 0) = 1")
                stale = store.query("SELECT * FROM positions WHERE symbol = 'MIXED' AND status = 'STALE_CARRY_OPEN'")

                self.assertAlmostEqual(result["open_quantity"], 2)
                self.assertAlmostEqual(result["suppressed_historical_open_quantity"], 5)
                self.assertEqual(len(active), 1)
                self.assertAlmostEqual(active[0]["quantity"], 2)
                self.assertEqual(len(stale), 1)
                self.assertEqual(stale[0]["active"], 0)
            finally:
                store.close()

    def test_repair_rebuild_clears_active_position_without_execution_net(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_position({
                    "position_key": "v67:2026-05-13:STALE",
                    "strategy_name": "v67",
                    "session_date": "2026-05-13",
                    "symbol": "STALE",
                    "status": "OPEN",
                    "quantity": 8,
                    "avg_price": 12,
                    "source": "live_buy",
                    "ibkr_quantity": 8,
                    "active": 1,
                    "raw_json": {"active": True},
                })
                result = store.rebuild_positions_from_executions(["STALE"])

                active = store.query("SELECT * FROM positions WHERE symbol = 'STALE' AND COALESCE(active, 0) = 1")

                self.assertEqual(active, [])
                self.assertEqual(result["symbols_processed"], 1)
                self.assertEqual(result["open_symbols_count"], 0)
            finally:
                store.close()

    def test_runtime_event_append_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.record_runtime_event(event_type="TEST_EVENT", symbol="RKLB")
                store.record_runtime_event(event_type="TEST_EVENT", symbol="RKLB")
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM runtime_events")[0]["n"], 2)
            finally:
                store.close()

    def test_repeated_buy_blocked_is_throttled_but_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                for _ in range(1000):
                    store.record_runtime_event(
                        event_time="2026-06-14T13:30:00+00:00",
                        event_type="BUY_BLOCKED",
                        session_date="2026-06-14",
                        symbol="AKTX",
                        reason="entries_blocked",
                        raw_json={"large_payload": "x" * 1000},
                    )

                runtime_events = store.query("SELECT event_type, raw_json FROM runtime_events")
                counters = store.query(
                    """
                    SELECT count
                    FROM runtime_event_counters
                    WHERE date = '2026-06-14'
                      AND event_type = 'BUY_BLOCKED'
                      AND symbol = 'AKTX'
                      AND reason = 'entries_blocked'
                    """
                )

                self.assertLessEqual(len(runtime_events), 2)
                self.assertEqual(counters[0]["count"], 1000)
                self.assertEqual(runtime_events[0]["event_type"], "BUY_BLOCKED_SUMMARY")
                self.assertLess(len(runtime_events[0]["raw_json"] or ""), 300)
            finally:
                store.close()

    def test_runtime_event_cleanup_preserves_ledger_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.record_runtime_event(event_time="2026-05-01T13:30:00+00:00", event_type="OLD_EVENT", session_date="2026-05-01")
                store.upsert_execution({"execution_id": "E_KEEP", "symbol": "KEEP", "side": "BOT", "quantity": 1, "price": 10, "executed_at": "2026-05-01T13:31:00+00:00"})
                store.upsert_order({"order_key": "O_KEEP", "symbol": "KEEP", "side": "BUY", "quantity": 1})
                store.upsert_trade({"trade_id": "T_KEEP", "symbol": "KEEP", "status": "CLOSED", "quantity": 1})

                result = cleanup_runtime_events(store, older_than_days=1, apply=True, batch_size=1)

                self.assertGreaterEqual(result["runtime_events_matching_before"], 1)
                self.assertGreaterEqual(result["runtime_events_deleted"], 1)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM runtime_events")[0]["n"], 0)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM executions")[0]["n"], 1)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM orders")[0]["n"], 1)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM trades")[0]["n"], 1)
            finally:
                store.close()

    def test_risk_event_upsert_increments_repeat_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.record_risk_event(risk_event_id="R1", event_type="RISK_GUARD_BLOCK_ENTRY", blocked=1, reason="max_loss")
                store.record_risk_event(risk_event_id="R1", event_type="RISK_GUARD_BLOCK_ENTRY", blocked=1, reason="max_loss")
                row = store.query("SELECT repeat_count FROM risk_events WHERE risk_event_id = 'R1'")[0]
                self.assertEqual(row["repeat_count"], 2)
            finally:
                store.close()

    def test_reconciliation_market_data_and_daily_feature_upserts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.record_reconciliation_run(run_id="REC1", mode="startup", clean=0, orphan_count=1)
                store.upsert_market_data_session({"date": "2026-05-22", "session_type": "RTH", "symbol": "RKLB", "rows": 390, "collection_status": "complete"})
                store.upsert_symbol_daily_feature({"date": "2026-05-22", "symbol": "RKLB", "feature_version": "v1", "rank": 1, "final_score": 12.3})
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM reconciliation_runs")[0]["n"], 1)
                self.assertEqual(store.query("SELECT rows FROM market_data_sessions")[0]["rows"], 390)
                self.assertEqual(store.query("SELECT rank FROM symbol_daily_features")[0]["rank"], 1)
            finally:
                store.close()

    def test_eod_success_marks_active_positions_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_position({
                    "session_date": "2026-05-28",
                    "strategy_name": "v67",
                    "symbol": "RKLB",
                    "quantity": 10,
                    "avg_price": 10,
                    "active": 1,
                    "status": "OPEN",
                    "raw_json": {"entry_price": 10},
                })

                updated = store.mark_all_positions_flat(reason="eod_success", status="FLAT_CONFIRMED")

                row = store.query("SELECT active, status, raw_json FROM positions WHERE symbol = 'RKLB'")[0]
                self.assertEqual(updated, 1)
                self.assertEqual(row["active"], 0)
                self.assertEqual(row["status"], "FLAT_CONFIRMED")
                self.assertTrue(json.loads(row["raw_json"])["ibkr_position_flat_confirmed"])
            finally:
                store.close()

    def test_reconciliation_clean_with_ibkr_flat_clears_active_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_position({
                    "session_date": "2026-05-28",
                    "strategy_name": "v67",
                    "symbol": "CRSR",
                    "quantity": 3,
                    "avg_price": 20,
                    "active": 1,
                    "status": "OPEN",
                })

                store.mark_all_positions_flat(reason="reconciliation_clean", status="FLAT_CONFIRMED")

                row = store.query("SELECT active, status, raw_json FROM positions WHERE symbol = 'CRSR'")[0]
                self.assertEqual(row["active"], 0)
                self.assertEqual(row["status"], "FLAT_CONFIRMED")
                self.assertEqual(json.loads(row["raw_json"])["flat_confirmed_reason"], "reconciliation_clean")
            finally:
                store.close()

    def test_sqlite_failure_does_not_break_csv_fill_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = LiveDataRecorder(Path(tmp), session_date="2026-05-22")
            setattr(recorder, "sqlite_store", FailingStore())

            count = record_recent_fills(FakeIB([fake_fill("E1")]), recorder, seen=set())

            self.assertEqual(count, 1)
            with recorder.path("fills.csv").open(errors="replace") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["execution_id"], "E1")


if __name__ == "__main__":
    unittest.main()
