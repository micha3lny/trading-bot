from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_runtime_event_append_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.record_runtime_event(event_type="TEST_EVENT", symbol="RKLB")
                store.record_runtime_event(event_type="TEST_EVENT", symbol="RKLB")
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM runtime_events")[0]["n"], 2)
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
