from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.cleanup_duplicate_stale_open_positions import cleanup as cleanup_duplicate_stale_open_positions
from scripts.cleanup_stale_positions import active_symbols_from_managed, find_stale_rows
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


class CleanupStalePositionsTests(unittest.TestCase):
    def test_cleanup_dry_run_identifies_stale_active_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            managed = Path(tmp) / "managed_positions.json"
            managed.write_text('{"positions":{"KEEP":{"quantity":2,"active":true}}}', encoding="utf-8")
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({"strategy_name": "v67", "session_date": "2026-05-28", "symbol": "KEEP", "quantity": 2, "active": 1, "status": "OPEN"})
                store.upsert_position({"strategy_name": "v67", "session_date": "2026-05-28", "symbol": "STALE", "quantity": 3, "active": 1, "status": "OPEN"})

                stale = find_stale_rows(store, active_symbols_from_managed(managed))

                self.assertEqual([row["symbol"] for row in stale], ["STALE"])
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM positions WHERE active = 1")[0]["n"], 2)
            finally:
                store.close()

    def test_cleanup_apply_clears_stale_active_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({"strategy_name": "v67", "session_date": "2026-05-28", "symbol": "STALE", "quantity": 3, "active": 1, "status": "OPEN"})
                stale = find_stale_rows(store, keep_symbols=set())
                for row in stale:
                    store.mark_position_flat(
                        symbol=row["symbol"],
                        strategy_name=row["strategy_name"],
                        session_date=row["session_date"],
                        reason="test_cleanup",
                        status="FLAT_CONFIRMED",
                    )

                row = store.query("SELECT active, status FROM positions WHERE symbol = 'STALE'")[0]
                self.assertEqual(row["active"], 0)
                self.assertEqual(row["status"], "FLAT_CONFIRMED")
            finally:
                store.close()

    def test_duplicate_stale_cleanup_marks_orphan_stale_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "strategy_name": "v67",
                    "session_date": "2026-05-11",
                    "symbol": "MRAM",
                    "quantity": 3,
                    "avg_price": 31.65,
                    "active": 1,
                    "status": "OPEN",
                    "updated_at": "2026-05-11T13:30:00+00:00",
                    "raw_json": {"entry_time": "2026-05-11T13:30:00+00:00"},
                })
            finally:
                store.close()

            dry_run = cleanup_duplicate_stale_open_positions(db, "2026-06-16", apply=False, stale_days=7)
            self.assertEqual(dry_run["stale_unconfirmed_to_suppress"], 1)

            applied = cleanup_duplicate_stale_open_positions(db, "2026-06-16", apply=True, stale_days=7)
            self.assertEqual(applied["stale_unconfirmed_to_suppress"], 1)

            store = SQLiteRuntimeStore(db)
            try:
                row = store.query("SELECT active, status, raw_json FROM positions WHERE symbol = 'MRAM'")[0]
                self.assertEqual(row["active"], 0)
                self.assertEqual(row["status"], "ORPHAN_STALE_POSITION")
                self.assertIn("orphan_stale_position", row["raw_json"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
