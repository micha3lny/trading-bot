from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
