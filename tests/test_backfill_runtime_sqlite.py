from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.backfill_runtime_sqlite import import_session
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class BackfillRuntimeSQLiteTests(unittest.TestCase):
    def test_backfill_imports_temp_session_twice_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recorder"
            session_dir = root / "2026-05-22"
            session_dir.mkdir(parents=True)

            write_csv(
                session_dir / "fills.csv",
                [
                    {
                        "execution_id": "EXEC1",
                        "symbol": "RKLB",
                        "action": "BOT",
                        "quantity": "10",
                        "fill_price": "10.5",
                        "order_id": "101",
                        "perm_id": "202",
                        "exchange": "SMART",
                        "liquidity": "1",
                        "commission": "0.35",
                        "commission_currency": "USD",
                        "realized_pnl": "0",
                        "commission_source": "ibkr",
                        "raw_json": "{}",
                    }
                ],
            )
            write_csv(
                session_dir / "trade_lifecycle.csv",
                [
                    {
                        "timestamp": "2026-05-22T13:30:00+00:00",
                        "event": "ENTRY_ORDER_FILLED",
                        "symbol": "RKLB",
                        "reason": "test",
                    }
                ],
            )
            with (session_dir / "order_lifecycle.jsonl").open("w") as fh:
                fh.write(json.dumps({"timestamp": "2026-05-22T13:31:00+00:00", "event": "POSITION_OPENED", "symbol": "RKLB"}) + "\n")
            (session_dir / "managed_positions.json").write_text(
                json.dumps({"RKLB": {"symbol": "RKLB", "quantity": 10, "entry_price": 10.5, "active": True}}),
            )
            (session_dir / "eod_summary.json").write_text(
                json.dumps({"clean": 1, "open_positions": 0, "fractional_orphans": 0, "whole_share_orphans": 0, "pending_orders": 0}),
            )
            write_csv(
                session_dir / "run_metadata.csv",
                [{"timestamp": "2026-05-22T13:00:00+00:00", "event": "RUN_START", "value": "paper"}],
            )

            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                import_session(store, session_dir)
                first_runtime_events = store.query("SELECT COUNT(*) AS n FROM runtime_events")[0]["n"]
                import_session(store, session_dir)
                second_runtime_events = store.query("SELECT COUNT(*) AS n FROM runtime_events")[0]["n"]

                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM executions")[0]["n"], 1)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM positions")[0]["n"], 1)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM reconciliation_runs")[0]["n"], 1)
                self.assertEqual(first_runtime_events, second_runtime_events)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
