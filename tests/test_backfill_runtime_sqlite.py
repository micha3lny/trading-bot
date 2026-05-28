from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.backfill_runtime_sqlite import enrich_trades_from_runtime_events, import_session
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

    def test_backfill_reconstructs_closed_trade_from_execution_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recorder"
            session_dir = root / "2026-05-27"
            session_dir.mkdir(parents=True)
            write_csv(
                session_dir / "fills.csv",
                [
                    {
                        "execution_id": "B1",
                        "symbol": "MRAM",
                        "action": "BOT",
                        "quantity": "2",
                        "fill_price": "10",
                        "commission": "",
                        "commission_source": "missing",
                        "raw_json": json.dumps({"execution": {"time": "2026-05-27T13:31:00+00:00"}}),
                    },
                    {
                        "execution_id": "S1",
                        "symbol": "MRAM",
                        "action": "SLD",
                        "quantity": "2",
                        "fill_price": "11",
                        "commission": "",
                        "commission_source": "missing",
                        "raw_json": json.dumps({"execution": {"time": "2026-05-27T13:41:00+00:00"}}),
                    },
                ],
            )
            write_csv(
                session_dir / "trade_lifecycle.csv",
                [
                    {
                        "recorded_at": "2026-05-27T13:35:00+00:00",
                        "event": "PEAK_UPDATED",
                        "symbol": "MRAM",
                        "price": "10.8",
                        "entry_price": "10",
                        "peak_price": "10.8",
                        "pnl_pct": "",
                        "reason": "",
                    },
                    {
                        "recorded_at": "2026-05-27T13:41:00+00:00",
                        "event": "SELL_ORDER_SENT",
                        "symbol": "MRAM",
                        "price": "11",
                        "entry_price": "10",
                        "peak_price": "10.8",
                        "pnl_pct": "10",
                        "reason": "trail",
                    },
                ],
            )
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                import_session(store, session_dir)
                trades = store.query("SELECT * FROM trades WHERE symbol = 'MRAM'")
                executions = store.query("SELECT execution_id, executed_at FROM executions ORDER BY execution_id")

                self.assertEqual(len(trades), 1)
                self.assertEqual(trades[0]["status"], "CLOSED")
                self.assertEqual(trades[0]["entry_fill_time"], "2026-05-27T13:31:00+00:00")
                self.assertEqual(trades[0]["exit_fill_time"], "2026-05-27T13:41:00+00:00")
                self.assertAlmostEqual(trades[0]["gross_pnl"], 2.0)
                self.assertAlmostEqual(trades[0]["mfe_pct"], 8.0)
                self.assertIn("drop_from_peak_pct", trades[0]["raw_json"])
                self.assertIn("executions_pair", trades[0]["raw_json"])
                self.assertEqual(executions[0]["executed_at"], "2026-05-27T13:31:00+00:00")
                self.assertEqual(executions[1]["executed_at"], "2026-05-27T13:41:00+00:00")
            finally:
                store.close()

    def test_backfill_enriches_peak_from_runtime_event_without_session_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                store.upsert_trade({
                    "trade_id": "TNULL",
                    "session_date": "2026-05-27",
                    "strategy_name": "unknown",
                    "symbol": "AKTX",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-05-27T13:30:00+00:00",
                    "exit_fill_time": "2026-05-27T13:40:00+00:00",
                    "entry_price": 17.0,
                    "exit_price": 17.3778,
                    "quantity": 1,
                    "gross_pnl": 0.3778,
                })
                store.conn.execute(
                    """
                    INSERT INTO runtime_events (
                        event_time, severity, event_type, strategy_name, session_date, symbol, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "2026-05-27T13:35:00+00:00",
                        "INFO",
                        "PEAK_UPDATED",
                        "unknown",
                        None,
                        "AKTX",
                        '{"entry_price": 17.0, "peak_price": 17.97}',
                    ),
                )
                updated = enrich_trades_from_runtime_events(store, "2026-05-27")
                trades = store.query("SELECT mfe_pct, raw_json FROM trades WHERE trade_id = 'TNULL'")

                self.assertEqual(updated, 1)
                self.assertAlmostEqual(trades[0]["mfe_pct"], 5.7059, places=3)
                self.assertIn("drop_from_peak_pct", trades[0]["raw_json"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
