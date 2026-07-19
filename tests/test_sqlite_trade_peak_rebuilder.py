from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live_trading.ranking.daily_top100_builder import parquet_path
from src.live_trading.storage import sqlite_store as sqlite_store_module
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


def add_execution(store: SQLiteRuntimeStore, execution_id: str, side: str, qty: float, price: float, ts: str, *, symbol: str = "PEAKSQL", realized_pnl: float | None = None) -> None:
    store.upsert_execution(
        {
            "execution_id": execution_id,
            "strategy_name": "v67",
            "session_date": ts[:10],
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "price": price,
            "executed_at": ts,
            "recorded_at": ts,
            "commission": 0.0,
            "commission_source": "ibkr",
            "realized_pnl": realized_pnl,
        }
    )


def write_history(history_dir: Path, symbol: str, session_date: str, rows: list[tuple[str, float, float]]) -> None:
    path = parquet_path(history_dir, symbol, pd.Timestamp(session_date).date(), "RTH")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime([row[0] for row in rows], utc=True),
            "open": [row[1] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[2] for row in rows],
        }
    ).to_parquet(path)


class SQLiteTradePeakRebuilderTests(unittest.TestCase):
    def test_store_calculates_peak_after_canonical_trade_exists(self) -> None:
        previous_async = sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED
        previous_history = sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR
        sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED = False
        with tempfile.TemporaryDirectory() as tmp:
            history_dir = Path(tmp) / "history"
            sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR = history_dir
            write_history(
                history_dir,
                "PEAKSQL",
                "2026-07-16",
                [
                    ("2026-07-16T13:30:00+00:00", 10.2, 9.9),
                    ("2026-07-16T13:31:00+00:00", 11.0, 10.1),
                    ("2026-07-16T13:32:00+00:00", 10.7, 10.4),
                ],
            )
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                add_execution(store, "B1", "BOT", 10, 10, "2026-07-16T13:30:00+00:00")
                add_execution(store, "S1", "SLD", 10, 10.5, "2026-07-16T13:32:00+00:00", realized_pnl=5)
                trade = store.query("SELECT trade_id FROM trades WHERE symbol = 'PEAKSQL' AND status = 'CLOSED'")[0]

                result = store.calculate_and_store_trade_peak(trade["trade_id"])
                row = store.query("SELECT mfe_pct, peak_price, giveback_from_peak, raw_json FROM trades WHERE trade_id = ?", [trade["trade_id"]])[0]
                raw = json.loads(row["raw_json"])

                self.assertEqual(result["peak_data_quality"], "EXACT")
                self.assertAlmostEqual(row["peak_price"], 11.0)
                self.assertAlmostEqual(row["mfe_pct"], 10.0)
                self.assertAlmostEqual(row["giveback_from_peak"], 5.0)
                self.assertEqual(raw["peak_source"], "canonical_trade_candles_1m")
                self.assertEqual(raw["peak_version"], 2)
                self.assertIn("peak_calculated_at", raw)
            finally:
                store.close()
                sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED = previous_async
                sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR = previous_history

    def test_missing_candles_store_null_peak_not_zero(self) -> None:
        previous_async = sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED
        previous_history = sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR
        sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED = False
        with tempfile.TemporaryDirectory() as tmp:
            sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR = Path(tmp) / "missing_history"
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                add_execution(store, "B1", "BOT", 10, 10, "2026-07-16T13:30:00+00:00", symbol="MISSPEAK")
                add_execution(store, "S1", "SLD", 10, 11, "2026-07-16T13:32:00+00:00", symbol="MISSPEAK", realized_pnl=10)
                trade = store.query("SELECT trade_id FROM trades WHERE symbol = 'MISSPEAK' AND status = 'CLOSED'")[0]

                result = store.calculate_and_store_trade_peak(trade["trade_id"])
                row = store.query("SELECT mfe_pct, peak_price, giveback_from_peak, raw_json FROM trades WHERE trade_id = ?", [trade["trade_id"]])[0]
                raw = json.loads(row["raw_json"])

                self.assertEqual(result["peak_data_quality"], "MISSING_CANDLES")
                self.assertIsNone(row["mfe_pct"])
                self.assertIsNone(row["peak_price"])
                self.assertIsNone(row["giveback_from_peak"])
                self.assertEqual(raw["peak_data_quality"], "MISSING_CANDLES")
                self.assertEqual(raw["peak_source"], "unavailable")
            finally:
                store.close()
                sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED = previous_async
                sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR = previous_history

    def test_canonical_peak_rebuild_removes_stale_raw_peak_fields(self) -> None:
        previous_async = sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED
        previous_history = sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR
        sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED = False
        with tempfile.TemporaryDirectory() as tmp:
            history_dir = Path(tmp) / "history"
            sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR = history_dir
            write_history(
                history_dir,
                "RAWPEAK",
                "2026-07-16",
                [
                    ("2026-07-16T13:30:00+00:00", 10.0, 9.8),
                    ("2026-07-16T13:31:00+00:00", 10.0, 9.6),
                ],
            )
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                add_execution(store, "B1", "BOT", 10, 10, "2026-07-16T13:30:00+00:00", symbol="RAWPEAK")
                add_execution(store, "S1", "SLD", 10, 9.9, "2026-07-16T13:31:00+00:00", symbol="RAWPEAK", realized_pnl=-1)
                trade = store.query("SELECT trade_id FROM trades WHERE symbol = 'RAWPEAK' AND status = 'CLOSED'")[0]
                store.execute(
                    "UPDATE trades SET raw_json = ? WHERE trade_id = ?",
                    [
                        json.dumps({"peak_gain_pct": 0.0, "drop_from_peak_pct": -29.0, "giveback_pct": -29.0, "peak_position_key": "stale"}),
                        trade["trade_id"],
                    ],
                )

                store.calculate_and_store_trade_peak(trade["trade_id"])
                row = store.query("SELECT mfe_pct, peak_price, giveback_from_peak, raw_json FROM trades WHERE trade_id = ?", [trade["trade_id"]])[0]
                raw = json.loads(row["raw_json"])

                self.assertAlmostEqual(row["mfe_pct"], 0.0)
                self.assertAlmostEqual(row["peak_price"], 10.0)
                self.assertAlmostEqual(raw["drop_from_peak_pct"], -1.0)
                self.assertNotIn("giveback_pct", raw)
                self.assertNotIn("peak_gain_pct", raw)
                self.assertNotIn("peak_position_key", raw)
            finally:
                store.close()
                sqlite_store_module.TRADE_PEAK_ASYNC_CALCULATION_ENABLED = previous_async
                sqlite_store_module.DEFAULT_TRADE_PEAK_HISTORY_DIR = previous_history


if __name__ == "__main__":
    unittest.main()
