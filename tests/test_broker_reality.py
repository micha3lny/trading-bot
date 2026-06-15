from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard.broker_reality import (
    compare_positions,
    compare_closed_trades,
    ensure_asyncio_event_loop,
    load_sqlite_executions,
    match_executions,
    parse_ibkr_activity_csv,
    reconstruct_closed_trades_fifo,
)
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


class BrokerRealityTests(unittest.TestCase):
    def test_ensure_asyncio_event_loop_in_worker_thread(self) -> None:
        result: dict[str, str] = {}

        def worker() -> None:
            result["info"] = ensure_asyncio_event_loop()
            loop = asyncio.get_event_loop()
            result["closed"] = str(loop.is_closed())
            loop.close()

        thread = threading.Thread(target=worker, name="ScriptRunner.scriptThread")
        thread.start()
        thread.join()

        self.assertIn("loop", result["info"])
        self.assertEqual(result["closed"], "False")

    def test_ibkr_activity_csv_parser_simple_header(self) -> None:
        csv = """Execution ID,Symbol,Side,Quantity,Price,Commission,Time,Order ID,Perm ID,Account,Exchange,Currency
E1,AKTX,BUY,5,17.01,0.86,2026-06-15 13:35:39,101,501,U123,SMART,USD
"""
        rows = parse_ibkr_activity_csv(csv)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["execution_id"], "E1")
        self.assertEqual(rows.iloc[0]["symbol"], "AKTX")
        self.assertEqual(rows.iloc[0]["side"], "BUY")
        self.assertAlmostEqual(rows.iloc[0]["commission"], 0.86)

    def test_ibkr_activity_csv_parser_flex_trades_section(self) -> None:
        csv = """Trades,Header,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,Comm/Fee,Buy/Sell,Exec ID
Trades,Data,Stocks,USD,MRAM,2026-06-15 13:35:39,3,31.65,-1.23,BUY,EX123
"""
        rows = parse_ibkr_activity_csv(csv)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["symbol"], "MRAM")
        self.assertEqual(rows.iloc[0]["side"], "BUY")
        self.assertAlmostEqual(rows.iloc[0]["commission"], 1.23)
        self.assertEqual(rows.iloc[0]["execution_id"], "EX123")

    def test_execution_matching_by_execution_id(self) -> None:
        broker = pd.DataFrame([
            {"execution_id": "E1", "symbol": "AKTX", "side": "BUY", "quantity": 5, "price": 17.01, "commission": 0.86, "execution_time": "2026-06-15T13:35:39+00:00"}
        ])
        sqlite = pd.DataFrame([
            {"execution_id": "E1", "symbol": "AKTX", "side": "BUY", "quantity": 5, "price": 17.01, "commission": 0.86, "execution_time": "2026-06-15T13:35:39+00:00"}
        ])

        matched, missing, extra, mismatches = match_executions(broker, sqlite)

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]["matched_by"], "execution_id")
        self.assertTrue(missing.empty)
        self.assertTrue(extra.empty)
        self.assertTrue(mismatches.empty)

    def test_execution_matching_fallback_without_execution_id(self) -> None:
        broker = pd.DataFrame([
            {"execution_id": "", "symbol": "AKTX", "side": "BUY", "quantity": 5, "price": 17.01, "commission": None, "execution_time": "2026-06-15T13:35:40+00:00"}
        ])
        sqlite = pd.DataFrame([
            {"execution_id": "", "symbol": "AKTX", "side": "BUY", "quantity": 5, "price": 17.01, "commission": None, "execution_time": "2026-06-15T13:35:39+00:00"}
        ])

        matched, missing, extra, mismatches = match_executions(broker, sqlite)

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]["matched_by"], "symbol_side_qty_price_time")
        self.assertTrue(missing.empty)
        self.assertTrue(extra.empty)
        self.assertTrue(mismatches.empty)

    def test_broker_fifo_reconstructs_closed_trade(self) -> None:
        executions = pd.DataFrame([
            {
                "execution_time": "2026-06-15T13:30:00+00:00",
                "symbol": "AKTX",
                "side": "BUY",
                "quantity": 5,
                "price": 10.0,
                "commission": 0.5,
                "execution_id": "B1",
            },
            {
                "execution_time": "2026-06-15T14:00:00+00:00",
                "symbol": "AKTX",
                "side": "SELL",
                "quantity": 5,
                "price": 11.0,
                "commission": 0.6,
                "execution_id": "S1",
            },
        ])

        trades = reconstruct_closed_trades_fifo(executions, "2026-06-15")

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["source"], "BROKER_FIFO_RECONSTRUCTED")
        self.assertAlmostEqual(trades.iloc[0]["realized_pnl"], 5.0)
        self.assertAlmostEqual(trades.iloc[0]["commission"], 1.1)
        self.assertAlmostEqual(trades.iloc[0]["net_pnl"], 3.9)

    def test_closed_trade_comparison_detects_pnl_mismatch(self) -> None:
        broker = pd.DataFrame([
            {"symbol": "AKTX", "quantity": 5, "entry_price": 10.0, "exit_price": 11.0, "realized_pnl": 5.0, "commission": 1.0, "net_pnl": 4.0}
        ])
        sqlite = pd.DataFrame([
            {"symbol": "AKTX", "quantity": 5, "entry_price": 10.0, "exit_price": 11.0, "realized_pnl": 4.5, "commission": 1.0, "net_pnl": 3.5}
        ])

        matched, missing, extra, mismatches = compare_closed_trades(broker, sqlite)

        self.assertEqual(len(matched), 1)
        self.assertTrue(missing.empty)
        self.assertTrue(extra.empty)
        self.assertEqual(mismatches.iloc[0]["status"], "PNL_MISMATCH")

    def test_local_stale_open_and_ibkr_orphan_detection(self) -> None:
        ibkr = pd.DataFrame([
            {"symbol": "ORPH", "quantity": 4, "average_cost": 10.0, "market_price": 10.5, "unrealized_pnl": 2.0}
        ])
        sqlite = pd.DataFrame([
            {"symbol": "STALE", "quantity": 3, "ibkr_quantity": 3, "avg_price": 11.0, "status": "OPEN", "source": "live_buy"}
        ])

        mismatches = compare_positions(ibkr, sqlite)

        statuses = {row["symbol"]: row["status"] for row in mismatches.to_dict("records")}
        self.assertEqual(statuses["STALE"], "LOCAL_STALE_OPEN")
        self.assertEqual(statuses["ORPH"], "IBKR_ORPHAN_POSITION")

    def test_sqlite_execution_selected_date_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_execution({
                    "execution_id": "TODAY",
                    "session_date": "2026-06-15",
                    "symbol": "AKTX",
                    "side": "BOT",
                    "quantity": 1,
                    "price": 10,
                    "executed_at": "2026-06-15T13:30:00+00:00",
                })
                store.upsert_execution({
                    "execution_id": "YDAY",
                    "session_date": "2026-06-14",
                    "symbol": "AKTX",
                    "side": "BOT",
                    "quantity": 1,
                    "price": 9,
                    "executed_at": "2026-06-14T13:30:00+00:00",
                })
            finally:
                store.close()

            rows = load_sqlite_executions(db, "2026-06-15")

        self.assertEqual(rows["execution_id"].tolist(), ["TODAY"])


if __name__ == "__main__":
    unittest.main()
