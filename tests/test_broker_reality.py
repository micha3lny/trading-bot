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
    closed_trades_from_commission_reports,
    load_sqlite_active_positions,
    load_sqlite_executions,
    load_sqlite_closed_trades,
    load_sqlite_trade_pnl,
    match_executions,
    normalize_execution_record,
    parse_ibkr_activity_csv,
    reconcile_broker_vs_sqlite,
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

    def test_normalize_execution_keeps_execution_client_id(self) -> None:
        row = normalize_execution_record(
            {
                "execution_id": "E1",
                "symbol": "AKTX",
                "side": "BUY",
                "quantity": 5,
                "price": 17.01,
                "commission": 0.86,
                "execution_time": "2026-06-15 13:35:39",
                "clientId": 67,
            },
            source="ib.fills()",
        )

        self.assertEqual(row["execution_client_id"], "67")
        self.assertEqual(row["source"], "ib.fills()")

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

    def test_broker_closed_trades_use_commission_report_realized_pnl(self) -> None:
        executions = pd.DataFrame([
            {
                "execution_time": "2026-06-17T13:30:00+00:00",
                "symbol": "AKTX",
                "side": "BUY",
                "quantity": 5,
                "price": 10.0,
                "commission": 0.5,
                "realized_pnl": 0.0,
                "execution_id": "B1",
            },
            {
                "execution_time": "2026-06-17T14:00:00+00:00",
                "symbol": "AKTX",
                "side": "SELL",
                "quantity": 5,
                "price": 11.0,
                "commission": 0.6,
                "realized_pnl": -2.25,
                "execution_id": "S1",
            },
        ])

        fifo = reconstruct_closed_trades_fifo(executions, "2026-06-17")
        broker_truth = closed_trades_from_commission_reports(executions, "2026-06-17")

        self.assertAlmostEqual(fifo.iloc[0]["realized_pnl"], 5.0)
        self.assertEqual(broker_truth.iloc[0]["source"], "IBKR_COMMISSION_REPORT_REALIZED_PNL")
        self.assertAlmostEqual(broker_truth.iloc[0]["realized_pnl"], -2.25)
        self.assertAlmostEqual(broker_truth.iloc[0]["commission"], 0.6)
        self.assertAlmostEqual(broker_truth.iloc[0]["net_pnl"], -2.85)

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
        self.assertIn("broker_trade_id", mismatches.columns)
        self.assertIn("sqlite_trade_id", mismatches.columns)
        self.assertIn("net_difference", mismatches.columns)
        self.assertAlmostEqual(mismatches.iloc[0]["net_difference"], 0.5)

    def test_carry_lot_grouping_matches_split_sqlite_trades(self) -> None:
        broker = pd.DataFrame([
            {
                "symbol": "VECO",
                "trade_id": "broker:VECO",
                "entry_time": "2026-06-05T13:30:00+00:00",
                "exit_time": "2026-06-15T19:55:00+00:00",
                "quantity": 3,
                "entry_price": 10.0,
                "exit_price": 12.0,
                "realized_pnl": 6.0,
                "commission": 1.2,
                "net_pnl": 4.8,
            }
        ])
        sqlite = pd.DataFrame([
            {
                "symbol": "VECO",
                "trade_id": "reconstructed:VECO:1",
                "entry_time": "2026-06-05T13:30:00+00:00",
                "exit_time": "2026-06-15T19:55:00+00:00",
                "quantity": 1,
                "entry_price": 10.0,
                "exit_price": 12.0,
                "realized_pnl": 2.0,
                "commission": 0.4,
                "net_pnl": 1.6,
            },
            {
                "symbol": "VECO",
                "trade_id": "reconstructed:VECO:2",
                "entry_time": "2026-06-05T13:31:00+00:00",
                "exit_time": "2026-06-15T19:55:00+00:00",
                "quantity": 2,
                "entry_price": 10.0,
                "exit_price": 12.0,
                "realized_pnl": 4.0,
                "commission": 0.8,
                "net_pnl": 3.2,
            },
        ])

        matched, missing, extra, mismatches = compare_closed_trades(broker, sqlite)

        self.assertEqual(len(matched), 1)
        self.assertTrue(missing.empty)
        self.assertTrue(extra.empty)
        self.assertTrue(mismatches.empty)
        self.assertAlmostEqual(matched.iloc[0]["sqlite_qty"], 3)
        self.assertAlmostEqual(matched.iloc[0]["sqlite_net"], 4.8)

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
        self.assertIn("broker_qty", mismatches.columns)
        self.assertIn("sqlite_qty", mismatches.columns)
        self.assertIn("qty_difference", mismatches.columns)
        self.assertIn("cost_difference", mismatches.columns)

    def test_sqlite_active_positions_prefers_latest_active_not_latest_closed_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-18:KOD",
                    "strategy_name": "v67",
                    "session_date": "2026-06-18",
                    "symbol": "KOD",
                    "status": "OPEN",
                    "quantity": 11,
                    "avg_price": 9.5,
                    "active": 1,
                    "updated_at": "2026-06-18T13:30:00+00:00",
                })
                store.upsert_position({
                    "position_key": "v67:2026-06-18:KOD:closed-shadow",
                    "strategy_name": "v67",
                    "session_date": "2026-06-18",
                    "symbol": "KOD",
                    "status": "CLOSED",
                    "quantity": 0,
                    "avg_price": 9.6,
                    "active": 0,
                    "updated_at": "2026-06-18T13:31:00+00:00",
                })
            finally:
                store.close()

            rows = load_sqlite_active_positions(db, "2026-06-18")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["symbol"], "KOD")
        self.assertEqual(rows.iloc[0]["quantity"], 11)

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

    def test_sqlite_closed_trades_use_execution_realized_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_trade({
                    "trade_id": "reconstructed:2026-06-16:2026-06-16:RXT:B1:S1",
                    "session_date": "2026-06-16",
                    "strategy_name": "unknown",
                    "symbol": "RXT",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-06-16T13:35:00+00:00",
                    "exit_fill_time": "2026-06-16T14:10:00+00:00",
                    "entry_price": 1.25,
                    "exit_price": 1.30,
                    "quantity": 58,
                    "gross_pnl": 2.9,
                    "commission": 0.7,
                    "net_pnl": 2.2,
                    "raw_json": {
                        "reconstruction_source": "sqlite_execution_reducer",
                        "buy_execution_id": "B1",
                        "sell_execution_id": "S1",
                    },
                })
                store.upsert_execution({
                    "execution_id": "B1",
                    "session_date": "2026-06-16",
                    "symbol": "RXT",
                    "side": "BOT",
                    "quantity": 58,
                    "price": 1.25,
                    "commission": 0.4,
                    "commission_source": "ibkr",
                    "realized_pnl": 0.0,
                    "executed_at": "2026-06-16T13:35:00+00:00",
                })
                store.upsert_execution({
                    "execution_id": "S1",
                    "session_date": "2026-06-16",
                    "symbol": "RXT",
                    "side": "SLD",
                    "quantity": 58,
                    "price": 1.30,
                    "commission": 0.7,
                    "commission_source": "ibkr",
                    "realized_pnl": 2.9,
                    "executed_at": "2026-06-16T14:10:00+00:00",
                })
            finally:
                store.close()

            rows = load_sqlite_closed_trades(db, "2026-06-16")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["entry_execution_id"], "B1")
        self.assertEqual(rows.iloc[0]["exit_execution_id"], "S1")
        self.assertEqual(rows.iloc[0]["source"], "SQLITE_EXECUTIONS_REALIZED_PNL")
        self.assertAlmostEqual(rows.iloc[0]["realized_pnl"], 2.9)
        self.assertAlmostEqual(rows.iloc[0]["commission"], 0.7)
        self.assertAlmostEqual(rows.iloc[0]["net_pnl"], 2.2)

    def test_sqlite_trade_pnl_uses_runtime_trusted_closed_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_trade({
                    "trade_id": "reconstructed:2026-05-13:2026-06-16:OUST:B_OLD:S_TODAY",
                    "session_date": "2026-05-13",
                    "strategy_name": "v67",
                    "symbol": "OUST",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-05-13T13:30:00+00:00",
                    "exit_fill_time": "2026-06-16T14:45:00+00:00",
                    "closed_at": "2026-06-16T14:45:00+00:00",
                    "entry_price": 1.00,
                    "exit_price": 12.487727,
                    "quantity": 22,
                    "gross_pnl": 252.73,
                    "commission": 1.025094,
                    "net_pnl": 251.704906,
                    "raw_json": {
                        "reconstruction_source": "sqlite_execution_reducer",
                        "buy_execution_id": "B_OLD",
                        "sell_execution_id": "S_TODAY",
                    },
                })
            finally:
                store.close()

            pnl = load_sqlite_trade_pnl(db, "2026-06-16")
            closed = load_sqlite_closed_trades(db, "2026-06-16")

        self.assertEqual(len(closed), 0)
        row = pnl.iloc[0]
        self.assertEqual(row["reconciliation_sqlite_trade_source"], "executions_realized_pnl_minus_sell_commission")
        self.assertEqual(row["runtime_trade_source"], "sqlite_executions")
        self.assertEqual(row["trades"], 0)
        self.assertEqual(row["closed_symbols"], 0)
        self.assertAlmostEqual(row["sqlite_gross"], 0.0)
        self.assertAlmostEqual(row["sqlite_net"], 0.0)

    def test_sqlite_trade_pnl_uses_executions_not_inflated_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "runtime.sqlite"
            store = SQLiteRuntimeStore(db)
            try:
                store.upsert_trade({
                    "trade_id": "inflated:OUST",
                    "session_date": "2026-06-18",
                    "strategy_name": "v67",
                    "symbol": "OUST",
                    "status": "CLOSED",
                    "exit_fill_time": "2026-06-18T14:45:00+00:00",
                    "entry_price": 1.0,
                    "exit_price": 12.0,
                    "quantity": 22,
                    "gross_pnl": 252.73,
                    "commission": 1.0,
                    "net_pnl": 251.73,
                })
                store.upsert_execution({
                    "execution_id": "B_REAL",
                    "session_date": "2026-06-18",
                    "strategy_name": "v67",
                    "symbol": "OUST",
                    "side": "BOT",
                    "quantity": 22,
                    "price": 10,
                    "commission": 1.0,
                    "commission_source": "ibkr",
                    "realized_pnl": 0.0,
                    "executed_at": "2026-06-18T13:30:00+00:00",
                })
                store.upsert_execution({
                    "execution_id": "S_REAL",
                    "session_date": "2026-06-18",
                    "strategy_name": "v67",
                    "symbol": "OUST",
                    "side": "SLD",
                    "quantity": 22,
                    "price": 10.1,
                    "commission": 1.5,
                    "commission_source": "ibkr",
                    "realized_pnl": 5.06,
                    "executed_at": "2026-06-18T14:45:00+00:00",
                })
            finally:
                store.close()

            pnl = load_sqlite_trade_pnl(db, "2026-06-18").iloc[0]

        self.assertEqual(pnl["reconciliation_sqlite_trade_source"], "executions_realized_pnl_minus_sell_commission")
        self.assertEqual(pnl["realized_pnl_semantics"], "gross_before_commission")
        self.assertEqual(pnl["net_formula"], "sum_sell_realized_pnl_minus_sell_commission")
        self.assertEqual(pnl["closed_symbols"], 1)
        self.assertAlmostEqual(pnl["sqlite_gross"], 5.06)
        self.assertAlmostEqual(pnl["sqlite_commission"], 1.5)
        self.assertAlmostEqual(pnl["sqlite_net"], 3.56)

    def test_reconciliation_status_uses_execution_closed_truth_not_trade_rows(self) -> None:
        executions = pd.DataFrame([
            {
                "execution_time": "2026-06-23T13:30:00+00:00",
                "symbol": "ARQQ",
                "side": "BUY",
                "quantity": 10,
                "price": 10.0,
                "commission": 0.4,
                "realized_pnl": 0.0,
                "execution_id": "B_ARQQ",
            },
            {
                "execution_time": "2026-06-23T14:00:00+00:00",
                "symbol": "ARQQ",
                "side": "SELL",
                "quantity": 10,
                "price": 11.0,
                "commission": 0.6,
                "realized_pnl": 10.0,
                "execution_id": "S_ARQQ",
            },
        ])
        broker_closed = closed_trades_from_commission_reports(executions, "2026-06-23")
        sqlite_closed = broker_closed.copy()
        sqlite_closed["source"] = "SQLITE_EXECUTIONS_REALIZED_PNL"
        sqlite_closed["trade_id"] = sqlite_closed["trade_id"].astype(str).str.replace("ibkr_realized:", "sqlite_realized:", regex=False)
        sqlite_pnl = pd.DataFrame([
            {
                "trades": 1,
                "closed_symbols": 1,
                "sqlite_gross": 10.0,
                "sqlite_commission": 0.6,
                "sqlite_net": 9.4,
                "reconciliation_sqlite_trade_source": "executions_realized_pnl_minus_sell_commission",
                "runtime_trade_source": "sqlite_executions",
                "trusted_closed_count": 29,
                "untrusted_carry_count": 0,
            }
        ])

        result = reconcile_broker_vs_sqlite(
            executions,
            executions.copy(),
            pd.DataFrame(),
            pd.DataFrame(),
            broker_closed,
            sqlite_closed,
            sqlite_pnl,
            selected_date="2026-06-23",
            broker_status="OK",
        )

        self.assertEqual(result.summary["status"], "IBKR_RECONCILED")
        self.assertEqual(result.summary["broker_closed_trades"], 1)
        self.assertEqual(result.summary["sqlite_closed_trades"], 1)
        self.assertEqual(result.summary["reconciliation_sqlite_trade_source"], "executions_realized_pnl_minus_sell_commission")


if __name__ == "__main__":
    unittest.main()
