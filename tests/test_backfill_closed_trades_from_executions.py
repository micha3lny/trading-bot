from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backfill_closed_trades_from_executions import reconstruct_closed_trades_from_executions
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


def execution_row(
    execution_id: str,
    *,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    executed_at: str,
    commission: float | None = None,
    strategy_name: str = "v67",
) -> dict:
    return {
        "execution_id": execution_id,
        "strategy_name": strategy_name,
        "session_date": executed_at[:10],
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "executed_at": executed_at,
        "recorded_at": executed_at,
        "commission": commission,
        "commission_currency": "USD" if commission is not None else None,
        "commission_source": "ibkr" if commission is not None else "missing",
        "raw_json": {
            "execution": {
                "execId": execution_id,
                "side": side,
                "shares": quantity,
                "price": price,
                "time": executed_at,
            }
        },
    }


class ClosedTradeExecutionBackfillTests(unittest.TestCase):
    def make_store(self, tmp: str) -> SQLiteRuntimeStore:
        return SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")

    def insert_execution_direct(self, store: SQLiteRuntimeStore, row: dict) -> None:
        store.conn.execute(
            """
            INSERT INTO executions (
                execution_id, strategy_name, session_date, symbol, side, quantity, price,
                executed_at, recorded_at, commission, commission_currency, commission_source, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("execution_id"),
                row.get("strategy_name"),
                row.get("session_date"),
                row.get("symbol"),
                row.get("side"),
                row.get("quantity"),
                row.get("price"),
                row.get("executed_at"),
                row.get("recorded_at"),
                row.get("commission"),
                row.get("commission_currency"),
                row.get("commission_source"),
                json.dumps(row.get("raw_json")),
            ),
        )
        store.conn.commit()

    def test_same_day_buy_sell_creates_one_closed_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(tmp)
            try:
                self.insert_execution_direct(
                    store,
                    execution_row("B1", symbol="AKTX", side="BOT", quantity=5, price=17.0, executed_at="2026-05-28T13:35:00+00:00", commission=0.86)
                )
                self.insert_execution_direct(
                    store,
                    execution_row("S1", symbol="AKTX", side="SLD", quantity=5, price=17.5, executed_at="2026-05-28T14:05:00+00:00", commission=0.88)
                )

                dry_run = reconstruct_closed_trades_from_executions(store, "2026-05-28", apply=False)
                self.assertEqual(dry_run["planned"], 1)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM trades")[0]["n"], 0)

                result = reconstruct_closed_trades_from_executions(store, "2026-05-28", apply=True)
                rows = store.query("SELECT * FROM trades")

                self.assertEqual(result["created"], 1)
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["trade_id"], "reconstructed:2026-05-28:2026-05-28:AKTX:B1:S1")
                self.assertEqual(row["session_date"], "2026-05-28")
                self.assertEqual(row["status"], "CLOSED")
                self.assertEqual(row["entry_fill_time"], "2026-05-28T13:35:00+00:00")
                self.assertEqual(row["exit_fill_time"], "2026-05-28T14:05:00+00:00")
                self.assertAlmostEqual(row["gross_pnl"], 2.5)
                self.assertAlmostEqual(row["commission"], 1.74)
                self.assertAlmostEqual(row["net_pnl"], 0.76)
                self.assertEqual(row["ibkr_entry_confirmed"], 1)
                self.assertEqual(row["ibkr_exit_confirmed"], 1)
                self.assertEqual(row["ibkr_position_flat_confirmed"], 1)
            finally:
                store.close()

    def test_previous_day_buy_selected_day_sell_creates_carried_closed_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(tmp)
            try:
                self.insert_execution_direct(
                    store,
                    execution_row("B_DUOT", symbol="DUOT", side="BOT", quantity=2, price=6.0, executed_at="2026-05-27T18:00:00+00:00", commission=0.42)
                )
                self.insert_execution_direct(
                    store,
                    execution_row("S_DUOT", symbol="DUOT", side="SLD", quantity=2, price=7.0, executed_at="2026-05-28T13:40:00+00:00", commission=0.44)
                )

                result = reconstruct_closed_trades_from_executions(store, "2026-05-28", apply=True)
                rows = store.query("SELECT * FROM trades")

                self.assertEqual(result["created"], 1)
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["trade_id"], "reconstructed:2026-05-27:2026-05-28:DUOT:B_DUOT:S_DUOT")
                self.assertEqual(row["session_date"], "2026-05-27")
                self.assertEqual(row["entry_fill_time"], "2026-05-27T18:00:00+00:00")
                self.assertEqual(row["exit_fill_time"], "2026-05-28T13:40:00+00:00")
                self.assertAlmostEqual(row["gross_pnl"], 2.0)
                self.assertAlmostEqual(row["commission"], 0.86)
            finally:
                store.close()

    def test_sell_only_without_recoverable_buy_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(tmp)
            try:
                self.insert_execution_direct(
                    store,
                    execution_row("S_ONLY", symbol="EOSE", side="SLD", quantity=4, price=3.2, executed_at="2026-05-28T13:45:00+00:00", commission=0.5)
                )

                result = reconstruct_closed_trades_from_executions(store, "2026-05-28", apply=True)

                self.assertEqual(result["created"], 0)
                self.assertEqual(result["planned"], 0)
                self.assertEqual(result["skipped_sell_only"], 1)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM trades")[0]["n"], 0)
            finally:
                store.close()

    def test_rerun_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(tmp)
            try:
                self.insert_execution_direct(
                    store,
                    execution_row("B1", symbol="MRAM", side="BOT", quantity=3, price=31.65, executed_at="2026-05-28T13:35:00+00:00", commission=0.91)
                )
                self.insert_execution_direct(
                    store,
                    execution_row("S1", symbol="MRAM", side="SLD", quantity=3, price=28.99, executed_at="2026-05-28T20:22:02+00:00", commission=0.92)
                )

                first = reconstruct_closed_trades_from_executions(store, "2026-05-28", apply=True)
                second = reconstruct_closed_trades_from_executions(store, "2026-05-28", apply=True)

                self.assertEqual(first["created"], 1)
                self.assertEqual(second["created"], 1)
                self.assertEqual(second["deleted_reconstructed"], 1)
                self.assertEqual(store.query("SELECT COUNT(*) AS n FROM trades")[0]["n"], 1)
            finally:
                store.close()

    def test_one_buy_two_sells_consumes_total_quantity_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(tmp)
            try:
                self.insert_execution_direct(store, execution_row("B100", symbol="FIFO", side="BOT", quantity=100, price=10, executed_at="2026-05-28T13:30:00+00:00"))
                self.insert_execution_direct(store, execution_row("S40", symbol="FIFO", side="SLD", quantity=40, price=11, executed_at="2026-05-29T13:40:00+00:00"))
                self.insert_execution_direct(store, execution_row("S60", symbol="FIFO", side="SLD", quantity=60, price=12, executed_at="2026-05-29T13:50:00+00:00"))

                result = reconstruct_closed_trades_from_executions(store, "2026-05-29", apply=True)
                rows = store.query("SELECT quantity, gross_pnl, raw_json FROM trades WHERE symbol = 'FIFO' ORDER BY exit_fill_time")

                self.assertEqual(result["created"], 2)
                self.assertEqual(len(rows), 2)
                self.assertAlmostEqual(sum(row["quantity"] for row in rows), 100)
                self.assertEqual(json.loads(rows[0]["raw_json"])["sell_execution_id"], "S40")
                self.assertEqual(json.loads(rows[1]["raw_json"])["sell_execution_id"], "S60")
            finally:
                store.close()

    def test_two_buys_one_sell_consumes_lots_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(tmp)
            try:
                self.insert_execution_direct(store, execution_row("B15", symbol="LOTX", side="BOT", quantity=15, price=10, executed_at="2026-05-28T13:30:00+00:00"))
                self.insert_execution_direct(store, execution_row("B101", symbol="LOTX", side="BOT", quantity=101, price=11, executed_at="2026-05-28T13:35:00+00:00"))
                self.insert_execution_direct(store, execution_row("S116", symbol="LOTX", side="SLD", quantity=116, price=12, executed_at="2026-05-29T13:40:00+00:00"))

                result = reconstruct_closed_trades_from_executions(store, "2026-05-29", apply=True)
                rows = store.query("SELECT quantity, gross_pnl, raw_json FROM trades WHERE symbol = 'LOTX' ORDER BY entry_fill_time")

                self.assertEqual(result["created"], 2)
                self.assertEqual(len(rows), 2)
                self.assertAlmostEqual(rows[0]["quantity"], 15)
                self.assertAlmostEqual(rows[1]["quantity"], 101)
                self.assertEqual(json.loads(rows[0]["raw_json"])["buy_execution_id"], "B15")
                self.assertEqual(json.loads(rows[1]["raw_json"])["buy_execution_id"], "B101")
            finally:
                store.close()

    def test_old_reconstructed_rows_for_date_are_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(tmp)
            try:
                self.insert_execution_direct(store, execution_row("B1", symbol="MRAM", side="BOT", quantity=3, price=31.65, executed_at="2026-05-28T13:35:00+00:00"))
                self.insert_execution_direct(store, execution_row("S1", symbol="MRAM", side="SLD", quantity=3, price=28.99, executed_at="2026-05-29T20:22:02+00:00"))
                store.upsert_trade({
                    "trade_id": "reconstructed:old:bad:MRAM:B_OLD:S1",
                    "strategy_name": "v67",
                    "session_date": "2026-05-27",
                    "symbol": "MRAM",
                    "status": "CLOSED",
                    "entry_fill_time": "2026-05-27T13:35:00+00:00",
                    "exit_fill_time": "2026-05-29T20:22:02+00:00",
                    "closed_at": "2026-05-29T20:22:02+00:00",
                    "entry_price": 40,
                    "exit_price": 28.99,
                    "quantity": 3,
                    "raw_json": {"reconstruction_source": "executions_pair_repair", "buy_execution_id": "B_OLD", "sell_execution_id": "S1", "exit_date": "2026-05-29"},
                })

                result = reconstruct_closed_trades_from_executions(store, "2026-05-29", apply=True)
                rows = store.query("SELECT trade_id, entry_fill_time FROM trades WHERE symbol = 'MRAM'")

                self.assertEqual(result["deleted_reconstructed"], 1)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["trade_id"], "reconstructed:2026-05-28:2026-05-29:MRAM:B1:S1")
                self.assertEqual(rows[0]["entry_fill_time"], "2026-05-28T13:35:00+00:00")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
