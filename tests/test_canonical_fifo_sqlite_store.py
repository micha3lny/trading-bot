from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore


def add_execution(store: SQLiteRuntimeStore, execution_id: str, side: str, qty: float, price: float, ts: str, **extra) -> None:
    store.upsert_execution(
        {
            "execution_id": execution_id,
            "strategy_name": "v67",
            "session_date": ts[:10],
            "symbol": extra.pop("symbol", "CFIFO"),
            "side": side,
            "quantity": qty,
            "price": price,
            "executed_at": ts,
            "recorded_at": extra.pop("recorded_at", ts),
            "commission": extra.pop("commission", 0.0),
            "commission_source": extra.pop("commission_source", "ibkr"),
            "realized_pnl": extra.pop("realized_pnl", None),
            **extra,
        }
    )


class CanonicalFifoSQLiteStoreTests(unittest.TestCase):
    def test_trade_components_are_idempotent_and_conserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                add_execution(store, "B1", "BOT", 4, 10, "2026-07-15T13:30:00+00:00")
                add_execution(store, "B2", "BOT", 6, 12, "2026-07-15T13:31:00+00:00")
                add_execution(store, "S1", "SLD", 10, 13, "2026-07-15T13:40:00+00:00", realized_pnl=18)

                stable_cols = (
                    "component_id, trade_id, buy_execution_id, sell_execution_id, "
                    "matched_qty, buy_price, sell_price, gross_pnl, net_pnl"
                )
                first = store.query(f"SELECT {stable_cols} FROM trade_components WHERE symbol = 'CFIFO' ORDER BY component_id")
                store.rebuild_symbol_trade_state("CFIFO")
                second = store.query(f"SELECT {stable_cols} FROM trade_components WHERE symbol = 'CFIFO' ORDER BY component_id")
                trades = store.query("SELECT * FROM trades WHERE symbol = 'CFIFO' AND status = 'CLOSED'")

                self.assertEqual(len(trades), 1)
                self.assertEqual(len(first), 2)
                self.assertEqual(first, second)
                self.assertAlmostEqual(sum(row["matched_qty"] for row in second), 10)
            finally:
                store.close()

    def test_stale_closed_buy_trade_id_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                add_execution(store, "B_OLD", "BOT", 8, 20, "2026-07-09T13:30:00+00:00", symbol="STALE")
                add_execution(store, "S_OLD", "SLD", 8, 21, "2026-07-09T13:40:00+00:00", symbol="STALE", realized_pnl=8)
                add_execution(store, "B_NEW", "BOT", 8, 30, "2026-07-15T13:30:00+00:00", symbol="STALE")
                add_execution(store, "S_NEW", "SLD", 8, 31, "2026-07-15T13:40:00+00:00", symbol="STALE", realized_pnl=8)

                trades = store.query("SELECT trade_id, entry_fill_time, exit_fill_time FROM trades WHERE symbol = 'STALE' ORDER BY entry_fill_time")
                components = store.query("SELECT trade_id, buy_execution_id, sell_execution_id FROM trade_components WHERE symbol = 'STALE' ORDER BY entry_time")

                self.assertEqual(len(trades), 2)
                self.assertEqual(components[0]["buy_execution_id"], "B_OLD")
                self.assertEqual(components[0]["sell_execution_id"], "S_OLD")
                self.assertEqual(components[1]["buy_execution_id"], "B_NEW")
                self.assertEqual(components[1]["sell_execution_id"], "S_NEW")
                self.assertNotEqual(components[0]["trade_id"], components[1]["trade_id"])
            finally:
                store.close()

    def test_unmatched_sell_quantity_is_reported_without_fake_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            try:
                add_execution(store, "B1", "BOT", 5, 10, "2026-07-15T13:30:00+00:00", symbol="OVERSELL")
                add_execution(store, "S1", "SLD", 8, 11, "2026-07-15T13:40:00+00:00", symbol="OVERSELL", realized_pnl=8)

                result = store.rebuild_symbol_trade_state("OVERSELL")
                trades = store.query("SELECT * FROM trades WHERE symbol = 'OVERSELL' AND status = 'CLOSED'")
                components = store.query("SELECT * FROM trade_components WHERE symbol = 'OVERSELL'")

                self.assertEqual(len(trades), 1)
                self.assertEqual(len(components), 1)
                self.assertAlmostEqual(trades[0]["quantity"], 5)
                self.assertAlmostEqual(result["unmatched_sell_quantity"], 3)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
