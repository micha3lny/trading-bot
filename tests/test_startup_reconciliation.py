from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.live_trading.order_lifecycle.models import LifecycleEventType
from src.live_trading.order_lifecycle.store import JsonlLifecycleStore
from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v67_live_top100_expansion_paper_trader import (
    ManagedPosition,
    startup_reconcile_runtime_state,
)


class FakeContract:
    def __init__(self, symbol: str, currency: str = "USD") -> None:
        self.symbol = symbol
        self.currency = currency
        self.exchange = "SMART"
        self.primaryExchange = "NASDAQ"


class FakePortfolioItem:
    def __init__(self, symbol: str, position: float, market_price: float = 10.0) -> None:
        self.contract = FakeContract(symbol)
        self.position = position
        self.marketPrice = market_price
        self.marketValue = position * market_price
        self.averageCost = market_price


class FakeTrade:
    def __init__(self, symbol: str, order_id: int = 1, action: str = "SELL", quantity: float = 1) -> None:
        self.contract = FakeContract(symbol)
        self.order = SimpleNamespace(orderId=order_id, action=action, totalQuantity=quantity)
        self.orderStatus = SimpleNamespace(status="Submitted")
        self.log = []


class FakeIB:
    def __init__(self, portfolio=None, open_trades=None) -> None:
        self._portfolio = list(portfolio or [])
        self._open_trades = list(open_trades or [])
        self.placed_orders = []
        self.cancelled_orders = []
        self.next_order_id = 100

    def portfolio(self):
        return list(self._portfolio)

    def openTrades(self):
        return list(self._open_trades)

    def openOrders(self):
        return []

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, contract, order):
        self.next_order_id += 1
        order.orderId = self.next_order_id
        trade = SimpleNamespace(contract=contract, order=order, orderStatus=SimpleNamespace(status="Submitted"), log=[])
        self.placed_orders.append(trade)
        return trade

    def cancelOrder(self, order):
        self.cancelled_orders.append(order)

    def sleep(self, seconds):
        return None


class FakeIBWithSubmittedOpenTrades(FakeIB):
    def openTrades(self):
        return list(self._open_trades) + list(self.placed_orders)


def recorder_in_tmp(tmp: str) -> LiveDataRecorder:
    return LiveDataRecorder(Path(tmp), session_date="2026-05-21")


class StartupReconciliationTests(unittest.TestCase):
    def test_local_open_but_ibkr_flat_marks_inactive_and_unblocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            contract = FakeContract("RKLB")
            managed = {
                "RKLB": ManagedPosition("RKLB", contract, 10, 12.0, "restored", 12.5),
            }
            runtime_state = {}

            result = startup_reconcile_runtime_state(FakeIB(), recorder, managed, {"RKLB": contract}, runtime_state)

            self.assertFalse(managed["RKLB"].active)
            self.assertFalse(runtime_state["entries_blocked"])
            self.assertIn("RKLB", result["closed_local"])
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            self.assertTrue(any(row["event_type"] == "POSITION_CLOSED" for row in events))

    def test_ibkr_orphan_whole_share_flattens_and_does_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            ib = FakeIB(portfolio=[FakePortfolioItem("AVLN", 3)])
            runtime_state = {}

            result = startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state)

            self.assertEqual(len(ib.placed_orders), 1)
            self.assertFalse(runtime_state["entries_blocked"])
            self.assertIn("AVLN", result["orphans"])
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            self.assertTrue(any(row["event_type"] == "EXIT_ORDER_SUBMITTED" for row in events))

    def test_startup_reconciliation_does_not_cancel_flatten_order_it_just_submitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            ib = FakeIBWithSubmittedOpenTrades(portfolio=[FakePortfolioItem("AVLN", 3)])
            runtime_state = {}

            result = startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state)

            self.assertEqual(len(ib.placed_orders), 1)
            self.assertEqual(len(ib.cancelled_orders), 0)
            self.assertIn(str(ib.placed_orders[0].order.orderId), result["pending_orders"])
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            event_types = [row["event_type"] for row in events]
            self.assertIn(LifecycleEventType.EXIT_ORDER_SUBMITTED.value, event_types)
            self.assertNotIn(LifecycleEventType.EXIT_ORDER_STALE.value, event_types)
            self.assertNotIn(LifecycleEventType.EXIT_ORDER_CANCEL_REQUESTED.value, event_types)

    def test_ibkr_orphan_fractional_records_manual_action_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            ib = FakeIB(portfolio=[FakePortfolioItem("ASST", 0.2)])
            runtime_state = {}

            result = startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state)

            self.assertEqual(len(ib.placed_orders), 0)
            self.assertFalse(runtime_state["entries_blocked"])
            self.assertIn("ASST", result["orphans"])
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            self.assertTrue(any(row["event_type"] == "EXIT_ORDER_REJECTED" for row in events))

    def test_quantity_drift_updates_managed_qty_when_same_direction_whole_share(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            contract = FakeContract("RKLB")
            managed = {
                "RKLB": ManagedPosition("RKLB", contract, 10, 12.0, "restored", 12.5),
            }
            ib = FakeIB(portfolio=[FakePortfolioItem("RKLB", 7)])
            runtime_state = {}

            result = startup_reconcile_runtime_state(ib, recorder, managed, {"RKLB": contract}, runtime_state)

            self.assertEqual(managed["RKLB"].quantity, 7)
            self.assertFalse(runtime_state["entries_blocked"])
            self.assertIn("RKLB", result["drift_symbols"])
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            self.assertTrue(any(row["event_type"] == "POSITION_DRIFT_DETECTED" for row in events))

    def test_stale_open_order_cancel_attempted_and_lifecycle_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            ib = FakeIB(open_trades=[FakeTrade("RKLB", order_id=55, action="SELL", quantity=10)])
            runtime_state = {}

            result = startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state)

            self.assertEqual(len(ib.cancelled_orders), 1)
            self.assertIn("55", result["pending_orders"])
            self.assertFalse(runtime_state["entries_blocked"])
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            event_types = {row["event_type"] for row in events}
            self.assertIn(LifecycleEventType.EXIT_ORDER_STALE.value, event_types)
            self.assertIn(LifecycleEventType.EXIT_ORDER_CANCEL_REQUESTED.value, event_types)
            self.assertIn(LifecycleEventType.EXIT_ORDER_CANCELLED.value, event_types)


if __name__ == "__main__":
    unittest.main()
