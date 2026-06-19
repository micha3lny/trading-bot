from __future__ import annotations

import contextlib
import tempfile
import unittest
import csv
import io
from pathlib import Path
from types import SimpleNamespace

from src.live_trading.order_lifecycle.models import LifecycleEvent, LifecycleEventType, OrderSide, PositionState
from src.live_trading.order_lifecycle.store import JsonlLifecycleStore
from src.live_trading.storage.sqlite_store import SQLiteRuntimeStore
from src.live_trading.v62_live_data_recorder import LiveDataRecorder
from src.live_trading.v67_live_top100_expansion_paper_trader import (
    ManagedPosition,
    restore_managed_positions,
    send_exit_order,
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
    def test_restore_managed_positions_rejects_candidates_when_broker_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            recorder.path("managed_positions.json").write_text(
                """
                {
                  "positions": {
                    "RKLB": {
                      "quantity": 10,
                      "entry_price": 12.0,
                      "peak_price": 12.5,
                      "active": true,
                      "entry_fill_verified": true
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            runtime_state = {}

            restored = restore_managed_positions(
                recorder,
                {"RKLB": FakeContract("RKLB")},
                broker_qty_by_symbol={},
                runtime_state=runtime_state,
            )

            self.assertEqual(restored, {})
            self.assertEqual(runtime_state["startup_restore_broker_snapshot_count"], 0)
            self.assertEqual(runtime_state["startup_restore_candidate_count"], 1)
            self.assertEqual(runtime_state["startup_restore_open_count"], 0)
            self.assertEqual(runtime_state["startup_restore_rejected_count"], 1)

    def test_restore_managed_positions_uses_broker_quantity_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            recorder.path("managed_positions.json").write_text(
                """
                {
                  "positions": {
                    "RKLB": {
                      "quantity": 10,
                      "entry_price": 12.0,
                      "peak_price": 12.5,
                      "active": true,
                      "entry_fill_verified": true
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            restored = restore_managed_positions(
                recorder,
                {"RKLB": FakeContract("RKLB")},
                broker_qty_by_symbol={"RKLB": 7},
                runtime_state={},
            )

            self.assertEqual(restored["RKLB"].quantity, 7)

    def test_restore_managed_positions_disabled_does_not_create_active_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            recorder.path("managed_positions.json").write_text(
                """
                {
                  "positions": {
                    "RKLB": {
                      "quantity": 10,
                      "entry_price": 12.0,
                      "peak_price": 12.5,
                      "active": true,
                      "entry_fill_verified": true
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            runtime_state = {}

            restored = restore_managed_positions(
                recorder,
                {"RKLB": FakeContract("RKLB")},
                broker_qty_by_symbol={"RKLB": 7},
                runtime_state=runtime_state,
                restore_enabled=False,
                disabled_reason="sqlite_broker_source_of_truth",
            )

            self.assertEqual(restored, {})
            self.assertEqual(runtime_state["startup_restore_candidate_count"], 1)
            self.assertEqual(runtime_state["startup_restore_open_count"], 0)
            self.assertEqual(runtime_state["startup_restore_rejected_count"], 1)
            self.assertFalse(runtime_state["startup_restore_enabled"])
            self.assertEqual(runtime_state["startup_restore_disabled_reason"], "sqlite_broker_source_of_truth")

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
            self.assertFalse(any(row["event_type"] == "POSITION_CLOSED" for row in events))
            self.assertTrue(any(row["event_type"] == LifecycleEventType.POSITION_DRIFT_DETECTED.value for row in events))
            lifecycle_text = recorder.path("trade_lifecycle.csv").read_text(encoding="utf-8")
            self.assertIn("ENTRY_NOT_FILLED", lifecycle_text)
            self.assertIn("entry_fill_verified", lifecycle_text)
            self.assertIn("false", lifecycle_text)

    def test_broker_sqlite_flat_ignores_stale_lifecycle_open_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            recorder.sqlite_store = store
            try:
                JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).append_event(
                    LifecycleEvent(
                        event_type=LifecycleEventType.ENTRY_ORDER_FILLED,
                        symbol="AVLN",
                        strategy="v67",
                        state_after=PositionState.OPEN,
                        execution_id="OLD_BUY",
                        quantity=3,
                        price=10.0,
                        raw_json={"side": OrderSide.BUY.value},
                    )
                )
                runtime_state = {}
                out = io.StringIO()

                with contextlib.redirect_stdout(out):
                    result = startup_reconcile_runtime_state(FakeIB(), recorder, {}, {"AVLN": FakeContract("AVLN")}, runtime_state)

                self.assertEqual(result["closed_local"], [])
                self.assertIn("STARTUP_RECONCILIATION_STALE_LOCAL_STATE_IGNORED", out.getvalue())
                self.assertEqual(runtime_state["startup_reconciliation_broker_open_count"], 0)
                self.assertEqual(runtime_state["startup_reconciliation_sqlite_active_count"], 0)
            finally:
                store.close()

    def test_broker_position_matching_sqlite_active_is_not_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            store = SQLiteRuntimeStore(Path(tmp) / "runtime.sqlite")
            recorder.sqlite_store = store
            try:
                store.upsert_position({
                    "position_key": "v67:2026-06-19:AVLN",
                    "strategy_name": "v67",
                    "session_date": "2026-06-19",
                    "symbol": "AVLN",
                    "status": "OPEN",
                    "quantity": 3,
                    "avg_price": 10.0,
                    "active": 1,
                    "updated_at": "2026-06-19T13:30:00+00:00",
                })
                ib = FakeIB(portfolio=[FakePortfolioItem("AVLN", 3)])
                runtime_state = {}

                result = startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state)

                self.assertEqual(len(ib.placed_orders), 0)
                self.assertEqual(result["orphans"], [])
                self.assertEqual(runtime_state["startup_reconciliation_sqlite_active_count"], 1)
                self.assertEqual(runtime_state["startup_reconciliation_sqlite_active_symbols"], ["AVLN"])
            finally:
                store.close()

    def test_ibkr_flat_after_verified_entry_without_sell_fill_is_unverified_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            fills_path = recorder.path("fills.csv")
            with fills_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["execution_id", "symbol", "action", "quantity", "fill_price"])
                writer.writeheader()
                writer.writerow({"execution_id": "B1", "symbol": "RKLB", "action": "BOT", "quantity": "10", "fill_price": "12"})
            contract = FakeContract("RKLB")
            managed = {
                "RKLB": ManagedPosition("RKLB", contract, 10, 12.0, "restored", 12.5, entry_fill_verified=True, exit_sent=True),
            }
            runtime_state = {}

            result = startup_reconcile_runtime_state(FakeIB(), recorder, managed, {"RKLB": contract}, runtime_state)

            self.assertFalse(managed["RKLB"].active)
            self.assertIn("RKLB", result["closed_local"])
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            self.assertFalse(any(row["event_type"] == LifecycleEventType.POSITION_CLOSED.value for row in events))
            lifecycle_text = recorder.path("trade_lifecycle.csv").read_text(encoding="utf-8")
            self.assertIn("RECONCILIATION_CLOSE_WITHOUT_FILL", lifecycle_text)
            self.assertIn("POSITION_CLOSED_UNVERIFIED", lifecycle_text)
            self.assertIn("close_fill_verified", lifecycle_text)
            self.assertIn("false", lifecycle_text)

    def test_exit_order_blocked_until_entry_fill_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            contract = FakeContract("RKLB")
            pos = ManagedPosition("RKLB", contract, 10, 12.0, "restored", 12.5, entry_fill_verified=False)
            ib = FakeIB()

            sent = send_exit_order(ib, recorder, pos, "unit_test_exit", 11.5)

            self.assertFalse(sent)
            self.assertEqual(len(ib.placed_orders), 0)
            lifecycle_text = recorder.path("trade_lifecycle.csv").read_text(encoding="utf-8")
            self.assertIn("EXIT_ORDER_BLOCKED_NO_ENTRY_FILL", lifecycle_text)
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            self.assertFalse(any(row["event_type"] == LifecycleEventType.EXIT_ORDER_SUBMITTED.value for row in events))

    def test_ibkr_orphan_whole_share_left_untouched_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            ib = FakeIB(portfolio=[FakePortfolioItem("AVLN", 3)])
            runtime_state = {}

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state)

            self.assertEqual(len(ib.placed_orders), 0)
            self.assertFalse(runtime_state["entries_blocked"])
            self.assertIn("AVLN", result["orphans"])
            self.assertIn("AVLN", result["untouched_orphans"])
            self.assertIn("STARTUP_RECONCILIATION_ORPHAN_LEFT_UNTOUCHED", out.getvalue())
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            self.assertFalse(any(row["event_type"] == "EXIT_ORDER_SUBMITTED" for row in events))

    def test_ibkr_orphan_whole_share_flatten_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            ib = FakeIB(portfolio=[FakePortfolioItem("AVLN", 3)])
            runtime_state = {}

            result = startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state, submit_orphan_flatten=True)

            self.assertEqual(len(ib.placed_orders), 1)
            self.assertFalse(runtime_state["entries_blocked"])
            self.assertIn("AVLN", result["orphans"])
            self.assertEqual(result["untouched_orphans"], [])
            events = JsonlLifecycleStore(recorder.path("order_lifecycle.jsonl")).load_events()
            self.assertTrue(any(row["event_type"] == "EXIT_ORDER_SUBMITTED" for row in events))

    def test_startup_reconciliation_does_not_cancel_flatten_order_it_just_submitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = recorder_in_tmp(tmp)
            ib = FakeIBWithSubmittedOpenTrades(portfolio=[FakePortfolioItem("AVLN", 3)])
            runtime_state = {}

            result = startup_reconcile_runtime_state(ib, recorder, {}, {}, runtime_state, submit_orphan_flatten=True)

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
            self.assertTrue(any(
                row["event_type"] == LifecycleEventType.POSITION_DRIFT_DETECTED.value
                and row.get("reason") == "startup_reconciliation_fractional_orphan_manual_required"
                for row in events
            ))

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
