from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.live_trading.order_lifecycle.models import (
    ExecutionRecord,
    LifecycleEvent,
    LifecycleEventType,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderState,
    PositionRecord,
    PositionState,
)
from src.live_trading.order_lifecycle.state_machine import apply_execution, apply_order_event, apply_position_transition
from src.live_trading.order_lifecycle.store import JsonlLifecycleStore


def entry_order(quantity: float = 10) -> OrderRecord:
    return OrderRecord(
        client_order_id="v67:2026-05-18:RKLB:BUY:ENTRY:001",
        symbol="RKLB",
        side=OrderSide.BUY,
        purpose=OrderPurpose.ENTRY,
        quantity=quantity,
        state=OrderState.PREPARED,
    )


def exit_order(quantity: float = 10) -> OrderRecord:
    return OrderRecord(
        client_order_id="v67:2026-05-18:RKLB:SELL:EOD_FLATTEN:001",
        symbol="RKLB",
        side=OrderSide.SELL,
        purpose=OrderPurpose.EOD_FLATTEN,
        quantity=quantity,
        state=OrderState.PREPARED,
    )


def position() -> PositionRecord:
    return PositionRecord(symbol="RKLB", strategy="v67", session_date="2026-05-18", target_quantity=10)


def execution(execution_id: str, side: OrderSide, quantity: float, price: float) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=execution_id,
        client_order_id="",
        symbol="RKLB",
        side=side,
        quantity=quantity,
        price=price,
    )


class OrderLifecycleStateMachineTests(unittest.TestCase):
    def test_submit_does_not_open_position_until_fill(self) -> None:
        order = apply_order_event(entry_order(), OrderState.SUBMITTED)
        pos = apply_position_transition(position(), PositionState.ENTRY_PENDING)
        self.assertEqual(order.state, OrderState.SUBMITTED)
        self.assertEqual(pos.state, PositionState.ENTRY_PENDING)
        self.assertEqual(pos.open_quantity, 0)

        pos, order, applied = apply_execution(pos, order, execution("E1", OrderSide.BUY, 10, 12.5))
        self.assertTrue(applied)
        self.assertEqual(order.state, OrderState.FILLED)
        self.assertEqual(pos.state, PositionState.OPEN)
        self.assertEqual(pos.open_quantity, 10)

    def test_submit_to_partial(self) -> None:
        order = apply_order_event(entry_order(), OrderState.SUBMITTED)
        pos = apply_position_transition(position(), PositionState.ENTRY_PENDING)

        pos, order, applied = apply_execution(pos, order, execution("E1", OrderSide.BUY, 4, 12.5))
        self.assertTrue(applied)
        self.assertEqual(order.state, OrderState.PARTIAL)
        self.assertEqual(pos.state, PositionState.OPEN)
        self.assertEqual(pos.open_quantity, 4)

    def test_partial_to_full(self) -> None:
        order = apply_order_event(entry_order(), OrderState.SUBMITTED)
        pos = apply_position_transition(position(), PositionState.ENTRY_PENDING)

        pos, order, _ = apply_execution(pos, order, execution("E1", OrderSide.BUY, 4, 12.0))
        pos, order, _ = apply_execution(pos, order, execution("E2", OrderSide.BUY, 6, 13.0))
        self.assertEqual(order.state, OrderState.FILLED)
        self.assertEqual(pos.state, PositionState.OPEN)
        self.assertEqual(pos.open_quantity, 10)
        self.assertAlmostEqual(pos.avg_entry_price or 0, 12.6)

    def test_exit_partial_keeps_position_exit_pending(self) -> None:
        pos = PositionRecord(
            symbol="RKLB",
            strategy="v67",
            session_date="2026-05-18",
            state=PositionState.OPEN,
            entry_filled_quantity=10,
            open_quantity=10,
            avg_entry_price=12.5,
        )
        order = apply_order_event(exit_order(), OrderState.SUBMITTED)

        pos, order, applied = apply_execution(pos, order, execution("X1", OrderSide.SELL, 3, 12.2))
        self.assertTrue(applied)
        self.assertEqual(order.state, OrderState.PARTIAL)
        self.assertEqual(pos.state, PositionState.EXIT_PENDING)
        self.assertEqual(pos.open_quantity, 7)

    def test_duplicate_execution_id_ignored(self) -> None:
        order = apply_order_event(entry_order(), OrderState.SUBMITTED)
        pos = apply_position_transition(position(), PositionState.ENTRY_PENDING)

        pos, order, applied_first = apply_execution(pos, order, execution("E1", OrderSide.BUY, 4, 12.5))
        pos, order, applied_second = apply_execution(pos, order, execution("E1", OrderSide.BUY, 4, 12.5))
        self.assertTrue(applied_first)
        self.assertFalse(applied_second)
        self.assertEqual(pos.open_quantity, 4)
        self.assertEqual(order.filled_quantity, 4)

    def test_cancel_requested_before_delayed_fill_still_opens_position(self) -> None:
        order = apply_order_event(entry_order(), OrderState.SUBMITTED)
        order = apply_order_event(order, OrderState.CANCEL_REQUESTED)
        pos = apply_position_transition(position(), PositionState.ENTRY_PENDING)

        pos, order, applied = apply_execution(pos, order, execution("LATE1", OrderSide.BUY, 10, 12.7))
        self.assertTrue(applied)
        self.assertEqual(order.state, OrderState.FILLED)
        self.assertEqual(pos.state, PositionState.OPEN)
        self.assertEqual(pos.open_quantity, 10)

    def test_full_exit_closes_position(self) -> None:
        pos = PositionRecord(
            symbol="RKLB",
            strategy="v67",
            session_date="2026-05-18",
            state=PositionState.OPEN,
            entry_filled_quantity=10,
            open_quantity=10,
            avg_entry_price=12.5,
        )
        order = apply_order_event(exit_order(), OrderState.SUBMITTED)

        pos, order, applied = apply_execution(pos, order, execution("X1", OrderSide.SELL, 10, 12.2))
        self.assertTrue(applied)
        self.assertEqual(order.state, OrderState.FILLED)
        self.assertEqual(pos.state, PositionState.CLOSED)
        self.assertEqual(pos.open_quantity, 0)


class JsonlLifecycleStoreTests(unittest.TestCase):
    def test_append_execution_once_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "order_lifecycle.jsonl"
            store = JsonlLifecycleStore(path)
            exec_record = execution("E1", OrderSide.BUY, 10, 12.5)
            event = LifecycleEvent(
                event_type=LifecycleEventType.ENTRY_ORDER_FILLED,
                symbol="RKLB",
                strategy="v67",
                execution_id="E1",
                quantity=10,
                price=12.5,
            )
            self.assertTrue(store.append_execution_once(exec_record, event))
            self.assertFalse(store.append_execution_once(exec_record, event))
            self.assertEqual(len(store.load_events()), 1)


if __name__ == "__main__":
    unittest.main()
