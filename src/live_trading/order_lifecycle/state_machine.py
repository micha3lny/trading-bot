from __future__ import annotations

from dataclasses import replace

from src.live_trading.order_lifecycle.models import (
    ExecutionRecord,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderState,
    PositionRecord,
    PositionState,
    utc_now_iso,
)


def _weighted_average(old_qty: float, old_avg: float | None, add_qty: float, add_price: float) -> float:
    if old_qty <= 0 or old_avg is None:
        return add_price
    total_qty = old_qty + add_qty
    if total_qty <= 0:
        return add_price
    return ((old_avg * old_qty) + (add_price * add_qty)) / total_qty


def apply_order_event(order: OrderRecord, new_state: OrderState, *, ib_order_id: str | None = None, perm_id: str | None = None) -> OrderRecord:
    if order.state in {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.STALE}:
        if new_state not in {order.state, OrderState.PARTIAL, OrderState.FILLED}:
            return order
    return replace(
        order,
        state=new_state,
        ib_order_id=ib_order_id if ib_order_id is not None else order.ib_order_id,
        perm_id=perm_id if perm_id is not None else order.perm_id,
        submitted_at=utc_now_iso() if new_state == OrderState.SUBMITTED and not order.submitted_at else order.submitted_at,
        updated_at=utc_now_iso(),
    )


def apply_position_transition(position: PositionRecord, new_state: PositionState) -> PositionRecord:
    return replace(position, state=new_state, updated_at=utc_now_iso())


def apply_execution(
    position: PositionRecord,
    order: OrderRecord,
    execution: ExecutionRecord,
) -> tuple[PositionRecord, OrderRecord, bool]:
    if execution.execution_id in position.seen_execution_ids:
        return position, order, False
    if execution.quantity <= 0:
        return position, order, False

    seen = tuple([*position.seen_execution_ids, execution.execution_id])
    filled_qty = order.filled_quantity + execution.quantity
    order_avg = _weighted_average(order.filled_quantity, order.avg_fill_price, execution.quantity, execution.price)
    order_state = OrderState.FILLED if filled_qty >= order.quantity else OrderState.PARTIAL
    updated_order = replace(
        order,
        state=order_state,
        filled_quantity=filled_qty,
        avg_fill_price=order_avg,
        updated_at=utc_now_iso(),
    )

    if order.side == OrderSide.BUY and order.purpose == OrderPurpose.ENTRY:
        entry_qty = position.entry_filled_quantity + execution.quantity
        avg_entry = _weighted_average(position.entry_filled_quantity, position.avg_entry_price, execution.quantity, execution.price)
        updated_position = replace(
            position,
            state=PositionState.OPEN,
            entry_filled_quantity=entry_qty,
            avg_entry_price=avg_entry,
            open_quantity=entry_qty - position.exit_filled_quantity,
            peak_price=max(position.peak_price or execution.price, execution.price),
            seen_execution_ids=seen,
            updated_at=utc_now_iso(),
        )
        return updated_position, updated_order, True

    if order.side == OrderSide.SELL:
        exit_qty = position.exit_filled_quantity + execution.quantity
        avg_exit = _weighted_average(position.exit_filled_quantity, position.avg_exit_price, execution.quantity, execution.price)
        open_qty = max(0.0, position.entry_filled_quantity - exit_qty)
        state = PositionState.CLOSED if open_qty <= 0 else PositionState.EXIT_PENDING
        updated_position = replace(
            position,
            state=state,
            exit_filled_quantity=exit_qty,
            avg_exit_price=avg_exit,
            open_quantity=open_qty,
            seen_execution_ids=seen,
            updated_at=utc_now_iso(),
        )
        return updated_position, updated_order, True

    return replace(position, seen_execution_ids=seen, updated_at=utc_now_iso()), updated_order, True

