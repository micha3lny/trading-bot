from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any

from src.live_trading.order_lifecycle.models import (
    LifecycleEventType,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderState,
    PositionRecord,
    PositionState,
)


@dataclass(frozen=True)
class LifecycleSnapshot:
    positions: dict[str, PositionRecord]
    orders: dict[str, OrderRecord]
    anomalies: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": {symbol: _dataclass_to_dict(pos) for symbol, pos in sorted(self.positions.items())},
            "orders": {order_id: _dataclass_to_dict(order) for order_id, order in sorted(self.orders.items())},
            "anomalies": list(self.anomalies),
            "summary": {
                "positions": len(self.positions),
                "orders": len(self.orders),
                "anomalies": len(self.anomalies),
                "open_positions": sum(1 for pos in self.positions.values() if pos.state == PositionState.OPEN),
                "exit_pending_positions": sum(1 for pos in self.positions.values() if pos.state == PositionState.EXIT_PENDING),
                "reconciling_positions": sum(1 for pos in self.positions.values() if pos.state == PositionState.RECONCILING),
            },
        }


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    row = asdict(value)
    for key, item in list(row.items()):
        if hasattr(item, "value"):
            row[key] = item.value
    return row


def _event_type(row: dict[str, Any]) -> str:
    return str(row.get("event_type") or row.get("event") or "").upper().strip()


def _symbol(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or "").upper().strip()


def _recorded_at(row: dict[str, Any]) -> str:
    return str(row.get("recorded_at") or "")


def _event_time(row: dict[str, Any]) -> datetime | None:
    value = _recorded_at(row)
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _session_date(row: dict[str, Any]) -> str:
    ts = _event_time(row)
    if ts is not None:
        return ts.date().isoformat()
    return str(row.get("session_date") or "")


def _strategy(row: dict[str, Any]) -> str:
    return str(row.get("strategy") or "unknown")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _weighted_average(old_qty: float, old_avg: float | None, add_qty: float, add_price: float | None) -> float | None:
    if add_price is None:
        return old_avg
    if old_qty <= 0 or old_avg is None:
        return add_price
    total = old_qty + add_qty
    if total <= 0:
        return add_price
    return ((old_avg * old_qty) + (add_price * add_qty)) / total


def _parse_order_state(value: Any) -> OrderState | None:
    if value is None or value == "":
        return None
    try:
        return OrderState(str(value))
    except Exception:
        return None


def _position_for_event(row: dict[str, Any]) -> PositionRecord:
    return PositionRecord(
        symbol=_symbol(row),
        strategy=_strategy(row),
        session_date=_session_date(row),
        updated_at=_recorded_at(row),
    )


def _infer_side(event_type: str, row: dict[str, Any]) -> OrderSide:
    action = str(row.get("action") or "").upper().strip()
    if action in {OrderSide.BUY.value, OrderSide.SELL.value}:
        return OrderSide(action)
    if event_type.startswith("ENTRY_"):
        return OrderSide.BUY
    return OrderSide.SELL


def _infer_purpose(event_type: str, row: dict[str, Any]) -> OrderPurpose:
    if event_type.startswith("ENTRY_"):
        return OrderPurpose.ENTRY
    reason = str(row.get("reason") or "").lower()
    raw = row.get("raw_json") if isinstance(row.get("raw_json"), dict) else {}
    raw_reason = str(raw.get("reason") or "").lower() if isinstance(raw, dict) else ""
    joined = f"{reason} {raw_reason}"
    if "manual" in joined:
        return OrderPurpose.MANUAL_FLATTEN
    if "reconciliation" in joined:
        return OrderPurpose.RECONCILIATION_FLATTEN
    if "emergency" in joined:
        return OrderPurpose.EMERGENCY_FLATTEN
    if "trail" in joined:
        return OrderPurpose.TRAILING_EXIT
    if "stop" in joined:
        return OrderPurpose.STOP_LOSS_EXIT
    return OrderPurpose.EOD_FLATTEN


def _order_key(event_type: str, row: dict[str, Any], fallback_index: int) -> str:
    for key in ("client_order_id", "ib_order_id", "order_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    symbol = _symbol(row) or "__UNKNOWN__"
    purpose = _infer_purpose(event_type, row).value
    side = _infer_side(event_type, row).value
    return f"{symbol}:{purpose}:{side}:{fallback_index:06d}"


def _default_order(event_type: str, row: dict[str, Any], order_id: str) -> OrderRecord:
    return OrderRecord(
        client_order_id=str(row.get("client_order_id") or ""),
        symbol=_symbol(row),
        side=_infer_side(event_type, row),
        purpose=_infer_purpose(event_type, row),
        quantity=max(0.0, _safe_float(row.get("quantity"))),
        state=OrderState.PREPARED,
        ib_order_id=str(row.get("ib_order_id") or row.get("order_id") or ""),
        perm_id=str(row.get("perm_id") or ""),
        updated_at=_recorded_at(row),
        raw_json={"reducer_order_key": order_id},
    )


def _order_state_for_event(event_type: str, row: dict[str, Any]) -> OrderState | None:
    explicit = _parse_order_state(row.get("order_state"))
    if explicit is not None:
        return explicit
    mapping = {
        LifecycleEventType.ENTRY_ORDER_PREPARED.value: OrderState.PREPARED,
        LifecycleEventType.EXIT_ORDER_PREPARED.value: OrderState.PREPARED,
        LifecycleEventType.ENTRY_ORDER_SUBMITTED.value: OrderState.SUBMITTED,
        LifecycleEventType.EXIT_ORDER_SUBMITTED.value: OrderState.SUBMITTED,
        LifecycleEventType.ENTRY_ORDER_PARTIAL.value: OrderState.PARTIAL,
        LifecycleEventType.EXIT_ORDER_PARTIAL.value: OrderState.PARTIAL,
        LifecycleEventType.ENTRY_ORDER_FILLED.value: OrderState.FILLED,
        LifecycleEventType.EXIT_ORDER_FILLED.value: OrderState.FILLED,
        LifecycleEventType.ENTRY_ORDER_CANCEL_REQUESTED.value: OrderState.CANCEL_REQUESTED,
        LifecycleEventType.EXIT_ORDER_CANCEL_REQUESTED.value: OrderState.CANCEL_REQUESTED,
        LifecycleEventType.ENTRY_ORDER_CANCELLED.value: OrderState.CANCELLED,
        LifecycleEventType.EXIT_ORDER_CANCELLED.value: OrderState.CANCELLED,
        LifecycleEventType.ENTRY_ORDER_REJECTED.value: OrderState.REJECTED,
        LifecycleEventType.EXIT_ORDER_REJECTED.value: OrderState.REJECTED,
        LifecycleEventType.ENTRY_ORDER_STALE.value: OrderState.STALE,
        LifecycleEventType.EXIT_ORDER_STALE.value: OrderState.STALE,
    }
    return mapping.get(event_type)


def _apply_order_event(order: OrderRecord, event_type: str, row: dict[str, Any]) -> OrderRecord:
    new_state = _order_state_for_event(event_type, row)
    quantity = _safe_float(row.get("quantity"), order.quantity)
    if quantity <= 0:
        quantity = order.quantity
    if new_state is None:
        return order
    return replace(
        order,
        state=new_state,
        quantity=max(order.quantity, quantity),
        ib_order_id=str(row.get("ib_order_id") or row.get("order_id") or order.ib_order_id),
        perm_id=str(row.get("perm_id") or order.perm_id),
        submitted_at=_recorded_at(row) if new_state == OrderState.SUBMITTED and not order.submitted_at else order.submitted_at,
        updated_at=_recorded_at(row),
    )


def _apply_execution(
    position: PositionRecord,
    order: OrderRecord,
    row: dict[str, Any],
    seen_execution_ids: set[str],
) -> tuple[PositionRecord, OrderRecord, bool]:
    execution_id = str(row.get("execution_id") or "").strip()
    if execution_id and execution_id in seen_execution_ids:
        return position, order, False
    if execution_id:
        seen_execution_ids.add(execution_id)

    quantity = _safe_float(row.get("quantity"))
    price = _safe_float(row.get("price"), default=0.0) if row.get("price") is not None else None
    if quantity <= 0:
        return position, order, False

    filled_quantity = order.filled_quantity + quantity
    avg_fill = _weighted_average(order.filled_quantity, order.avg_fill_price, quantity, price)
    target_quantity = max(order.quantity, filled_quantity)
    order_state = OrderState.FILLED if target_quantity > 0 and filled_quantity >= target_quantity else OrderState.PARTIAL
    order = replace(
        order,
        quantity=target_quantity,
        state=order_state,
        filled_quantity=filled_quantity,
        avg_fill_price=avg_fill,
        updated_at=_recorded_at(row),
    )

    seen = tuple(sorted([*position.seen_execution_ids, execution_id])) if execution_id else position.seen_execution_ids
    if order.side == OrderSide.BUY and order.purpose == OrderPurpose.ENTRY:
        entry_qty = position.entry_filled_quantity + quantity
        avg_entry = _weighted_average(position.entry_filled_quantity, position.avg_entry_price, quantity, price)
        position = replace(
            position,
            state=PositionState.OPEN,
            target_quantity=max(position.target_quantity, target_quantity),
            entry_filled_quantity=entry_qty,
            avg_entry_price=avg_entry,
            open_quantity=entry_qty - position.exit_filled_quantity,
            peak_price=max(position.peak_price or price or 0.0, price or 0.0) or position.peak_price,
            seen_execution_ids=seen,
            updated_at=_recorded_at(row),
        )
        return position, order, True

    exit_qty = position.exit_filled_quantity + quantity
    avg_exit = _weighted_average(position.exit_filled_quantity, position.avg_exit_price, quantity, price)
    open_qty = max(0.0, position.entry_filled_quantity - exit_qty)
    position = replace(
        position,
        state=PositionState.CLOSED if open_qty <= 0 else PositionState.EXIT_PENDING,
        exit_filled_quantity=exit_qty,
        avg_exit_price=avg_exit,
        open_quantity=open_qty,
        seen_execution_ids=seen,
        updated_at=_recorded_at(row),
    )
    return position, order, True


def _anomaly(anomalies: list[dict[str, Any]], kind: str, row: dict[str, Any], **fields: Any) -> None:
    anomalies.append(
        {
            "kind": kind,
            "event_type": _event_type(row),
            "symbol": _symbol(row),
            "recorded_at": _recorded_at(row),
            **fields,
        }
    )


def reduce_lifecycle_events(events: list[dict[str, Any]]) -> LifecycleSnapshot:
    positions: dict[str, PositionRecord] = {}
    orders: dict[str, OrderRecord] = {}
    anomalies: list[dict[str, Any]] = []
    seen_execution_ids: set[str] = set()
    last_time: datetime | None = None

    for idx, row in enumerate(events):
        if not isinstance(row, dict):
            anomalies.append({"kind": "bad_event_row", "index": idx, "row_type": type(row).__name__})
            continue

        event_type = _event_type(row)
        symbol = _symbol(row)
        event_time = _event_time(row)
        if last_time is not None and event_time is not None and event_time < last_time:
            _anomaly(anomalies, "out_of_order_event", row, index=idx)
        if event_time is not None:
            last_time = event_time
        if not symbol:
            _anomaly(anomalies, "missing_symbol", row, index=idx)
            continue

        pos = positions.get(symbol) or _position_for_event(row)

        if event_type in {
            LifecycleEventType.ENTRY_ORDER_PREPARED.value,
            LifecycleEventType.ENTRY_ORDER_SUBMITTED.value,
            LifecycleEventType.ENTRY_ORDER_PARTIAL.value,
            LifecycleEventType.ENTRY_ORDER_FILLED.value,
            LifecycleEventType.ENTRY_ORDER_CANCEL_REQUESTED.value,
            LifecycleEventType.ENTRY_ORDER_CANCELLED.value,
            LifecycleEventType.ENTRY_ORDER_REJECTED.value,
            LifecycleEventType.ENTRY_ORDER_STALE.value,
            LifecycleEventType.EXIT_ORDER_PREPARED.value,
            LifecycleEventType.EXIT_ORDER_SUBMITTED.value,
            LifecycleEventType.EXIT_ORDER_PARTIAL.value,
            LifecycleEventType.EXIT_ORDER_FILLED.value,
            LifecycleEventType.EXIT_ORDER_CANCEL_REQUESTED.value,
            LifecycleEventType.EXIT_ORDER_CANCELLED.value,
            LifecycleEventType.EXIT_ORDER_REJECTED.value,
            LifecycleEventType.EXIT_ORDER_STALE.value,
        }:
            order_id = _order_key(event_type, row, idx)
            order = orders.get(order_id) or _default_order(event_type, row, order_id)
            if event_type == LifecycleEventType.ENTRY_ORDER_SUBMITTED.value and pos.state not in {PositionState.NONE, PositionState.ENTRY_PENDING}:
                _anomaly(anomalies, "entry_submit_from_non_empty_position", row, previous_state=pos.state.value)
            if event_type == LifecycleEventType.EXIT_ORDER_SUBMITTED.value and pos.state not in {PositionState.OPEN, PositionState.EXIT_PENDING, PositionState.RECONCILING}:
                _anomaly(anomalies, "exit_submit_without_open_position", row, previous_state=pos.state.value)

            order = _apply_order_event(order, event_type, row)

            if event_type == LifecycleEventType.ENTRY_ORDER_PREPARED.value:
                pos = replace(pos, state=PositionState.ENTRY_PENDING, target_quantity=max(pos.target_quantity, order.quantity), updated_at=_recorded_at(row))
            elif event_type == LifecycleEventType.ENTRY_ORDER_SUBMITTED.value:
                pos = replace(pos, state=PositionState.ENTRY_PENDING, target_quantity=max(pos.target_quantity, order.quantity), updated_at=_recorded_at(row))
            elif event_type in {LifecycleEventType.ENTRY_ORDER_PARTIAL.value, LifecycleEventType.ENTRY_ORDER_FILLED.value}:
                pos, order, applied = _apply_execution(pos, order, row, seen_execution_ids)
                if not applied:
                    _anomaly(anomalies, "duplicate_or_invalid_entry_execution", row, order_key=order_id)
            elif event_type == LifecycleEventType.ENTRY_ORDER_REJECTED.value:
                pos = replace(pos, state=PositionState.RECONCILING, updated_at=_recorded_at(row))
            elif event_type in {
                LifecycleEventType.ENTRY_ORDER_CANCEL_REQUESTED.value,
                LifecycleEventType.ENTRY_ORDER_CANCELLED.value,
                LifecycleEventType.ENTRY_ORDER_STALE.value,
            }:
                next_state = PositionState.OPEN if pos.open_quantity > 0 else PositionState.NONE
                pos = replace(pos, state=next_state, updated_at=_recorded_at(row))
            elif event_type == LifecycleEventType.EXIT_ORDER_PREPARED.value:
                next_state = PositionState.EXIT_PENDING if pos.open_quantity > 0 else PositionState.RECONCILING
                pos = replace(pos, state=next_state, updated_at=_recorded_at(row))
            elif event_type == LifecycleEventType.EXIT_ORDER_SUBMITTED.value:
                next_state = PositionState.EXIT_PENDING if pos.open_quantity > 0 else PositionState.RECONCILING
                pos = replace(pos, state=next_state, updated_at=_recorded_at(row))
            elif event_type in {LifecycleEventType.EXIT_ORDER_PARTIAL.value, LifecycleEventType.EXIT_ORDER_FILLED.value}:
                if pos.entry_filled_quantity <= 0 and pos.open_quantity <= 0:
                    _anomaly(anomalies, "exit_fill_without_known_open_quantity", row, order_key=order_id)
                    pos = replace(pos, state=PositionState.RECONCILING, updated_at=_recorded_at(row))
                    order = replace(order, state=OrderState.FILLED if event_type == LifecycleEventType.EXIT_ORDER_FILLED.value else OrderState.PARTIAL, updated_at=_recorded_at(row))
                else:
                    pos, order, applied = _apply_execution(pos, order, row, seen_execution_ids)
                    if not applied:
                        _anomaly(anomalies, "duplicate_or_invalid_exit_execution", row, order_key=order_id)
            elif event_type == LifecycleEventType.EXIT_ORDER_REJECTED.value:
                pos = replace(pos, state=PositionState.RECONCILING, updated_at=_recorded_at(row))
            elif event_type in {
                LifecycleEventType.EXIT_ORDER_CANCEL_REQUESTED.value,
                LifecycleEventType.EXIT_ORDER_CANCELLED.value,
                LifecycleEventType.EXIT_ORDER_STALE.value,
            }:
                next_state = PositionState.OPEN if pos.open_quantity > 0 else PositionState.RECONCILING
                pos = replace(pos, state=next_state, updated_at=_recorded_at(row))

            orders[order_id] = order
            positions[symbol] = pos
            continue

        if event_type == LifecycleEventType.POSITION_OPENED.value:
            quantity = _safe_float(row.get("quantity"))
            price = _safe_float(row.get("price"), default=0.0) if row.get("price") is not None else None
            positions[symbol] = replace(
                pos,
                state=PositionState.OPEN,
                target_quantity=max(pos.target_quantity, quantity),
                entry_filled_quantity=max(pos.entry_filled_quantity, quantity),
                avg_entry_price=price or pos.avg_entry_price,
                open_quantity=max(pos.open_quantity, quantity),
                updated_at=_recorded_at(row),
            )
        elif event_type == LifecycleEventType.POSITION_CLOSED.value:
            positions[symbol] = replace(pos, state=PositionState.CLOSED, open_quantity=0.0, updated_at=_recorded_at(row))
        elif event_type == LifecycleEventType.POSITION_ADOPTED.value:
            quantity = _safe_float(row.get("quantity"))
            price = _safe_float(row.get("price"), default=0.0) if row.get("price") is not None else None
            positions[symbol] = replace(
                pos,
                state=PositionState.OPEN,
                target_quantity=max(pos.target_quantity, quantity),
                entry_filled_quantity=max(pos.entry_filled_quantity, quantity),
                avg_entry_price=price or pos.avg_entry_price,
                open_quantity=max(pos.open_quantity, quantity),
                updated_at=_recorded_at(row),
            )
        elif event_type in {LifecycleEventType.POSITION_DRIFT_DETECTED.value, LifecycleEventType.POSITION_RECONCILING.value}:
            positions[symbol] = replace(pos, state=PositionState.RECONCILING, updated_at=_recorded_at(row))
        elif event_type == LifecycleEventType.ENTRY_SIGNAL.value:
            positions.setdefault(symbol, pos)
        else:
            _anomaly(anomalies, "unsupported_event_type", row, index=idx)
            positions.setdefault(symbol, pos)

    return LifecycleSnapshot(positions=positions, orders=orders, anomalies=anomalies)
