from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TextEnum(str, Enum):
    pass


class OrderSide(TextEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderPurpose(TextEnum):
    ENTRY = "ENTRY"
    STOP_LOSS_EXIT = "STOP_LOSS_EXIT"
    TRAILING_EXIT = "TRAILING_EXIT"
    EOD_FLATTEN = "EOD_FLATTEN"
    MANUAL_FLATTEN = "MANUAL_FLATTEN"
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"
    RECONCILIATION_FLATTEN = "RECONCILIATION_FLATTEN"


class OrderState(TextEnum):
    PREPARED = "PREPARED"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    STALE = "STALE"


class PositionState(TextEnum):
    NONE = "NONE"
    ENTRY_PENDING = "ENTRY_PENDING"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    RECONCILING = "RECONCILING"


class LifecycleEventType(TextEnum):
    ENTRY_SIGNAL = "ENTRY_SIGNAL"
    ENTRY_ORDER_PREPARED = "ENTRY_ORDER_PREPARED"
    ENTRY_ORDER_SUBMITTED = "ENTRY_ORDER_SUBMITTED"
    ENTRY_ORDER_PARTIAL = "ENTRY_ORDER_PARTIAL"
    ENTRY_ORDER_FILLED = "ENTRY_ORDER_FILLED"
    ENTRY_ORDER_CANCEL_REQUESTED = "ENTRY_ORDER_CANCEL_REQUESTED"
    ENTRY_ORDER_CANCELLED = "ENTRY_ORDER_CANCELLED"
    ENTRY_ORDER_REJECTED = "ENTRY_ORDER_REJECTED"
    ENTRY_ORDER_STALE = "ENTRY_ORDER_STALE"
    EXIT_ORDER_PREPARED = "EXIT_ORDER_PREPARED"
    EXIT_ORDER_SUBMITTED = "EXIT_ORDER_SUBMITTED"
    EXIT_ORDER_PARTIAL = "EXIT_ORDER_PARTIAL"
    EXIT_ORDER_FILLED = "EXIT_ORDER_FILLED"
    EXIT_ORDER_CANCEL_REQUESTED = "EXIT_ORDER_CANCEL_REQUESTED"
    EXIT_ORDER_CANCELLED = "EXIT_ORDER_CANCELLED"
    EXIT_ORDER_REJECTED = "EXIT_ORDER_REJECTED"
    EXIT_ORDER_STALE = "EXIT_ORDER_STALE"
    EXECUTION_RECORDED = "EXECUTION_RECORDED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_RECONCILING = "POSITION_RECONCILING"
    POSITION_ADOPTED = "POSITION_ADOPTED"
    POSITION_DRIFT_DETECTED = "POSITION_DRIFT_DETECTED"


@dataclass(frozen=True)
class OrderRecord:
    client_order_id: str
    symbol: str
    side: OrderSide
    purpose: OrderPurpose
    quantity: float
    state: OrderState = OrderState.PREPARED
    ib_order_id: str = ""
    perm_id: str = ""
    order_type: str = ""
    limit_price: float | None = None
    submitted_at: str = ""
    updated_at: str = field(default_factory=utc_now_iso)
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    executed_at: str = field(default_factory=utc_now_iso)
    ib_order_id: str = ""
    perm_id: str = ""
    commission: float | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionRecord:
    symbol: str
    strategy: str
    session_date: str
    state: PositionState = PositionState.NONE
    target_quantity: float = 0.0
    entry_filled_quantity: float = 0.0
    exit_filled_quantity: float = 0.0
    avg_entry_price: float | None = None
    avg_exit_price: float | None = None
    open_quantity: float = 0.0
    peak_price: float | None = None
    seen_execution_ids: tuple[str, ...] = ()
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class LifecycleEvent:
    event_type: LifecycleEventType
    symbol: str
    strategy: str
    state_before: PositionState | None = None
    state_after: PositionState | None = None
    client_order_id: str = ""
    ib_order_id: str = ""
    execution_id: str = ""
    order_state: OrderState | None = None
    position_state: PositionState | None = None
    quantity: float | None = None
    price: float | None = None
    reason: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key, value in list(row.items()):
            if isinstance(value, TextEnum):
                row[key] = value.value
        return row
