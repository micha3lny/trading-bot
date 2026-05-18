# v67 Order Lifecycle Foundation

This document defines the first formal order lifecycle layer for the v67 intraday momentum paper trader.

Stage 1 is intentionally non-invasive:

- current order behavior remains unchanged,
- ENTRY still uses the existing runtime path,
- EXIT/EOD/control API behavior remains unchanged,
- formal lifecycle events are emitted alongside existing CSV telemetry,
- no SQLite migration yet,
- no live marketable limit migration yet.

## Why This Exists

The previous runtime was position-first: after `placeOrder()`, the bot could already create or mutate a managed position. That is convenient for prototyping, but it is not a correct trading lifecycle.

Correct lifecycle is event-driven:

```text
Signal -> Order -> Execution -> Position
```

An order submission is not a fill. A fill is not guaranteed to arrive immediately. A cancel request is not a cancel confirmation. IBKR portfolio is the final source of truth for actual exposure.

## Core Invariants

1. BUY submit is not `OPEN`.
2. SELL submit is not `CLOSED`.
3. Only executions/fills can open or close position quantity.
4. Partial fills must update quantities, not pretend full completion.
5. Duplicate `execution_id` must be ignored.
6. Cancel requested can still receive a delayed fill.
7. `exit_sent` is not equivalent to closed.
8. IBKR portfolio is the exposure source of truth.
9. Restart recovery must reconcile local state, IBKR orders, executions, and portfolio before trusting local state.
10. Formal lifecycle state should be testable without IBKR.

## State Definitions

### OrderState

```text
PREPARED
SUBMITTED
PARTIAL
FILLED
CANCEL_REQUESTED
CANCELLED
REJECTED
STALE
```

### PositionState

```text
NONE
ENTRY_PENDING
OPEN
EXIT_PENDING
CLOSED
RECONCILING
```

### OrderSide

```text
BUY
SELL
```

### OrderPurpose

```text
ENTRY
STOP_LOSS_EXIT
TRAILING_EXIT
EOD_FLATTEN
MANUAL_FLATTEN
EMERGENCY_FLATTEN
RECONCILIATION_FLATTEN
```

## Stage 1 Files

```text
src/live_trading/order_lifecycle/
  __init__.py
  models.py
  state_machine.py
  store.py
  reconciliation.py
```

`models.py` contains enums and dataclasses.

`state_machine.py` is pure deterministic logic. It has no IBKR dependency.

`store.py` is append-only JSONL. This is deliberately smaller and safer than a first-pass SQLite migration.

`reconciliation.py` is a skeleton for future dry-run reconciliation reports.

## Runtime Emission

v67 still writes the legacy CSV lifecycle files. In addition, selected lifecycle events are mirrored to:

```text
data/live/recorder/{session_date}/order_lifecycle.jsonl
```

Examples:

```text
ENTRY_SIGNAL
ENTRY_ORDER_SUBMITTED
ENTRY_ORDER_FILLED
EXIT_ORDER_SUBMITTED
EXIT_ORDER_FILLED
POSITION_OPENED
POSITION_CLOSED
POSITION_ADOPTED
POSITION_DRIFT_DETECTED
```

This file is append-only and should be treated as telemetry/foundation in Stage 1, not yet the only source of truth.

## Why BUY Submit Is Not OPEN

IBKR can:

- reject the order,
- partially fill,
- fill after a cancel request,
- disconnect before status arrives,
- report execution before local state is persisted.

Therefore:

```text
ENTRY_ORDER_SUBMITTED -> ENTRY_PENDING
ENTRY_ORDER_FILLED -> OPEN
```

## Why exit_sent Is Not CLOSED

Sending SELL means only:

```text
EXIT_ORDER_SUBMITTED -> EXIT_PENDING
```

The position is closed only when fills or portfolio reconciliation show zero remaining quantity:

```text
EXIT_ORDER_FILLED -> CLOSED
```

## Source Of Truth

During runtime:

1. Local lifecycle state explains intent and expected transitions.
2. IBKR executions provide idempotent fills.
3. IBKR portfolio is the final exposure truth.

When they disagree, bot must prefer safety:

- block new entries,
- allow exits,
- mark `RECONCILING`,
- emit drift/orphan telemetry.

## Stage 2 Direction

The next stage should not switch to marketable limits yet. Recommended order:

1. Expand formal event emission for all order statuses.
2. Add startup dry-run reconciliation using IBKR open orders, executions, and portfolio.
3. Make EXIT/EOD lifecycle state-driven while preserving MarketOrder exits.
4. Only then add ENTRY marketable limit in shadow mode.
