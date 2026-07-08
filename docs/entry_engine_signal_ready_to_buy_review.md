# Entry Engine Review: `SIGNAL_READY` to `BUY_ORDER_SENT`

Date reviewed: 2026-07-08

Scope: read-only review of `src/live_trading/v67_live_top100_expansion_paper_trader.py`, focused on every path where a candidate can leave the entry engine after `SIGNAL_READY` but before `BUY_ORDER_SENT`.

Key operational finding from SHS/NBAS:

- Offline says `should_have_signaled`.
- Runtime has `SIGNAL_READY`.
- Candidate was present/selected and final filters appear passed.
- `order_dispatch_attempted = 0`.
- No `BUY_ORDER_SENT` / `PAPER BUY SENT`.
- NBAS points to `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`.

## Important Code Ordering

In the current v67 loop, `SIGNAL_READY` is emitted only after the symbol has already:

1. Had a usable ticker price.
2. Computed live features.
3. Passed `features["ready"]`.
4. Entered `entry_candidates`.
5. Survived `ready_candidate_rejection_reason(...)`.
6. Survived `max_entries_per_cycle`.
7. Survived `entry_minute_capacity(...)`.
8. Been selected inside `ordered_entry_candidates`.

Therefore, for cases where `SIGNAL_READY` exists, the candidate did not disappear in the early subscription/ready/ranking path. It disappeared after line ~6661 and before line ~6880 or ~6898.

## Entry Loop Map

Source: `src/live_trading/v67_live_top100_expansion_paper_trader.py`

| Stage | Approx lines | What happens |
|---|---:|---|
| Ticker snapshot | 6471-6477 | `snapshot_from_ticker`; `continue` if no usable price |
| Feature/ready calculation | 6481-6498 | `update_state`, `compute_live_safe_features`; resets ready state if not ready |
| Candidate creation | 6500-6540 | active/ineligible/entry-symbol checks; appends to `entry_candidates` |
| Global entry block lifecycle | 6541-6559 | records `BUY_BLOCKED` when ready but entries blocked |
| Candidate rejection map | 6575-6584 | calls `ready_candidate_rejection_reason` |
| Candidate sort | 6597-6603 | sorts `entry_candidates` by score |
| Stale/backfill rejection | 6617-6641 | logs `STALE_OR_BACKFILL_READY_SKIPPED`; `continue` |
| Per-cycle limit | 6642-6644 | `break` if `entries_submitted_this_cycle >= max_entries_per_cycle` |
| Per-minute limit | 6645-6656 | logs `ENTRY_RATE_LIMIT_BLOCK`; `break` |
| `SIGNAL_READY` | 6660-6670 | records entry signal lifecycle |
| Low-score filter | 6671-6694 | logs `ENTRY_BLOCKED_LOW_SCORE`; `continue` |
| Risk guard | 6695-6745 | logs `RISK_GUARD_BLOCK_ENTRY`; `continue` |
| Order creation/submission | 6746-6749 | `MarketOrder`, `ib.placeOrder` |
| Runtime/SQLite order state | 6750-6818 | order ids, entry metadata, `upsert_trade`, `upsert_order`, entry submission count |
| Managed position persistence | 6850-6872 | creates `ManagedPosition`, `persist_managed_positions` |
| `BUY_ORDER_SENT` lifecycle | 6873-6897 | records `BUY_ORDER_SENT` |
| Console order log | 6898-6908 | prints `PAPER BUY SENT` |
| Main loop exception handler | 7151-7161 | unhandled exception marks shutdown reason and re-raises |

## Candidate Exit Paths

### 1. Low Live Entry Score Block

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`
- Approx lines: 6671-6694
- Condition:
  - `low_live_entry_score_blocked(live_entry_score, min_live_entry_score)` returns true.
- Control flow:
  - `continue`
- Log emitted today:
  - Console: `ENTRY_BLOCKED_LOW_SCORE symbol=... live_entry_score=... min_live_entry_score=...`
  - Lifecycle: `ENTRY_BLOCKED_LOW_SCORE`
- Why NBAS would detect it:
  - It searches symbol-specific text after `SIGNAL_READY`; this event includes the symbol and reason.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - No, unless the lifecycle/console write failed. Normally NBAS should classify it as a final score filter, not a dispatch gap.
- Suggested diagnostic log:
  - Already sufficient. Optional: `ENTRY_FINAL_FILTER_BLOCKED_LOW_SCORE`.

### 2. Risk Guard Block After Signal

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`, with helper `evaluate_risk_guard`
- Approx lines:
  - `evaluate_risk_guard`: 4936-4991
  - risk guard branch: 6695-6745
- Condition:
  - `risk_status.get("blocked")` is true.
  - Reasons include `max_daily_loss`, `max_trades_per_day`, `max_open_positions`, `max_gross_exposure`, `max_single_position`.
- Control flow:
  - `continue`
- Log emitted today:
  - Console, rate-limited by symbol/reason: `RISK_GUARD_BLOCK_ENTRY symbol=... reason=...`
  - Lifecycle: `RISK_GUARD_BLOCK_ENTRY`
  - SQLite risk event: `record_risk_event`
- Why NBAS would detect it:
  - Symbol-specific `RISK_GUARD_BLOCK_ENTRY` should be visible in lifecycle/journal/SQLite.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - No, if logs are present. Yes only if logging is suppressed or failed. The console log is rate-limited but lifecycle recording is not intentionally rate-limited.
- Suggested diagnostic log:
  - Already mostly sufficient.
  - Add explicit `ENTRY_FINAL_FILTER_BLOCKED_RISK_GUARD` if we want one uniform post-signal namespace.

### 3. `ib.placeOrder` Throws Before Any Dispatch Log

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`
- Approx lines: 6746-6749
- Condition:
  - `ib.placeOrder(q, order)` raises an exception.
  - Examples could include disconnect, invalid contract object, IB API thread issue, pacing/state problem, unexpected ib_insync error.
- Control flow:
  - No local `try/except`.
  - Exception exits the entry loop and is caught only by the outer `except Exception` at lines 7159-7161, which re-raises after marking shutdown reason.
- Log emitted today:
  - No symbol-specific post-signal log before the call.
  - Outer crash handling may emit `BOT_CRASH` at process level, not `symbol=...`.
- Why NBAS would not detect it:
  - NBAS sees `SIGNAL_READY` but no `BUY_ORDER_SENT`, no `PAPER BUY SENT`, no order row.
  - Unless logs include crash context close enough to symbol/time, it cannot tie the failure to that symbol.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - Yes. This is one of the highest-probability silent gaps.
- Suggested diagnostic log:
  - Before call: `ENTRY_ORDER_DISPATCH_ATTEMPT symbol=... qty=... price=... ranking_position=...`
  - Around call: catch and log `ENTRY_ORDER_DISPATCH_EXCEPTION symbol=... error=... stage=ib.placeOrder`
  - After call: `ENTRY_ORDER_DISPATCH_RETURNED symbol=... orderId=... permId=...`

### 4. `ib.placeOrder` Returns a Trade Without Expected Order Shape

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`
- Approx lines: 6749-6751, 6880-6898
- Condition:
  - `trade = ib.placeOrder(...)` returns, but `trade.order` is missing/None or lacks `orderId`.
  - Early metadata uses safe `getattr`, but later code uses `trade.order.orderId` directly at line ~6887.
- Control flow:
  - Could proceed with blank `order_id_for_entry`.
  - Could throw later at `trade.order.orderId`.
- Log emitted today:
  - No explicit warning for blank/missing `orderId`.
  - If exception happens at `BUY_ORDER_SENT` lifecycle call, no `BUY_ORDER_SENT` and no `PAPER BUY SENT`.
- Why NBAS would not detect it:
  - `SIGNAL_READY` exists.
  - There may be no order row if failure occurs before or during `upsert_order`.
  - If failure occurs at `BUY_ORDER_SENT`, SQLite `upsert_order` may exist, but `BUY_ORDER_SENT` is absent.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - Yes.
- Suggested diagnostic log:
  - `ENTRY_ORDER_DISPATCH_RETURNED_MISSING_ORDER_ID symbol=... trade_repr=...`
  - Use safe `order_id_for_entry` in `record_lifecycle_with_formal` instead of direct `trade.order.orderId`.

### 5. `json.dumps(features, ...)` or Metadata Construction Throws

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`
- Approx lines: 6752-6765
- Condition:
  - `json.dumps(features, ensure_ascii=False, default=str, sort_keys=True)` unexpectedly raises.
  - `_runtime_dict(...)` or metadata conversion throws.
- Control flow:
  - No local `try/except`; outer exception handler re-raises.
- Log emitted today:
  - No symbol-specific log after `SIGNAL_READY`.
- Why NBAS would not detect it:
  - Same signature: `SIGNAL_READY`, no order dispatch record.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - Possible, but lower probability because `default=str` should handle most values.
- Suggested diagnostic log:
  - `ENTRY_ORDER_METADATA_BUILD_FAILED symbol=... error=...`

### 6. Runtime Entry Order Map Update Throws

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`
- Approx lines: 6766-6776
- Condition:
  - `_runtime_dict(runtime_state, "entry_order_by_order_id")` or `_runtime_order_id(order_id_for_entry)` fails.
- Control flow:
  - No local `try/except`.
- Log emitted today:
  - None.
- Why NBAS would not detect it:
  - No symbol-specific stage log.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - Possible, but lower probability.
- Suggested diagnostic log:
  - `ENTRY_ORDER_RUNTIME_MAP_FAILED symbol=... order_id=... error=...`

### 7. SQLite `upsert_trade` / `upsert_order` Blocks or Throws

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`
- Approx lines: 6777-6816
- Condition:
  - `safe_sqlite_call(..., "upsert_trade", ...)` or `safe_sqlite_call(..., "upsert_order", ...)` blocks, times out, or raises.
  - We have seen SQLite ACK timeouts elsewhere.
- Control flow:
  - Depends on `safe_sqlite_call` behavior in `src/live_trading/storage/sqlite_store.py`.
  - In this call site there is no local `try/except`.
- Log emitted today:
  - SQLite writer may emit `SQLITE_WRITE_ACK_TIMEOUT` or write errors.
  - No entry-stage log such as "order submitted to IBKR but SQLite record failed".
- Why NBAS would not detect it:
  - If `ib.placeOrder` succeeded but SQLite write blocked/threw before `BUY_ORDER_SENT`, NBAS sees no `BUY_ORDER_SENT`.
  - Broker might still receive the order, causing hard-to-reconcile cases.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - Yes, especially on days with SQLite queue pressure.
- Suggested diagnostic log:
  - Before DB writes: `ENTRY_ORDER_IBKR_SUBMITTED symbol=... orderId=...`
  - On DB issue: `ENTRY_ORDER_SQLITE_PERSIST_FAILED symbol=... orderId=... method=... error=...`
  - Do not wait until after SQLite writes to emit the first durable dispatch marker.

### 8. Entry Submission Counter / Backlog Detection Throws

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Functions:
  - `record_entry_submission`: 950-955
  - `main`: 6817-6849
- Condition:
  - `record_entry_submission` or backlog code throws.
- Control flow:
  - No local `try/except`.
- Log emitted today:
  - `ENTRY_BACKLOG_DETECTED` only if threshold exceeded; not a failure log.
- Why NBAS would not detect it:
  - `BUY_ORDER_SENT` is logged after this block, so an exception here leaves `SIGNAL_READY` without `BUY_ORDER_SENT`.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - Possible, but lower probability.
- Suggested diagnostic log:
  - `ENTRY_SUBMISSION_COUNTER_FAILED symbol=... orderId=... error=...`

### 9. Managed Position Creation or Persistence Throws

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`
- Approx lines: 6850-6872
- Condition:
  - `ManagedPosition(...)` constructor receives unexpected value.
  - `persist_managed_positions(...)` fails on file I/O or serialization.
- Control flow:
  - No local `try/except`.
- Log emitted today:
  - None at this stage.
- Why NBAS would not detect it:
  - `BUY_ORDER_SENT` and `PAPER BUY SENT` happen after this block.
  - A persistence exception here creates the exact `SIGNAL_READY` without order-sent signature.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - Yes.
- Suggested diagnostic log:
  - Before persistence: `ENTRY_ORDER_IBKR_SUBMITTED`
  - On failure: `ENTRY_MANAGED_POSITION_PERSIST_FAILED symbol=... orderId=... error=...`

### 10. `record_lifecycle_with_formal("BUY_ORDER_SENT", ...)` Throws Before Print

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Functions:
  - `record_lifecycle_with_formal`: 1129+
  - call site: 6880-6897
- Condition:
  - `record_lifecycle(...)` throws before formal wrapper can catch anything.
  - Direct `trade.order.orderId` access throws.
  - Recorder file write fails.
- Control flow:
  - `record_lifecycle_with_formal` does not wrap the initial `record_lifecycle(...)` call in a try/except.
  - The formal event subpath catches `formal_lifecycle_record_error`, but only after legacy lifecycle succeeds.
- Log emitted today:
  - If formal subpath fails: `formal_lifecycle_record_error`.
  - If legacy `record_lifecycle` or direct `trade.order.orderId` fails before that: no `BUY_ORDER_SENT`, no `PAPER BUY SENT`.
- Why NBAS would not detect it:
  - It keys off `BUY_ORDER_SENT` / `PAPER BUY SENT`.
  - An exception in this recorder call erases both signals.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - Yes. This is a high-probability silent gap because the final order log happens after this call.
- Suggested diagnostic log:
  - Emit `ENTRY_ORDER_IBKR_SUBMITTED` before any recorder/SQLite persistence.
  - Wrap `BUY_ORDER_SENT` lifecycle recording with `ENTRY_ORDER_LIFECYCLE_RECORD_FAILED symbol=... orderId=... error=...`.

### 11. Final `print("PAPER BUY SENT ...")` Throws

- Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`
- Function: `main`
- Approx lines: 6898-6908
- Condition:
  - Formatting accesses unexpected values.
  - stdout write fails.
- Control flow:
  - No local `try/except`.
- Log emitted today:
  - `BUY_ORDER_SENT` lifecycle should already exist if the failure is only here.
- Why NBAS would or would not detect it:
  - NBAS should still see `BUY_ORDER_SENT` from recorder/SQLite if that succeeded.
  - It would only miss console `PAPER BUY SENT`.
- Can explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?
  - No, unless lifecycle recording also failed earlier.
- Suggested diagnostic log:
  - Not critical; lifecycle is the better source.

## Paths That Do Not Explain `SIGNAL_READY` Without `BUY_ORDER_SENT`

These paths happen before `SIGNAL_READY`. They are important for other missed-signal cases, but they do not explain the current NBAS signature when `SIGNAL_READY` is already present.

### No Usable Ticker Price

- Approx lines: 6471-6477
- Control flow: `continue`
- Log: `NO_USABLE_TICKER_PRICE` once per interval.
- NBAS relation: cannot produce `SIGNAL_READY`.

### Not Ready / Feature Rejection

- Approx lines: 6492-6498
- Control flow: state reset; no candidate append.
- Log: heartbeat rejection summary only.
- NBAS relation: cannot produce `SIGNAL_READY`.

### Ineligible Symbol

- Approx lines: 6503-6523
- Control flow: `continue`
- Log: `ENTRY_SYMBOL_INELIGIBLE_SKIPPED`.
- NBAS relation: cannot produce `SIGNAL_READY`.

### Entries Blocked

- Approx lines: 6541-6559 and `if not entries_blocked` at 6597
- Control flow: candidate is recorded as `BUY_BLOCKED`, entry loop skipped globally.
- Log: `BUY_BLOCKED`.
- NBAS relation: if `SIGNAL_READY` exists, this was not the blocker for that candidate in that scan.

### Stale / Backfill Candidate

- Helper: `ready_candidate_rejection_reason`, lines 898-921
- Branch: 6617-6641
- Control flow: `continue`
- Log: `STALE_OR_BACKFILL_READY_SKIPPED`.
- NBAS relation: this branch is before `SIGNAL_READY`.

### Max Entries Per Cycle

- Approx lines: 6642-6644
- Control flow: `break`
- Log: none today.
- NBAS relation: this branch is before `SIGNAL_READY`; it cannot explain `SIGNAL_READY` for the skipped symbol. It can explain candidates ranked below selected entries that never got `SIGNAL_READY`.
- Suggested diagnostic log anyway:
  - `ENTRY_CYCLE_LIMIT_REACHED ready_candidates=... entries_submitted_this_cycle=... next_symbol=...`

### Max Entries Per Minute

- Approx lines: 6645-6656
- Control flow: `break`
- Log: `ENTRY_RATE_LIMIT_BLOCK` rate-limited globally.
- NBAS relation: before `SIGNAL_READY`; cannot explain post-signal disappearance.

## Most Likely Explanations for Current NBAS Pattern

Ranked by likelihood for `SIGNAL_READY` exists, selected/passed filters, but no `BUY_ORDER_SENT`.

1. 90%: Exception or interruption between `ib.placeOrder` and `BUY_ORDER_SENT` lifecycle.
   - The order dispatch and persistence block has no local stage logs.
   - Any exception after `SIGNAL_READY` and before line 6880 creates the exact NBAS signature.

2. 80%: `record_lifecycle_with_formal("BUY_ORDER_SENT")` or direct `trade.order.orderId` fails.
   - `BUY_ORDER_SENT` is the first definitive post-dispatch lifecycle event.
   - It is emitted after IBKR submission, SQLite writes, submission counter, managed-position persistence, and payload assembly.

3. 75%: SQLite writer `safe_sqlite_call` blocks/raises after IBKR submission but before lifecycle log.
   - Previous live logs showed SQLite ACK timeouts and queue pressure.
   - The code emits no `ENTRY_ORDER_IBKR_SUBMITTED` marker before SQLite writes.

4. 70%: `persist_managed_positions` fails before `BUY_ORDER_SENT`.
   - Managed position persistence sits between IBKR submission and final order log.
   - No local catch/log.

5. 60%: `ib.placeOrder` itself throws.
   - No local catch/log.
   - Would likely cause process-level exception, but not symbol-specific evidence.

6. 35%: Missing/blank `trade.order.orderId` causes later failure.
   - Early access is safe, final lifecycle uses `trade.order.orderId` directly.

7. 10%: Low-score or risk-guard branch with failed logging.
   - These branches normally emit symbol-specific logs and lifecycle rows.
   - NBAS should not classify them as dispatch gap unless those writes failed.

## Recommended Diagnostics Before Changing Behavior

Do not change strategy decisions. Add only stage markers and exception logs around the existing order path.

Suggested minimal instrumentation:

1. Before `ib.placeOrder`:
   - `ENTRY_ORDER_DISPATCH_ATTEMPT`
   - Fields: `symbol`, `qty`, `price`, `score`, `ranking_position`, `candidate_age_seconds`, `ready_since`, `signal_time`.

2. Immediately after `ib.placeOrder` returns:
   - `ENTRY_ORDER_DISPATCH_RETURNED`
   - Fields: `symbol`, `orderId`, `permId`, `ibkr_status`, `trade_repr`.

3. Wrap the post-signal order block:
   - `ENTRY_ORDER_DISPATCH_EXCEPTION`
   - Field: `stage`, one of:
     - `market_order_create`
     - `ib_place_order`
     - `entry_metadata_build`
     - `runtime_order_map`
     - `sqlite_upsert_trade`
     - `sqlite_upsert_order`
     - `entry_submission_counter`
     - `managed_position_persist`
     - `buy_order_lifecycle_record`
     - `paper_buy_print`

4. Move one durable dispatch marker before SQLite and managed-position persistence:
   - If `ib.placeOrder` returns, record `ENTRY_ORDER_IBKR_SUBMITTED` before `safe_sqlite_call`.
   - This distinguishes "IBKR accepted/returned trade object but local persistence failed" from "no IBKR dispatch".

5. Add explicit branch log for max per-cycle break:
   - `ENTRY_CYCLE_LIMIT_REACHED`
   - Useful for candidates without `SIGNAL_READY`, not the current post-signal issue.

## Conclusion

For the current NBAS cases, the symbol has already passed the early runtime path. `SIGNAL_READY` proves the candidate reached the selected entry-evaluation branch. The missing evidence is concentrated in one uninstrumented region:

`SIGNAL_READY` -> low-score/risk guard -> `ib.placeOrder` -> SQLite/order persistence -> managed position persistence -> `BUY_ORDER_SENT` lifecycle -> `PAPER BUY SENT`.

The current code logs low-score and risk-guard exits. It does not log pre-dispatch attempt, post-`placeOrder` return, or per-stage exceptions in the order submission/persistence block. That is the exact blind spot that can produce `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`.
