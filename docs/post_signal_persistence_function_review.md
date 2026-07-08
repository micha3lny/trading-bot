# Post-Signal Persistence Function Review

Date reviewed: 2026-07-08

Scope: read-only review of the three persistence/logging areas that sit on the entry path after `SIGNAL_READY` and before `BUY_ORDER_SENT` / `PAPER BUY SENT`.

Operational context from NBAS:

- `SIGNAL_READY` exists.
- Candidate appears selected / passed final filters.
- `BUY_ORDER_SENT` never appears.
- `PAPER BUY SENT` never appears.
- The suspected gap is between order dispatch and lifecycle/order persistence.

The current code now has explicit entry-dispatch diagnostics around this path, but this review documents the underlying function behavior and failure semantics.

## Entry Path Ordering

Source: `src/live_trading/v67_live_top100_expansion_paper_trader.py`

Relevant order in the BUY path:

1. `ib.placeOrder(...)` returns.
2. `ENTRY_ORDER_IBKR_SUBMITTED` diagnostic is emitted.
3. Runtime order metadata is written into `runtime_state`.
4. `safe_sqlite_call(..., "upsert_trade", ...)`.
5. `safe_sqlite_call(..., "upsert_order", ...)`.
6. `persist_managed_positions(...)`.
7. `record_lifecycle_with_formal(..., "BUY_ORDER_SENT", ...)`.
8. `print("PAPER BUY SENT ...")`.
9. `state.signal_sent = True`.

Therefore, failures in `safe_sqlite_call` normally should not prevent `BUY_ORDER_SENT`, but failures in `persist_managed_positions` or `record_lifecycle_with_formal` can.

## 1. `safe_sqlite_call`

Source file: `src/live_trading/storage/sqlite_store.py`

Function name: `safe_sqlite_call`

Approximate lines: 3417-3462

Related queue implementation: `SQLiteWriteQueue.call`, lines 3293-3384

### What It Does

`safe_sqlite_call` is a safety wrapper for SQLite operations. It supports two store types:

- `SQLiteWriteQueue`: enqueue a write into the single-writer queue and optionally wait for acknowledgement.
- Direct `SQLiteRuntimeStore`: call the method directly with retry/backoff for SQLite busy/locked errors.

For queued writes, `SQLiteWriteQueue.call` determines priority, acknowledgement behavior, timeout, queue coalescing, queue depth metrics, timeout logs, and eventual return value.

### Can It Throw?

Mostly no, by design.

It re-raises only:

- `KeyboardInterrupt`
- `SystemExit`
- a queued request exception after the request finishes and `request.exception is not None`

For most normal runtime errors, including SQLite write failures, queue full, ACK timeout, and exhausted busy retries, it logs and returns `None`.

### Does It Catch Exceptions?

Yes.

- For `SQLiteWriteQueue`, `safe_sqlite_call` catches general exceptions around `store.call(...)`, logs `SQLITE_WRITE_FAILED`, and returns `None`.
- For direct store calls, it retries busy errors and then logs `SQLITE_WRITE_FAILED` and returns `None`.

### Does It Swallow Exceptions Silently?

Not silently: it usually emits `SQLITE_WRITE_FAILED`, `SQLITE_BUSY_RETRY`, `SQLITE_WRITE_DROPPED`, or `SQLITE_WRITE_ACK_TIMEOUT`.

But it converts failures into `None`, so callers must check return status if they need durable confirmation. Many callers treat it as best-effort.

### Can It Block Or Timeout?

Yes.

For `SQLiteWriteQueue`, critical writes wait for ACK by default. The default critical timeout is configured by `TRADING_BOT_SQLITE_WRITER_CRITICAL_TIMEOUT_SECONDS`, currently defaulting to 2 seconds globally, with method-specific overrides such as:

- `upsert_trade`: default 8 seconds
- `upsert_order`: default 8 seconds
- `upsert_position`: default 8 seconds
- `upsert_execution`: default 8 seconds
- `finalize_pending_trades`: default 15 seconds

If the ACK wait times out, `SQLiteWriteQueue.call` returns `None`; the write may still complete later in the background.

For direct `SQLiteRuntimeStore`, it can block during the actual SQLite call and during busy-retry sleeps. Busy timeout is configured at the SQLite connection level and retries are bounded by `TRADING_BOT_SQLITE_LOCK_RETRY_ATTEMPTS`.

### Does It Return Success/Failure Status?

Yes, but loosely:

- Returns the underlying store method result on success.
- Returns `"queued"` for best-effort queued writes without ACK.
- Returns `None` for skipped/coalesced writes, ACK timeout, queue full, direct write failure, store missing, or exhausted busy retry.

This means `None` is ambiguous.

### Does The Entry Caller Check Status?

Current entry path checks `None` for:

- `upsert_trade`
- `upsert_order`

When `None` is returned and a SQLite store exists, it emits:

- `ENTRY_ORDER_SQLITE_PERSIST_FAILED stage=sqlite_upsert_trade`
- `ENTRY_ORDER_SQLITE_PERSIST_FAILED stage=sqlite_upsert_order`

The entry path then continues toward managed position persistence and `BUY_ORDER_SENT`.

### Logs/Events Emitted Today

SQLite layer:

- `SQLITE_WRITE_ACK_TIMEOUT`
- `SQLITE_WRITE_FAILED`
- `SQLITE_BUSY_RETRY`
- `SQLITE_WRITE_DROPPED`
- `SQLITE_SLOW_WRITE`
- `SQLITE_WRITER_CRASH`
- `SQLITE_CALL_INTERRUPTED`

Entry path:

- `ENTRY_ORDER_SQLITE_PERSIST_FAILED`
- `ENTRY_ORDER_DISPATCH_EXCEPTION stage=sqlite_upsert_trade` only if an exception escapes despite the wrapper
- `ENTRY_ORDER_DISPATCH_EXCEPTION stage=sqlite_upsert_order` only if an exception escapes despite the wrapper

### Can Failure Prevent `BUY_ORDER_SENT` / `PAPER BUY SENT`?

Usually no.

Because `safe_sqlite_call` normally returns `None` on failure and the caller continues, a SQLite ACK timeout on `upsert_trade` or `upsert_order` should not by itself prevent `BUY_ORDER_SENT` or `PAPER BUY SENT`.

However, it can delay the path by up to the ACK timeout per call. In the current order path, `upsert_trade` plus `upsert_order` can add up to roughly 16 seconds of delay with default method timeouts if the writer is backed up.

It can prevent later logs only if:

- the call raises an exception instead of returning `None`, or
- the main loop is interrupted/shut down during the wait window, or
- the write queue/store call blocks unexpectedly longer than configured.

### Can It Explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?

Likelihood: MEDIUM-LOW.

Reasoning:

- It can delay post-signal logging.
- It can create a state where `ib.placeOrder` succeeded but SQLite did not immediately persist trade/order rows.
- But because the caller now continues after `None`, normal ACK timeout alone should not eliminate `BUY_ORDER_SENT`.
- Historical NBAS gaps without diagnostics could have been ambiguous because SQLite timeouts were not stage-specific at the entry path.

### Suggested Minimal Diagnostic Log Names

Already present or recommended:

- `ENTRY_ORDER_SQLITE_PERSIST_FAILED`
- `ENTRY_ORDER_DISPATCH_EXCEPTION stage=sqlite_upsert_trade`
- `ENTRY_ORDER_DISPATCH_EXCEPTION stage=sqlite_upsert_order`
- `SQLITE_WRITE_ACK_TIMEOUT`
- `SQLITE_SLOW_WRITE`

Useful additional diagnostic, if needed later:

- `ENTRY_ORDER_SQLITE_PERSIST_DELAYED` when ACK timeout occurs but order flow continues.

## 2. `record_lifecycle_with_formal`

Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`

Function name: `record_lifecycle_with_formal`

Approximate lines: 1129-1179

Related functions:

- `record_lifecycle`, lines 1018-1048
- `record_formal_lifecycle`, lines 1087-1126

### What It Does

`record_lifecycle_with_formal` records a legacy lifecycle row first, then optionally records a formal JSONL lifecycle event.

For `BUY_ORDER_SENT`, it maps the legacy event to `LifecycleEventType.ENTRY_ORDER_SUBMITTED`, sets order/position state metadata, and writes:

- legacy `trade_lifecycle.csv` row via `record_lifecycle`
- SQLite runtime event via `record_lifecycle` -> `safe_sqlite_call(..., "record_runtime_event", ...)`
- formal `order_lifecycle.jsonl` event via `record_formal_lifecycle`

### Does Legacy `record_lifecycle` Run Before Formal Recording?

Yes.

Line 1130 calls `record_lifecycle(...)` before `formal_event_type_for_legacy_event(...)`.

### If Legacy `record_lifecycle` Throws, Does Formal Recording Still Happen?

No.

There is no `try/except` around the initial `record_lifecycle(...)` call inside `record_lifecycle_with_formal`. If legacy CSV recording or pre-CSV JSON serialization throws, the function exits immediately and formal recording is never attempted.

### Can It Throw?

Yes.

The formal recorder catches its own exceptions and prints `formal_lifecycle_record_error`, but the legacy recorder can throw before formal recording.

Potential legacy throw points:

- `json.dumps(raw_json, ...)` if raw payload contains an object that `default=str` cannot handle in practice.
- `recorder.path(...)` if recorder/session path is invalid.
- `append_dict_csv(...)` file open/write/header/write row.
- filesystem errors such as permission denied, disk full, file descriptor issues, path issues.
- CSV serialization issues.

The SQLite runtime event inside `record_lifecycle` uses `safe_sqlite_call`, so SQLite failures there generally return `None` and should not throw.

### Does It Catch Exceptions?

`record_lifecycle_with_formal` itself does not catch exceptions from `record_lifecycle`.

`record_formal_lifecycle` does catch and only prints:

- `formal_lifecycle_record_error event=... symbol=... error=...`

The entry path currently wraps the `record_lifecycle_with_formal("BUY_ORDER_SENT", ...)` call and emits:

- `ENTRY_ORDER_LIFECYCLE_RECORD_FAILED`
- then re-raises

### Does It Swallow Exceptions Silently?

Formal lifecycle failures are swallowed after a print, so formal JSONL failure alone does not break entry.

Legacy lifecycle failures are not swallowed.

### Can It Block Or Timeout?

Yes, via file I/O.

`append_dict_csv` opens and appends to `trade_lifecycle.csv` synchronously. There is no explicit timeout. On a slow disk, full disk, filesystem stall, or locked/network-like filesystem behavior, it can block the main loop.

The SQLite runtime event call inside `record_lifecycle` may also wait through `safe_sqlite_call`, but `record_runtime_event` is categorized as best-effort, so it should not normally wait for a long ACK.

### Does It Return Success/Failure Status?

No.

It returns `None` on success. Failure is represented by an exception from the legacy path, or a printed warning from formal path.

### Does Caller Check Status?

There is no status to check.

The entry path wraps it in `try/except` and logs `ENTRY_ORDER_LIFECYCLE_RECORD_FAILED`, then re-raises.

### Logs/Events Emitted Today

On success:

- `BUY_ORDER_SENT` in `trade_lifecycle.csv`
- `BUY_ORDER_SENT` runtime event
- formal `ENTRY_ORDER_SUBMITTED` in `order_lifecycle.jsonl`

On formal-only failure:

- `formal_lifecycle_record_error`

On legacy failure in the entry path:

- `ENTRY_ORDER_LIFECYCLE_RECORD_FAILED`
- `ENTRY_ORDER_DISPATCH_EXCEPTION stage=buy_order_lifecycle_record`

### Can Failure Prevent `BUY_ORDER_SENT` / `PAPER BUY SENT`?

Yes.

This function is called before `PAPER BUY SENT`. If legacy `record_lifecycle` fails while recording `BUY_ORDER_SENT`, then:

- `BUY_ORDER_SENT` may be absent or partially absent.
- formal lifecycle may not be attempted.
- `PAPER BUY SENT` will not print because the exception is re-raised before the print.

This is one of the strongest explanations for historical cases where order dispatch occurred but both `BUY_ORDER_SENT` and `PAPER BUY SENT` were absent.

### Can It Explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?

Likelihood: HIGH.

Reasoning:

- It sits exactly between successful entry persistence/managed position handling and `PAPER BUY SENT`.
- Legacy lifecycle runs before formal lifecycle and is not internally protected.
- Any exception here directly prevents both `BUY_ORDER_SENT` and `PAPER BUY SENT`.
- Before explicit diagnostics, this failure could look like the candidate vanished after `SIGNAL_READY`.

### Suggested Minimal Diagnostic Log Names

Already present or recommended:

- `ENTRY_ORDER_LIFECYCLE_RECORD_FAILED`
- `ENTRY_ORDER_DISPATCH_EXCEPTION stage=buy_order_lifecycle_record`

Potential later hardening log:

- `ENTRY_ORDER_LIFECYCLE_LEGACY_FAILED_FORMAL_ATTEMPTED` if legacy and formal are split so formal can still record.

## 3. `persist_managed_positions`

Source file: `src/live_trading/v67_live_top100_expansion_paper_trader.py`

Function name: `persist_managed_positions`

Approximate lines: 1392-1448

Entry-path call site: lines 6955-6989

### What It Does

`persist_managed_positions` snapshots current managed positions to disk and SQLite:

1. Copies `positions.items()` into `position_items`.
2. Builds active payloads with market-price/excursion metadata.
3. Writes `managed_positions.json` synchronously.
4. For every position in the copied list, calls `safe_sqlite_call(..., "upsert_position", ...)`.

### Can File Write / Serialization Fail?

Yes.

Potential failure points before SQLite:

- `managed_position_payload(...)` can mutate position excursion state and may fail if unexpected data shapes appear.
- `json.dumps(payload, indent=2, ensure_ascii=False, default=str)` can still fail for pathological objects despite `default=str`.
- `recorder.path("managed_positions.json").write_text(...)` can fail on disk full, permission denied, path issue, interrupted I/O, or filesystem error.

Potential SQLite failures:

- `safe_sqlite_call(..., "upsert_position", ...)` usually returns `None` on failure or timeout.
- The function does not check that return value.

### Is It Called Before `BUY_ORDER_SENT`?

Yes.

In the entry path, the newly submitted position is inserted into `managed_positions`, then `persist_managed_positions(...)` is called before `record_lifecycle_with_formal("BUY_ORDER_SENT", ...)`.

### Can It Throw?

Yes.

The synchronous JSON write is the most obvious throw point. The function itself has no `try/except`.

SQLite failures usually do not throw because `safe_sqlite_call` catches most exceptions and returns `None`, but a queued request can still raise if `SQLiteWriteQueue.call` re-raises `request.exception`.

### Does It Catch Exceptions?

No.

The entry path call site catches exceptions around `persist_managed_positions`, logs `ENTRY_MANAGED_POSITION_PERSIST_FAILED`, and re-raises.

### Does It Swallow Exceptions Silently?

The function itself does not swallow Python exceptions.

However, it does ignore `safe_sqlite_call` return values for each `upsert_position`. A SQLite position-persist failure or ACK timeout can be logged at the SQLite layer but is not surfaced by this function.

### Can It Block Or Timeout?

Yes.

- JSON file write is synchronous and has no explicit timeout.
- `upsert_position` is a critical SQLite method and waits for ACK by default. Its method-specific timeout defaults to 8 seconds.
- Because the function iterates all managed positions, repeated `upsert_position` calls can add significant delay when the queue is blocked.

### Does It Return Success/Failure Status?

No.

It returns `None` on success. Exceptions signal hard failures. SQLite `None` results are ignored.

### Does Caller Check Status?

There is no status to check.

The entry path wraps exceptions and logs `ENTRY_MANAGED_POSITION_PERSIST_FAILED`, then re-raises.

### Logs/Events Emitted Today

From `persist_managed_positions` itself:

- None.

From SQLite layer:

- `SQLITE_WRITE_ACK_TIMEOUT method=upsert_position`
- `SQLITE_WRITE_FAILED method=upsert_position`
- `SQLITE_BUSY_RETRY method=upsert_position`
- `SQLITE_WRITE_DROPPED method=upsert_position`
- `SQLITE_SLOW_WRITE method=upsert_position`

From entry path on exception:

- `ENTRY_MANAGED_POSITION_PERSIST_FAILED`
- `ENTRY_ORDER_DISPATCH_EXCEPTION stage=managed_position_persist`

### Can Failure Prevent `BUY_ORDER_SENT` / `PAPER BUY SENT`?

Yes.

If the JSON write or payload construction throws, the entry path re-raises before `BUY_ORDER_SENT` and before `PAPER BUY SENT`.

If SQLite `upsert_position` only times out and returns `None`, it does not currently prevent `BUY_ORDER_SENT`; it only delays the path and leaves a potentially missing/stale position row.

### Can It Explain `missing_after_SIGNAL_READY_before_BUY_ORDER_SENT`?

Likelihood: HIGH-MEDIUM.

Reasoning:

- It is called before `BUY_ORDER_SENT`.
- It performs synchronous disk I/O.
- It performs one critical SQLite write per managed position, not just the new symbol.
- Any hard exception prevents `BUY_ORDER_SENT`.
- ACK timeout alone should not prevent `BUY_ORDER_SENT`, but can delay and can interact with shutdown/restart timing.

### Suggested Minimal Diagnostic Log Names

Already present or recommended:

- `ENTRY_MANAGED_POSITION_PERSIST_FAILED`
- `ENTRY_ORDER_DISPATCH_EXCEPTION stage=managed_position_persist`

Useful additional diagnostic, if needed later:

- `ENTRY_MANAGED_POSITION_SQLITE_UPSERT_FAILED`
- `ENTRY_MANAGED_POSITION_FILE_WRITE_STARTED`
- `ENTRY_MANAGED_POSITION_FILE_WRITE_DONE`

## Focus Question Answers

### Can `safe_sqlite_call` block long enough to delay or prevent `BUY_ORDER_SENT`?

It can delay. It should not usually prevent.

Queued critical calls wait for method-specific ACK timeout, and `upsert_trade` / `upsert_order` default to 8 seconds each. Direct SQLite calls can block during SQLite busy timeout and retry sleeps. But failures normally return `None`, so caller can continue.

### Does `safe_sqlite_call` raise on timeout or only log?

For queued ACK timeout, it logs `SQLITE_WRITE_ACK_TIMEOUT` only when queue/dropped conditions justify it, increments timeout counters, and returns `None`.

It does not raise on ACK timeout.

### Does caller continue after timeout?

For entry `upsert_trade` and `upsert_order`, yes. The caller logs `ENTRY_ORDER_SQLITE_PERSIST_FAILED` and continues.

For `persist_managed_positions`, SQLite `upsert_position` return values are ignored, so it also continues after `None`.

### Can SQLite ACK timeout create a state where `ib.placeOrder` succeeded but `BUY_ORDER_SENT` is never logged?

By itself, usually no.

It can create a delayed/incomplete SQLite state after `ib.placeOrder`, but the code should proceed to `BUY_ORDER_SENT` unless:

- timeout delay overlaps with shutdown/interruption,
- an unexpected exception escapes,
- later `persist_managed_positions` or `record_lifecycle_with_formal` fails.

### Does legacy `record_lifecycle` run before formal recording?

Yes.

### If legacy `record_lifecycle` throws, does formal recording still happen?

No.

### Can `BUY_ORDER_SENT` recording crash before `PAPER BUY SENT`?

Yes.

The `BUY_ORDER_SENT` lifecycle call happens before the console print. A legacy lifecycle failure prevents the print.

### Should `BUY_ORDER_SENT` recording be wrapped so it cannot break the entry path?

From a diagnostics/reliability perspective, yes. That would prevent a local recorder failure from interrupting an already-submitted broker order.

This review does not implement that change because the request was read-only.

### Can `persist_managed_positions` file write/serialization fail?

Yes.

### Is `persist_managed_positions` called before `BUY_ORDER_SENT`?

Yes.

### Can it raise and prevent `BUY_ORDER_SENT`?

Yes.

### Should failure be logged as `ENTRY_MANAGED_POSITION_PERSIST_FAILED`?

Yes, and the current entry path already does this.

## Risk Ranking

| Function / area | Likelihood | Rationale |
|---|---|---|
| `record_lifecycle_with_formal` | HIGH | Called immediately before `PAPER BUY SENT`; legacy lifecycle is unprotected; legacy failure prevents formal record and print. Exact fit for missing `BUY_ORDER_SENT` / missing `PAPER BUY SENT`. |
| `persist_managed_positions` | HIGH-MEDIUM | Called before `BUY_ORDER_SENT`; synchronous JSON write can throw; per-position SQLite writes can delay significantly. Exact fit if exception occurs; timeout alone mostly delay-only. |
| `safe_sqlite_call` | MEDIUM-LOW | Can delay and can hide persistence failure as `None`, but entry caller generally continues. More likely to cause missing SQLite rows or delayed state than complete absence of `BUY_ORDER_SENT`, unless combined with shutdown/interruption or escaped exception. |

## Most Important Evidence To Look For In New Logs

If a future case has:

- `ENTRY_ORDER_IBKR_SUBMITTED` present
- no `BUY_ORDER_SENT`

then check the next event for that symbol:

1. `ENTRY_MANAGED_POSITION_PERSIST_FAILED`: managed position JSON/file/payload failure.
2. `ENTRY_ORDER_LIFECYCLE_RECORD_FAILED`: lifecycle CSV/formal recorder failure.
3. `ENTRY_ORDER_SQLITE_PERSIST_FAILED`: SQLite delayed/failed but should not by itself stop the flow.
4. `ENTRY_ORDER_DISPATCH_EXCEPTION stage=...`: exact stage where the path aborted.

If even `ENTRY_ORDER_DISPATCH_ATTEMPT` is absent after `SIGNAL_READY`, then the missing path is before order construction and outside the three functions reviewed here.
