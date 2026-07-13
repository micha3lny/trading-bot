# Runtime State Growth Audit

## Scope

This is a read-only audit of long-lived mutable state in `src/live_trading/v67_live_top100_expansion_paper_trader.py`, plus SQLite writer state in `src/live_trading/storage/sqlite_store.py`.

The implementation added observability only:

- `RUNTIME_STATE_GROWTH_SNAPSHOT`
- `RUNTIME_STATE_SESSION_DIFF`

No strategy thresholds, entry rules, exit rules, sizing, or subscription behavior were intentionally changed.

## Instrumentation

`RUNTIME_STATE_GROWTH_SNAPSHOT` is emitted at:

- startup after in-memory initialization,
- startup reconciliation completion,
- session boundary / new trading session,
- before and after Top100 reload,
- session end,
- every 15 minutes during an active session.

`RUNTIME_STATE_SESSION_DIFF` is emitted at session boundaries and session end. It compares the current metrics with the session-start baseline and lists structures that changed.

Each snapshot includes current counters plus a `delta_json` object with current/startup/session-start values and deltas for key growth counters.

## Findings Ranked By Impact

### 1. `SymbolState` Can Carry Session-Sensitive Fields

Source:

- `SymbolState` fields: `src/live_trading/v67_live_top100_expansion_paper_trader.py:76`
- live updates: `update_state(...)`
- rebuild from candles: `rebuild_symbol_states_from_1m_candles(...)`
- Top100 reload state pruning: `reload_top100_universe_if_requested(...)`

Classification:

- expected to track current Top100 only,
- potential stale-state leak across sessions.

Fields at risk:

- `signal_sent`
- `ready_since_ts`
- `ready_since_utc`
- `first_price`
- `open_price`
- `first_5m_high`
- `first_15m_high`
- `or_high`
- `or_low`
- `first_seen_utc`
- `last_live_update_utc`
- `bars`

Restart clears it:

- yes, process restart recreates `states = {symbol: SymbolState(symbol=symbol) ...}`.

Session boundary clears it:

- not generally. Current code emits diagnostics at session boundary but does not forcibly reset every `SymbolState`.

Top100 reload reconciles it:

- partially. Removed symbols are pruned if they are not active positions; selected symbols use `states.setdefault(...)`, preserving existing state for retained symbols.

Could affect entries:

- yes. `signal_sent=True` suppresses new candidate creation. Stale first/open/first5/first15/OR values can alter `compute_live_safe_features(...)`.

Could explain better behavior after restart:

- yes.

Mechanism:

- restart clears in-memory state, so stale `signal_sent`, stale `ready_since`, old first5/first15/OR, or old bars disappear. A continuous process may preserve them.

New evidence:

- `state_with_old_session_date_count`
- `state_with_old_first_seen_count`
- `state_with_old_last_live_update_count`
- `state_with_first5_from_previous_session_count`
- `state_with_first15_from_previous_session_count`
- `state_with_open_price_from_previous_session_count`
- `signal_sent_count`

## 2. Top100 / Contract / Ticker Reconciliation Can Diverge

Source:

- startup subscriptions around `src/live_trading/v67_live_top100_expansion_paper_trader.py:6941`
- Top100 reload around `reload_top100_universe_if_requested(...)`
- reconnect resubscribe around `handle_ibkr_disconnect_and_recover(...)`

Classification:

- expected to track current Top100 only,
- potential stale subscription leak.

Restart clears it:

- yes, contract/ticker dictionaries start empty.

Session boundary clears it:

- no.

Top100 reload reconciles it:

- mostly. It cancels symbols removed from the selected subscription set and subscribes new ones. Active carried positions can stay subscribed even if not in Top100.

Could affect entries:

- yes. Removed symbols can consume subscription capacity. Added symbols without ticker/state cannot produce signals.

Could explain better behavior after restart:

- yes.

Mechanism:

- startup builds contract/ticker/state from the current Top100. A long-running process depends on reload/reconnect correctness.

New evidence:

- `desired_top100_count`
- `active_contract_count`
- `active_ticker_count`
- `active_subscription_count`
- `symbols_removed_from_top100_but_still_subscribed`
- `symbols_in_top100_without_subscription`
- `symbols_in_top100_without_ticker`
- `symbols_in_top100_without_state`
- `subscription_capacity_used_pct`
- `reqMktData_total_count`
- `cancelMktData_total_count`

## 3. Runtime Order Maps Can Grow

Source:

- `runtime_state["entry_order_by_order_id"]` initialized near runtime state setup.
- mutated after `ENTRY_ORDER_IBKR_SUBMITTED`.
- rejection processing reads it in the IBKR error handler.

Classification:

- expected to hold active/recent entry order metadata,
- potential unbounded growth.

Restart clears it:

- yes.

Session boundary clears it:

- no explicit full clear observed.

Top100 reload reconciles it:

- no.

Could affect entries:

- indirect. It can affect memory, diagnostics, rejection attribution, and stale order metadata.

Could explain better behavior after restart:

- uncertain.

New evidence:

- `runtime_entry_order_map_count`
- `runtime_exit_order_map_count`
- `pending_entry_orders_count`
- `orders_with_no_recent_update_count`

## 4. Execution / Fill Dedupe Caches Grow

Source:

- `seen_fills` loaded at startup.
- `entry_rejection_processed`
- `fill_diagnostic_execution_ids`
- `partial_fill_states`

Classification:

- expected to grow during the process lifetime,
- potential unbounded growth.

Restart clears it:

- partly. `seen_fills` is loaded from recorder/history; in-memory transient sets reset.

Session boundary clears it:

- not generally.

Could affect entries:

- mostly indirect through latency/memory. Incorrect stale dedupe could affect fill diagnostics/reconciliation, but not primary signal generation.

Could explain better behavior after restart:

- uncertain.

New evidence:

- `executions_seen_cache_count`
- `duplicate_execution_guard_count`
- `rejection_reason_cache_count`
- `fills_buffer_count`

## 5. SQLite Writer Queue Can Accumulate Backlog

Source:

- `SQLiteWriteQueue` in `src/live_trading/storage/sqlite_store.py:3207`
- queue status fields around `src/live_trading/storage/sqlite_store.py:3237`

Classification:

- expected to persist for process lifetime,
- potential latency/backpressure growth.

Restart clears it:

- yes, the queue thread and queue state restart from zero.

Session boundary clears it:

- no.

Could affect entries:

- yes, indirectly. Critical writes may wait for ACK. If the writer is stuck or slow, order/fill/reconcile paths can be delayed.

Could explain better behavior after restart:

- yes.

Mechanism:

- restart clears queue backlog, coalesced keys, timeout counters, and current writer operation.

New evidence:

- `sqlite_queue_depth`
- `sqlite_max_queue_depth`
- `sqlite_oldest_queued_age_seconds`
- `sqlite_ack_timeouts_total`
- `sqlite_ack_timeouts_delta_since_session_start`
- `sqlite_current_write_method`
- `sqlite_current_write_duration_seconds`
- `pending_sqlite_requests_count`

## 6. IBKR Callback Registration Could Duplicate

Source:

- commission handler installed at startup and reconnect.
- order rejection handler guarded by `_v67_order_rejection_handler_installed`.
- reconnect path calls `install_commission_report_handler(...)`.

Classification:

- expected to remain constant,
- duplicate callback risk if installer is not idempotent.

Restart clears it:

- yes.

Session boundary clears it:

- no.

Could affect entries:

- yes indirectly. Duplicate callbacks can duplicate processing, increase SQLite load, and create repeated diagnostics.

Could explain better behavior after restart:

- yes, if callback counts grow after reconnects.

New evidence:

- `callback_count_pendingTickersEvent`
- `callback_count_orderStatusEvent`
- `callback_count_execDetailsEvent`
- `callback_count_commissionReportEvent`
- `callback_count_errorEvent`
- `callback_count_disconnectedEvent`
- `callback_duplicate_registration_risk`

## 7. Rate-Limited Log State Grows By Bucket/Key

Source:

- `runtime_rate_limited_log(...)`
- `runtime_state["rate_limited_log_state"]`

Classification:

- expected to persist within windows,
- potential growth if many buckets/reasons/symbols are introduced.

Restart clears it:

- yes.

Session boundary clears it:

- no.

Could affect entries:

- low. It can hide diagnostics if stale keys suppress logs, but current code resets keys by window.

Could explain better behavior after restart:

- unlikely for trading decisions, possible for observability.

New evidence:

- `rate_limit_state_count`
- `log_throttle_key_count`

## 8. History Collector / Daily Top100 Processes Can Compete For Resources

Source:

- `history_collector_queue`
- `history_collector_process`
- `daily_top100_process`
- overnight automation and startup repair paths.

Classification:

- background tasks expected to be low frequency,
- potential CPU/IBKR/SQLite contention.

Restart clears it:

- yes.

Session boundary clears it:

- no.

Could affect entries:

- yes indirectly through CPU, IBKR calls, and SQLite write pressure.

Could explain better behavior after restart:

- yes, if long-running background tasks accumulate or overlap with market open.

New evidence:

- `pending_async_task_count`
- process memory/thread/fd counters
- SQLite queue counters

## Suspected Cross-Session Leaks Ranked

1. `SymbolState` stale `signal_sent` / first5 / first15 / OR / first_seen: high impact.
2. Top100 reload leaving desired symbols without subscription/ticker/state: high impact.
3. SQLite writer backlog after long uptime: medium-high impact.
4. Duplicate IBKR callbacks after reconnect: medium-high impact if callback counts grow above 1.
5. Old Top100 subscriptions consuming subscription capacity: medium impact.
6. Runtime order/rejection/fill maps growing: medium impact, mostly latency/metadata risk.
7. Rate limiter/log throttle state: low direct trading impact, medium diagnostics impact.
8. Background collector/top100 tasks competing during session: medium operational impact.

## How To Use The New Logs

During a live session:

```bash
journalctl -u v67-trader -o cat --no-pager | grep 'RUNTIME_STATE_GROWTH_SNAPSHOT'
journalctl -u v67-trader -o cat --no-pager | grep 'RUNTIME_STATE_SESSION_DIFF'
```

Key red flags:

- `state_with_old_session_date_count > 0`
- `state_with_first5_from_previous_session_count > 0`
- `signal_sent_count` unexpectedly high before new entries
- `symbols_in_top100_without_ticker > 0`
- `symbols_removed_from_top100_but_still_subscribed > 0`
- callback counts greater than expected
- `sqlite_queue_depth` or `sqlite_oldest_queued_age_seconds` increasing
- `rate_limit_state_count` or `total_cached_candles` growing monotonically across sessions

## Next Step

Do not change reset behavior yet. First collect one no-restart session and one fresh-restart session with these logs, then compare:

- `RUNTIME_STATE_GROWTH_SNAPSHOT reason=session_boundary`
- `RUNTIME_STATE_GROWTH_SNAPSHOT reason=periodic_15m`
- `RUNTIME_STATE_SESSION_DIFF`

If stale state or subscription mismatch appears, the next safe fix should be narrowly targeted to that structure only.
