# PRE_SIGNAL_RUNTIME_SNAPSHOT Review

## Where The Snapshot Is Emitted

`PRE_SIGNAL_RUNTIME_SNAPSHOT` is emitted in `src/live_trading/v67_live_top100_expansion_paper_trader.py` inside the main runtime scan.

The snapshot is built after:

1. `snapshot_from_ticker(...)` returned a usable price,
2. `update_state(...)` updated `SymbolState`,
3. `compute_live_safe_features(...)` calculated readiness and live score,
4. active-position, Top100/entry-symbol, and ineligible-symbol flags were known,
5. `ready_since` was initialized for symbols that would become candidates.

It is emitted after `ranking_position_by_symbol` is calculated and before:

1. `candidate_rejection_reasons`,
2. `ordered_entry_candidates`,
3. `SIGNAL_READY`,
4. any entry order dispatch.

This location guarantees the next runtime logs can answer why runtime did not produce `SIGNAL_READY`:

- If a symbol has no `PRE_SIGNAL_RUNTIME_SNAPSHOT`, it did not reach final pre-signal evaluation with a usable ticker price.
- If it has a snapshot, the row contains the exact runtime feature/state/decision reason before `SIGNAL_READY`.

## Fields

The event includes:

- symbol, timestamp, session_date, scan_id, ranking_position, candidate_age_seconds
- Top100 rank/score/source membership
- contract/ticker/usable price and bid/ask/spread
- state presence, signal_sent, ready, ready_since, first_seen, last_live_update
- first price / first5 / first15 initialization flags
- first_5m_high_pct, first_15m_high_pct, or_range_pct, live_entry_score
- rejection_reason, entries_blocked, entries_blocked_reason, stale_reason, already_open
- quantity, quantity_reason
- would_emit_signal_ready, signal_ready_reason

## Runtime Safety

This is observability only.

It does not change:

- strategy thresholds,
- entry filters,
- position sizing,
- order dispatch,
- exits,
- EOD behavior.

The lifecycle recording is wrapped so diagnostic failure logs `PRE_SIGNAL_RUNTIME_SNAPSHOT_RECORD_FAILED` rather than interrupting the trading loop.

## Could This Explain Better Results Immediately After Restart?

### Subscription or Ticker Refresh Gap

Answer: yes.

Mechanism: if daily Top100 refresh does not fully reconcile contracts/tickers, a symbol may never reach this snapshot. Restart rebuilds subscriptions from scratch and can mask the issue.

Evidence to look for: offline symbol has no `PRE_SIGNAL_RUNTIME_SNAPSHOT`, plus `TOP100_SUBSCRIPTION_RECONCILE` shows missing subscription or `NO_USABLE_TICKER_PRICE`.

Restart effect: restart resets/rebuilds contract and ticker maps.

### Stale SymbolState Carryover

Answer: yes.

Mechanism: `signal_sent`, `ready_since`, first price, first5/first15, or OR values may survive across session boundaries. Snapshot will show `signal_sent=1`, stale `ready_since`, or initialized values inconsistent with current session.

Evidence to look for: `PRE_SIGNAL_RUNTIME_SNAPSHOT signal_sent=1`, stale `first_seen`, stale `last_live_update`, or `signal_ready_reason=signal_sent_already_true`.

Restart effect: restart clears in-memory `SymbolState` unless restored elsewhere.

### Missing First5/First15 Initialization

Answer: uncertain.

Mechanism: if runtime subscribes late or loses early ticks, first5/first15 can stay missing and readiness never becomes true. Restart after open may or may not help depending on whether state rebuild from candles fills this context.

Evidence to look for: snapshot with `first5_initialized=0` or `first15_initialized=0` and `signal_ready_reason=first_5m_high_too_low` or `first_15m_high_too_low`.

Restart effect: restart can mask this only if startup state rebuild backfills early-session candles.

### Entries Blocked After Restart

Answer: yes for missed early signals, no for later unblocked signals.

Mechanism: if the signal would otherwise be ready but `entries_blocked=1`, snapshot will show `would_emit_signal_ready=0` and `signal_ready_reason=entries_blocked`.

Evidence to look for: snapshot with `entries_blocked=1` and `entries_blocked_reason`.

Restart effect: restart creates a cooldown/unblock boundary, so it can both cause early misses and later clear blocks.

### Spread / Price Eligibility

Answer: uncertain.

Mechanism: wide spread or missing usable price can stop readiness. Restart can help only if the issue was stale ticker state, not actual live market spread.

Evidence to look for: `NO_USABLE_TICKER_PRICE`, or snapshot with `rejection_reason=spread_too_wide`.

Restart effect: may reset stale ticker, but does not fix genuine market spread.
