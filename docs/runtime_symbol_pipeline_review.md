# Runtime Symbol Pipeline Review

Date reviewed: 2026-06-28

Scope: `src/live_trading/v67_live_top100_expansion_paper_trader.py`

Case that triggered the review: OMER on 2026-06-26.

Known facts for OMER:

- Top100 rank: 6
- Offline rules: first5 PASS, first15 PASS, OR PASS, breakout PASS
- Offline result: `offline_signal_ready=YES`
- Runtime evidence: no `SIGNAL_READY`, no `BUY_BLOCKED`, no `BUY`, no order, no execution, no runtime event, no risk event
- Current best classification: `runtime_never_processed_symbol`

The central question is not why OMER was not bought after signal generation. The evidence says OMER disappeared before symbol-specific signal generation.

## Pipeline Summary

The runtime pipeline is:

`daily_top100_latest.csv`

to `load_tradeable_top_symbols()` / `load_top100_entry_metadata()`

to startup or reload contract qualification

to `reqMktData()`

to `tickers` and `contracts`

to `SymbolState`

to `snapshot_from_ticker()`

to `update_state()`

to `compute_live_safe_features()`

to `entry_candidates`

to `SIGNAL_READY`

to `RISK_GUARD_BLOCK_ENTRY` or `BUY_ORDER_SENT`

to `PAPER BUY SENT`

The most important runtime invariant is this:

Only symbols present in `contracts` are scanned in the main feature loop.

Code:

```python
for symbol, q in contracts:
    snap = snapshot_from_ticker(symbol, tickers[symbol])
    if snap.get("price") is None:
        continue
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6345`

If a symbol is not in `contracts`, it cannot produce `SIGNAL_READY`, `BUY_BLOCKED`, risk events, orders, fills, or lifecycle rows. If it is in `contracts` but its ticker has no usable price, it also silently skips the whole feature path for that loop.

## Stage 1: Top100 CSV Loading

Purpose:

Load the trading universe from `daily_top100_latest.csv` or a reload path.

Code:

```python
def load_top_symbols(alpha_rank_csv: str, top_n: int, min_price: float | None = None) -> list[str]:
    p = Path(alpha_rank_csv)
    if not p.exists():
        raise FileNotFoundError(...)
    df = pd.read_csv(p)
    ...
    if "alpha_score" in df.columns:
        df = df.sort_values("alpha_score", ascending=False)
    if min_price is not None and "last_close" in df.columns:
        df = df[df["last_close"] >= min_price]
    return df["symbol"].dropna().drop_duplicates().head(top_n).tolist()
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:410`

Can a symbol disappear here?

Yes.

Conditions:

- Top100 file missing: raises, bot does not continue normally.
- `symbol` column missing: raises.
- Symbol duplicated later in file: first occurrence wins due to `drop_duplicates()`.
- `min_price` is configured and `last_close < min_price`: symbol is filtered out.
- Symbol appears outside `top_n` after sorting.
- If `alpha_score` exists, rows are re-sorted by `alpha_score`, not necessarily existing file order or `rank`.

Logged?

Only indirectly:

- Startup prints `Symbols loaded: N`.
- It does not print which symbols were removed by min price, duplicate handling, or `top_n`.

Can it disappear silently?

Yes. A Top100 symbol can be filtered by `min_price`, duplicate handling, or ranking truncation without symbol-specific logging.

Recommendation:

Log a startup/reload diff with `top100_rows`, `selected_symbols`, `dropped_min_price`, `dropped_outside_top_n`, and include the first 20 dropped symbols.

## Stage 2: Denylist / Ineligible Filtering

Purpose:

Remove symbols known to be non-tradeable before contract qualification.

Code:

```python
symbols = load_top_symbols(...)
ineligible = combined_ineligible_symbols(...)
for symbol in symbols:
    info = ineligible.get(symbol)
    if info:
        skipped[symbol] = info
        continue
    selected.append(symbol)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:427`

Can a symbol disappear here?

Yes.

Conditions:

- Symbol appears in persistent denylist.
- Symbol appears in runtime ineligible cache.

Logged?

Yes at startup:

```python
ENTRY_SYMBOL_INELIGIBLE_SKIPPED symbol=... source=startup_filter
STARTUP_INELIGIBLE_SYMBOLS_SKIPPED ...
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:5872`

Yes during reload:

```python
ENTRY_SYMBOL_INELIGIBLE_SKIPPED symbol=... source=top100_reload
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:4397`

Can it disappear silently?

Mostly no. It logs per symbol.

Recommendation:

Keep as-is for forensic purposes. For OMER, this is unlikely because there are no OMER ineligible logs.

## Stage 3: Top100 Entry Metadata

Purpose:

Load immutable metadata for entry records.

Code:

```python
def load_top100_entry_metadata(...):
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    if "symbol" not in df.columns:
        return {}
    ...
    df = df.dropna(subset=["symbol"]).drop_duplicates("symbol").head(top_n)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:485`

Can a symbol disappear here?

Only from metadata, not from trading.

Conditions:

- Missing CSV, unreadable CSV, missing `symbol`, min price filter, outside `top_n`.

Logged?

No.

Can it disappear silently?

Yes, metadata can be missing silently. But this does not by itself prevent a BUY.

Recommendation:

Do not use metadata presence as proof that a symbol was tradeable. It is audit metadata only.

## Stage 4: Startup Contract Qualification

Purpose:

Convert selected Top100 symbols to IBKR contracts.

Code:

```python
for symbol in symbols:
    contract = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        continue
    q = qualified[0]
    ...
    contracts.append((symbol, q))
    contract_by_symbol[symbol] = q
    tickers[symbol] = ib.reqMktData(q, "", False, False)
    print(f"Subscribed {symbol} conId={q.conId}", flush=True)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6015`

Can a symbol disappear here?

Yes.

Conditions:

- `ib.qualifyContracts(contract)` returns an empty list.
- `ib.qualifyContracts(contract)` raises an exception. In startup, this is not caught per symbol, so a raised exception likely leaves the startup `try` block and goes to outer exception handling.
- Contract metadata marks symbol ineligible.

Logged?

Partial.

- Success logs `Subscribed SYMBOL`.
- Ineligible metadata logs `ENTRY_SYMBOL_INELIGIBLE_SKIPPED`.
- Empty `qualified` has a silent `continue` and no symbol-specific log.

Can it disappear silently?

Yes. This is one of the highest-risk silent drops:

```python
qualified = ib.qualifyContracts(contract)
if not qualified:
    continue
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6017`

If OMER had `qualifyContracts()` return empty during startup, it would not enter `contracts`, would not get a ticker, would not get `SymbolState` processing, and would produce exactly the observed shape: no symbol-specific runtime evidence.

Recommendation:

Add a per-symbol startup log equivalent to reload:

`TOP100_STARTUP_CONTRACT_FAILED symbol=OMER reason=not_qualified`

## Stage 5: Startup Market Data Subscription

Purpose:

Create one IBKR market data subscription per qualified contract.

Code:

```python
tickers[symbol] = ib.reqMktData(q, "", False, False)
print(f"Subscribed {symbol} conId={q.conId}", flush=True)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6040`

Can a symbol disappear here?

Yes.

Conditions:

- `reqMktData()` raises.
- IBKR accepts request but ticker never receives bid/ask/last/close.

Logged?

- Successful request logs `Subscribed SYMBOL`.
- Startup has no per-symbol try/except around `reqMktData()`. If it raises, the whole startup block is likely interrupted.
- If ticker object exists but never has usable price, there is no per-symbol missing-market-data log.

Can it disappear silently?

Yes, if ticker object exists but no usable price arrives. The main loop silently skips:

```python
snap = snapshot_from_ticker(symbol, tickers[symbol])
if snap.get("price") is None:
    continue
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6346`

Recommendation:

Track and log per-symbol `NO_TICKER_PRICE` after N seconds when a Top100 symbol has subscription but no usable price.

## Stage 6: Top100 Freshness Gate

Purpose:

Block entries if fresh Top100 is missing/stale.

Code:

```python
if state["ready"]:
    runtime_state["top100_entries_blocked"] = False
...
runtime_state["top100_entries_blocked"] = True
runtime_state["entries_blocked_reason"] = "stale_top100"
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:3825`

Can a symbol disappear here?

No. It blocks entries globally, but symbols still remain in `contracts` if already subscribed.

Logged?

Yes:

- `DAILY_TOP100_USING_STALE_BLOCKED`
- `DAILY_TOP100_USING_STALE_ALLOWED`

Can it disappear silently?

No, but it can suppress buys. If a symbol was ready during this block, it should produce `BUY_BLOCKED` if it reaches feature-ready state.

For OMER:

Since OMER has no `BUY_BLOCKED`, this is unlikely as the direct final cause at 15:02.

## Stage 7: Daily Top100 Build Completion and Reload Trigger

Purpose:

When a daily Top100 build completes, request a runtime reload.

Code:

```python
if rc == 0:
    ...
    runtime_state["top100_reload_requested"] = True
    runtime_state["top100_reload_path"] = command.get("latest_output")
    runtime_state["top100_reload_ranking_date"] = command.get("ranking_date")
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:4216`

Can a symbol disappear here?

Indirectly.

Conditions:

- Build process never completes.
- Build fails.
- Freshness gate blocks because dated/latest files do not match.
- Reload is requested late, after missed signals already occurred.

Logged?

Yes at build level, not symbol level.

Can it disappear silently?

A symbol does not disappear silently here, but it may never enter the live universe if the reload is not triggered or is delayed.

Recommendation:

Heartbeat should include `top100_reload_requested`, `top100_reload_done_at`, and count of `entry_symbols`.

## Stage 8: Top100 Reload Selection and Subscription Cap

Purpose:

Replace the active runtime universe with current Top100 plus active positions, respecting `max_market_data_subscriptions`.

Code:

```python
active_symbols = sorted(symbol for symbol, pos in managed_positions.items() if pos.active)
max_subscriptions = ...
selected_symbols = []
for symbol in active_symbols:
    selected_symbols.append(symbol)
top100_slots = max_subscriptions - len(selected_symbols)
...
for symbol in entry_symbols:
    if max_subscriptions > 0 and len([s for s in selected_symbols if s not in active_symbol_set]) >= top100_slots:
        skipped_symbols_due_to_cap.append(symbol)
        continue
    selected_symbols.append(symbol)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:4421`

Can a symbol disappear here?

Yes.

Conditions:

- `active_symbols` consume subscription slots.
- Top100 rank is below remaining available slots.
- `max_market_data_subscriptions` is too low.

Logged?

Partially.

`TOP100_RELOAD_DONE` logs:

- `skipped_due_to_subscription_cap`
- first 20 `skipped_symbols_due_to_cap`
- `subscribed_top100`
- `subscribed_total`

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:4561`

Can it disappear silently?

For symbols beyond the first 20 skipped symbols, yes. Also, there is no per-symbol `TOP100_RELOAD_SKIPPED_DUE_TO_CAP` line.

For OMER:

Rank 6 makes cap skip unlikely unless active symbols plus earlier Top100 consumed all slots in a pathological state. But if OMER does not appear in journal at all and `subscriptions_active=100 subscription_cap_block=1`, this stage remains plausible if logs did not include OMER because it was not among first 20 skipped.

Recommendation:

Log every skipped symbol to CSV/runtime diagnostics, not just first 20 in the text log.

## Stage 9: Top100 Reload Contract Qualification

Purpose:

Qualify new contracts during reload.

Code:

```python
qualified = ib.qualifyContracts(contract)
if not qualified:
    failed_symbols.append(symbol)
    print(f"... TOP100_RELOAD_CONTRACT_FAILED symbol={symbol} reason=not_qualified")
    continue
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:4467`

Can a symbol disappear here?

Yes.

Conditions:

- IBKR returns no qualified contract.
- IBKR raises during contract qualification.
- Contract metadata marks it ineligible.

Logged?

Yes:

- `TOP100_RELOAD_CONTRACT_FAILED`
- `ENTRY_SYMBOL_INELIGIBLE_SKIPPED`

Can it disappear silently?

No for reload. This is better instrumented than startup.

For OMER:

No OMER journal lines means no reload contract failure was logged for OMER.

## Stage 10: Top100 Reload `reqMktData()`

Purpose:

Subscribe new reload symbols.

Code:

```python
print(f"... TOP100_RELOAD_REQUESTED symbol={symbol} ...")
tickers[symbol] = ib.reqMktData(contract, "", False, False)
print(f"... TOP100_RELOAD_SUBSCRIBED symbol={symbol} ...")
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:4497`

Can a symbol disappear here?

Yes.

Conditions:

- `reqMktData()` raises, including Error 101.
- Request succeeds but ticker never gets usable price.

Logged?

Yes for request/subscribed/exception.

Can it disappear silently?

Request failures are logged. Missing live price after subscription is silent later in the main loop.

For OMER:

No `TOP100_RELOAD_REQUESTED/SUBSCRIBED` lines for OMER implies either:

- OMER was already subscribed before the inspected journal window,
- OMER was not part of `subscription_symbols`,
- OMER was skipped before this stage.

## Stage 11: Reload State Pruning

Purpose:

Remove states for symbols no longer in the runtime subscription universe.

Code:

```python
for symbol in list(states):
    if symbol not in subscription_symbol_set and symbol not in active_symbols:
        states.pop(symbol, None)

contracts[:] = new_contracts
contract_by_symbol.clear()
contract_by_symbol.update(new_contract_by_symbol)
runtime_state["entry_symbols"] = set(entry_symbols)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:4527`

Can a symbol disappear here?

Yes.

Conditions:

- Symbol not selected for subscription due to cap.
- Symbol failed contract qualification.
- Symbol failed subscription.
- Symbol not active.

Logged?

Only indirectly through `TOP100_RELOAD_DONE`, `failed=N`, and cap diagnostics.

Can it disappear silently?

Yes if the reason is not among per-symbol logged cases, especially cap skip beyond the first 20 or state prune.

Recommendation:

After reload, emit `TOP100_RELOAD_SYMBOL_STATE` rows with `symbol`, `in_entry_symbols`, `in_subscription_symbols`, `in_contracts`, `in_tickers`, `state_present`, and `reason`.

## Stage 12: Main Loop Contract Scan

Purpose:

Iterate through active subscribed contracts and update feature state.

Code:

```python
for symbol, q in contracts:
    snap = snapshot_from_ticker(symbol, tickers[symbol])
    if snap.get("price") is None:
        continue
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6345`

Can a symbol disappear here?

Yes.

Conditions:

- Symbol not in `contracts`.
- Symbol in `contracts`, but no usable price in ticker.
- `tickers[symbol]` missing would raise, not silently skip.

Logged?

No per-symbol log for `price is None`.

Can it disappear silently?

Yes. This is another highest-risk silent drop.

For OMER:

This fits the observed evidence if OMER was in `contracts` but never had `last`, bid/ask midpoint, close, bid, or ask at the inspected time. It would not update `latest_snapshots`, would not update `SymbolState`, would not increment `data_count`, and would not generate symbol-specific lifecycle events.

Recommendation:

After N loops, log `NO_USABLE_TICKER_PRICE symbol=OMER subscribed=1 contract_present=1 ticker_fields=...`.

## Stage 13: `SymbolState` Creation

Purpose:

Maintain per-symbol feature memory.

Startup code:

```python
states = {symbol: SymbolState(symbol=symbol) for symbol in symbols}
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:5909`

Reload code:

```python
states.setdefault(symbol, SymbolState(symbol=symbol))
...
for symbol in list(states):
    if symbol not in subscription_symbol_set and symbol not in active_symbols:
        states.pop(symbol, None)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:4496`

Can a symbol disappear here?

Yes, on reload pruning.

Conditions:

- Not in subscription universe and not active.

Logged?

No per-symbol prune log.

Can it disappear silently?

Yes.

Recommendation:

Log pruned symbols or write them to reload diagnostics.

## Stage 14: `update_state()`

Purpose:

Convert ticker snapshots into live feature state.

Code:

```python
price = safe_float(snap.get("price"))
if price is None or price <= 0:
    return
...
if session_elapsed < 0:
    return
...
if state.first_seen_ts is None:
    state.first_price = price
...
if 0 <= session_elapsed < 5 * 60:
    state.first_5m_high = max(...)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:611`

Can a symbol disappear here?

Not completely, but feature state can remain incomplete.

Conditions:

- Price invalid: returns silently.
- Pre-market session_elapsed < 0: records `last_price` and bar, but does not initialize first/open/high/low.
- If first live tick for the symbol is late, `first_price` becomes the first seen runtime price, not the actual RTH open.

Logged?

No per-symbol log.

Can it disappear silently?

It can silently fail to build features. If first/open values are missing or first5/first15 windows were missed, downstream features remain not ready.

For OMER:

If OMER's first usable ticker arrived after the first 15 minutes, `first_5m_high` and `first_15m_high` may be `None`; however, the heartbeat `rejects` summary might show counts, but no symbol-specific event.

Recommendation:

Track `state.first_seen_utc` and `first_live_update_lag_seconds` per symbol; warn if first live update arrives after opening-range windows.

## Stage 15: Feature Calculation

Purpose:

Calculate entry readiness from live state.

Code:

```python
ready = (
    first_5m_high_pct is not None
    and first_15m_high_pct is not None
    and or_range_pct is not None
    and first_5m_high_pct >= args.min_first_5m_high_pct
    and first_15m_high_pct >= args.min_first_15m_high_pct
    and or_range_pct >= args.min_or_range_pct
    and price is not None
    and price >= args.min_price
    and (spread_bps is None or spread_bps <= args.max_spread_bps)
)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:671`

Can a symbol disappear here?

No, but it can fail readiness without symbol-specific logging.

Conditions:

- Missing first5/first15/OR values.
- Live-computed values fail thresholds.
- Current price below `min_price`.
- Spread too wide.

Logged?

Only aggregate heartbeat `rejects=[first_5m_high_too_low=...]`.

Can it disappear silently?

Yes from a per-symbol forensic perspective. There is no per-symbol `FEATURE_NOT_READY` log.

For OMER:

Offline says PASS from historical candles. Runtime may still fail if it had incomplete live state, stale subscription, late first tick, or missing first5/first15 windows.

Recommendation:

Emit per-symbol feature diagnostics at least for Top100 rank <= 10 when offline/Top100 symbols remain not ready after 15 minutes.

## Stage 16: Candidate Creation

Purpose:

Create entry candidates from ready features.

Code:

```python
if features["ready"] and not state.signal_sent and not has_active_position and entry_symbol_allowed:
    ...
    entry_candidates.append(...)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6394`

Can a symbol disappear here?

Yes.

Conditions:

- `features["ready"]` false.
- `state.signal_sent` true.
- Active position already exists.
- Symbol not in `runtime_state["entry_symbols"]`.

Logged?

Only if symbol is ineligible and ready.

Can it disappear silently?

Yes:

- `state.signal_sent` true is silent.
- `has_active_position` true is silent.
- `entry_symbol_allowed` false is silent.
- `features["ready"]` false is only aggregate logged.

For OMER:

Because there is no symbol evidence, `features["ready"]` likely never became true, or OMER was never scanned.

Recommendation:

For candidate suppression, log `ENTRY_CANDIDATE_SUPPRESSED` with reason for symbols that are Top100 rank <= 20 and offline-ready.

## Stage 17: BUY Block While Entries Blocked

Purpose:

Record that a ready symbol was blocked by global entry block.

Code:

```python
if features["ready"] and not state.signal_sent and not has_active_position and entry_symbol_allowed and entries_blocked:
    record_lifecycle_with_formal(..., "BUY_BLOCKED", symbol, ...)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6411`

Can a symbol disappear here?

No. If it reaches this point while blocked, it should create a symbol-specific `BUY_BLOCKED`.

Logged?

Yes to lifecycle, not necessarily stdout.

Can it disappear silently?

Not if `features["ready"]` is true and conditions are met.

For OMER:

The absence of `BUY_BLOCKED symbol=OMER` implies OMER did not reach this condition. Therefore global `entries_blocked=1` in heartbeat is not enough to explain OMER unless OMER was ready and somehow lifecycle write failed. There is no evidence for that.

## Stage 18: Candidate Rejection as Stale / Backfill

Purpose:

Reject ready candidates that are not fresh live signals.

Code:

```python
skip_reason = candidate_rejection_reasons.get(symbol) or ""
if skip_reason:
    record_lifecycle_with_formal(..., "STALE_OR_BACKFILL_READY_SKIPPED", symbol, ...)
    print(f"... STALE_OR_BACKFILL_READY_SKIPPED symbol={symbol} ...")
    ...
    continue
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6469`

Can a symbol disappear here?

It cannot disappear silently; it logs per symbol.

Conditions:

- `signal_source != live`
- missing signal time
- signal before last unblock
- missing live update
- live update before last unblock
- candidate age exceeded

Logged?

Yes.

Can it disappear silently?

No.

For OMER:

No `STALE_OR_BACKFILL_READY_SKIPPED symbol=OMER` means this path is unlikely.

## Stage 19: `SIGNAL_READY`

Purpose:

Log and persist a valid ready candidate before scoring/risk/order checks.

Code:

```python
record_lifecycle_with_formal(
    recorder,
    "SIGNAL_READY",
    symbol,
    ...
)
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6519`

Can a symbol disappear here?

Only if it never reaches ordered candidate processing or if entries are blocked.

Conditions:

- Not in `entry_candidates`.
- `entries_blocked` true, so ordered candidate processing is skipped.
- Stale/backfill rejection before `SIGNAL_READY`.
- Max entries per cycle/minute breaks before reaching lower-ranked candidate.

Logged?

`SIGNAL_READY` is logged if reached.

Can it disappear silently?

Yes for candidates behind rate limits:

```python
if max_per_cycle > 0 and entries_submitted_this_cycle >= max_per_cycle:
    break
...
if minute_capacity <= 0:
    ... ENTRY_RATE_LIMIT_BLOCK ...
    break
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6502`

This can stop processing lower-scored candidates without a per-symbol log. It does log global rate limit, but not every candidate skipped.

For OMER:

If OMER was ready but ranked below candidates selected before the break, it may have no symbol-specific log. However heartbeat should include ready candidate counts. This is plausible only if OMER was in `entry_candidates`.

Recommendation:

When breaking due to per-cycle or per-minute limits, log the skipped candidate symbols and scores.

## Stage 20: Low Live Entry Score

Purpose:

Optional score threshold block.

Code:

```python
if low_live_entry_score_blocked(live_entry_score, min_live_entry_score):
    print("ENTRY_BLOCKED_LOW_SCORE symbol=...")
    record_lifecycle_with_formal(..., "ENTRY_BLOCKED_LOW_SCORE", symbol, ...)
    continue
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6530`

Can a symbol disappear here?

No, it logs per symbol.

For OMER:

Unlikely, because no OMER log.

## Stage 21: Risk Guard

Purpose:

Block entries due to portfolio/risk limits.

Code:

```python
risk_status = evaluate_risk_guard(..., symbol=symbol, ...)
if risk_status.get("blocked"):
    runtime_rate_limited_log(..., f"RISK_GUARD_BLOCK_ENTRY symbol={symbol} reason={reason}")
    record_lifecycle_with_formal(..., "RISK_GUARD_BLOCK_ENTRY", symbol, ...)
    safe_sqlite_call(..., "record_risk_event", symbol=symbol, ...)
    continue
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6554`

Can a symbol disappear here?

No. It produces symbol-specific evidence.

Logged?

Yes, per symbol.

Can it disappear silently?

Mostly no. `runtime_rate_limited_log` can suppress repeated stdout after its unique/window rules, but lifecycle and SQLite risk event are still attempted.

For OMER:

No `RISK_GUARD_BLOCK_ENTRY symbol=OMER` means risk guard is not a valid symbol-specific explanation. Heartbeat `risk_guard_block=1` only reflects the last evaluated candidate, not OMER.

Recommendation:

In analysis and dashboards, never infer symbol-level risk guard from heartbeat alone.

## Stage 22: BUY Order Submission

Purpose:

Submit Market BUY and persist entry metadata.

Code:

```python
order = MarketOrder("BUY", qty)
trade = ib.placeOrder(q, order)
...
record_lifecycle_with_formal(..., "BUY_ORDER_SENT", symbol, ...)
print(f"PAPER BUY SENT symbol={symbol} ...")
state.signal_sent = True
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6605`

Can a symbol disappear here?

No if reached; it logs.

Conditions:

- `placeOrder()` raises. There is no try/except around this symbol-level call, so it may bubble to outer exception handling.

Logged?

Success logs.

Can it disappear silently?

Not for successful order submission.

For OMER:

No order logs or order rows means it did not reach this stage.

## Stage 23: Heartbeat

Purpose:

Summarize loop state.

Code:

```python
heartbeat scanned={len(contracts)} with_data={data_count} ready_new={ready_count}
ready_candidates=...
...
best=... top5=[...] rejects=[...]
```

Reference: `src/live_trading/v67_live_top100_expansion_paper_trader.py:6973`

Can a symbol disappear here?

No, but heartbeat can hide per-symbol details.

Limitations:

- `rejects` is aggregate reason counts only.
- `top5` only shows top 5 by runtime score.
- `scanned` is count of contracts, not entry_symbols.
- `with_data` is count of symbols with usable `snapshot.price`.
- It does not list symbols missing price.
- `risk_guard_block` reflects last risk guard status, not all symbols.

Recommendation:

Add a diagnostic heartbeat/debug table: `contracts_without_price`, `top100_entry_symbols_not_in_contracts`, `entry_symbols_not_in_tickers`.

## Most Likely Explanations For OMER

These are ranked based on the evidence given:

### 95%: OMER never entered runtime processing after Top100 selection

Why:

- No `SIGNAL_READY`
- No `BUY_BLOCKED`
- No `STALE_OR_BACKFILL_READY_SKIPPED`
- No `RISK_GUARD_BLOCK_ENTRY`
- No order/fill/trade/runtime event
- `journal_symbol_lines_count=0`

The most direct way to get this shape is that OMER was not in `contracts` at the time of signal. The main loop only scans `contracts`.

Relevant code:

- Startup silent contract drop: `qualifyContracts()` empty then `continue` at `v67_live_top100_expansion_paper_trader.py:6017`
- Reload selected/subscribed universe replaces `contracts` at `v67_live_top100_expansion_paper_trader.py:4531`
- Main loop scans `for symbol, q in contracts` at `v67_live_top100_expansion_paper_trader.py:6345`

### 80%: OMER was subscribed/contracted but had no usable ticker price

Why:

If OMER was in `contracts` but the ticker had no `last`, bid/ask midpoint, close, bid, or ask, the loop silently skips before state update and before any event.

Relevant code:

```python
if snap.get("price") is None:
    continue
```

Reference: `v67_live_top100_expansion_paper_trader.py:6347`

This also produces no `SIGNAL_READY`, no `BUY_BLOCKED`, no risk event.

### 70%: Subscription cap / selected subscription universe excluded OMER

Why:

Logs around the session had `subscriptions_active=100`, `subscriptions_cap=100`, and `subscription_cap_block=1`. Reload selection explicitly keeps active symbols first and only then fills Top100 slots.

Relevant code:

- cap selection: `v67_live_top100_expansion_paper_trader.py:4421`
- diagnostics only list first 20 skipped symbols: `v67_live_top100_expansion_paper_trader.py:4568`

Counterpoint:

OMER rank 6 makes this unlikely unless many active positions consumed slots or runtime Top100 file/order differed from offline Top100.

### 55%: Runtime feature state was incomplete despite offline candle PASS

Why:

Offline uses complete historical 1m candles. Runtime uses live ticker snapshots. If OMER did not receive usable live ticks during first 5/15/opening range windows, `first_5m_high_pct`, `first_15m_high_pct`, or `or_range_pct` can remain `None`.

Relevant code:

- state update depends on live ticker snapshots: `v67_live_top100_expansion_paper_trader.py:611`
- feature readiness requires non-null first5/first15/OR: `v67_live_top100_expansion_paper_trader.py:682`

Counterpoint:

Even then OMER might appear in aggregate `rejects`, but not symbol-specific logs.

### 35%: Candidate existed but was below rate-limit / max-per-cycle break

Why:

The ordered candidate loop can break after `max_entries_per_cycle` or `max_entries_per_minute`, leaving later candidates without symbol-specific logs.

Relevant code:

- max per cycle break: `v67_live_top100_expansion_paper_trader.py:6502`
- minute capacity break: `v67_live_top100_expansion_paper_trader.py:6505`

Counterpoint:

This requires OMER to be in `entry_candidates`, but current runtime evidence has no OMER line.

### 20%: OMER was suppressed by `state.signal_sent`, active position, or `entry_symbol_allowed`

Why:

Candidate creation requires:

```python
features["ready"] and not state.signal_sent and not has_active_position and entry_symbol_allowed
```

Reference: `v67_live_top100_expansion_paper_trader.py:6394`

These suppressions are not symbol-specific logged.

Counterpoint:

No evidence of prior OMER trade/position in the provided case.

### 5%: Risk guard

Why low:

Risk guard is only reached after `SIGNAL_READY` is recorded and candidate processing starts. It logs per symbol:

`RISK_GUARD_BLOCK_ENTRY symbol=...`

No OMER risk guard event exists. Global heartbeat `risk_guard_block=1` is not symbol-specific and appears to refer to another candidate such as MU.

Relevant code:

- `SIGNAL_READY` before risk guard: `v67_live_top100_expansion_paper_trader.py:6519`
- per-symbol risk event: `v67_live_top100_expansion_paper_trader.py:6564`

## Highest-Value Instrumentation Gaps

These are not implemented here; they are recommendations from the review.

1. Startup contract qualification should log empty qualification per symbol.

Current silent code:

```python
if not qualified:
    continue
```

2. Main loop should log or count symbols in `contracts` with no usable ticker price.

Current silent code:

```python
if snap.get("price") is None:
    continue
```

3. Reload should persist complete `skipped_symbols_due_to_cap`, not only first 20 in stdout.

4. Candidate creation suppressions should be logged for high-rank Top100 symbols:

- `state.signal_sent`
- active position exists
- not in `entry_symbols`
- feature not ready after offline-ready condition

5. Rate-limit breaks should log skipped candidate symbols.

## Final OMER Assessment

OMER's evidence does not support `missed_due_to_risk_guard`.

The most likely failure class is:

`runtime_never_processed_symbol`

Most likely technical causes, in order:

1. OMER was not in `contracts` at signal time.
2. OMER was in `contracts` but had no usable ticker price and hit the silent `snap.price is None` continue.
3. Subscription cap or reload state excluded OMER from active subscriptions.
4. Runtime live state missed early-session feature windows despite offline candle PASS.

The current runtime code has several places where this exact shape can happen silently before `SIGNAL_READY`, especially startup `qualifyContracts()` empty result and main-loop missing ticker price.
