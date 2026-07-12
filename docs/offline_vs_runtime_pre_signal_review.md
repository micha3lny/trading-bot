# Offline vs Runtime Pre-SIGNAL_READY Review

## Scope

This review covers the gap where offline replay says a Top100 symbol should have become ready, but runtime never emitted `SIGNAL_READY`.

It does not change trading behavior. It documents the feature definitions and the new read-only analyzer:

```bash
python scripts/investigate_offline_runtime_pre_signal.py --date 2026-07-09 --force
```

## Offline Feature Definition

Offline replay reads historical 1m parquet candles from:

```text
data/history/universe_1m/session_type=RTH/symbol=SYMBOL/year=YYYY/month=MM/day=DD.parquet
```

For each symbol it calculates:

- RTH open price from the first parquet candle.
- first 5 minute high and `first_5m_high_pct`.
- first 15 minute high and `first_15m_high_pct`.
- opening range high/low/range over the first 15 minutes.
- breakout time as first candle after the opening range whose high reaches OR high.
- offline score from the same component weights used by live feature scoring where possible.

Offline can prove that a symbol should have been ready according to historical candles. It cannot prove that the live bot had a contract, ticker, usable price, valid spread, initialized state, or a clean session boundary at that moment.

## Runtime Feature Definition

Runtime readiness is calculated in `src/live_trading/v67_live_top100_expansion_paper_trader.py`:

- `snapshot_from_ticker(...)` extracts bid/ask/last/close/midpoint/volume from IBKR ticker objects.
- `update_state(...)` initializes and updates `SymbolState`, including first price, first 5 minute high, first 15 minute high, and OR high/low.
- `compute_live_safe_features(...)` calculates:
  - `first_5m_high_pct`
  - `first_15m_high_pct`
  - `or_range_pct`
  - spread bps
  - ready flag
  - live score

The runtime ready gate additionally requires:

- usable live price,
- price >= `min_price`,
- spread <= `max_spread_bps` if spread is available,
- symbol not already `signal_sent`,
- symbol not actively held,
- symbol allowed in current entry universe.

## Important Difference

Offline candles and runtime ticks are not equivalent evidence.

An offline candle-derived `possible_signal_time` is not an observed runtime event. It must not be treated as `SIGNAL_READY`. The analyzer therefore separates:

- offline expected readiness,
- observed runtime evidence,
- inferred/missing runtime evidence.

## Analyzer Output

The analyzer writes:

```text
data/analysis/offline_runtime_pre_signal_cases_YYYY-MM-DD.csv
data/analysis/offline_runtime_pre_signal_summary_YYYY-MM-DD.csv
data/analysis/offline_runtime_pre_signal_summary_ALL.csv
```

Each case includes:

- offline candle source and feature values,
- observed runtime subscription/state/feature values when present,
- `first_divergence_stage`,
- `first_divergence_reason`,
- `runtime_evidence_source`,
- `confidence`,
- `likely_live_impact`,
- `restart_could_explain`,
- `restart_mechanism`.

## Divergence Stages

The analyzer can classify first divergence as:

- `top100_mismatch`
- `subscription`
- `ticker/price`
- `runtime_state_missing`
- `session/open-price initialization`
- `first5`
- `first15`
- `OR range`
- `spread`
- `score`
- `stale state`
- `signal_sent carryover`
- `runtime feature snapshot missing`
- `runtime rejection reason not logged`
- `runtime evidence missing`
- `runtime evidence unavailable`

It does not assume runtime passed a condition just because no rejection log exists.

## Could This Explain Why The Bot Works Better After Restart?

### Subscription / Ticker Missing

Answer: yes.

Mechanism: startup qualification/subscription can rebuild the desired Top100 contract/ticker set from scratch. A daily Top100 refresh path may leave symbols unsubscribed, fail to reconcile stale ticker objects, or leave old symbols occupying capacity.

Evidence supporting it: runtime logs now include `TOP100_REFRESH_DIFF`, `TOP100_SUBSCRIPTION_RECONCILE`, `MARKET_DATA_SUBSCRIPTION_ACTIONS`, and `SYMBOL_PIPELINE_HEALTH`. If these show missing contract/ticker/usable price for affected symbols, restart plausibly masks the issue.

Missing evidence: symbol-specific pre-signal state snapshots at the offline expected signal time.

### Stale State / Session Boundary Carryover

Answer: yes.

Mechanism: if `signal_sent`, `ready_since`, first price, first5/first15, or OR values carry across sessions, the runtime gate can suppress a new-day signal. Restart clears most in-memory state and can therefore improve behavior.

Evidence supporting it: `RUNTIME_STATE_SESSION_BOUNDARY_CHECK` and `SYMBOL_PIPELINE_HEALTH` can show stale `signal_sent` or state not in Top100.

Missing evidence: per-symbol state values exactly at the offline signal time.

### No Usable Price

Answer: yes.

Mechanism: live `snapshot_from_ticker` may return no usable price even though parquet candles later contain data. Restart can recreate ticker subscriptions and recover price flow.

Evidence supporting it: `NO_USABLE_TICKER_PRICE` or `SYMBOL_PIPELINE_HEALTH usable_price=0`.

Missing evidence: full ticker payload over time if no journal log was captured.

### Spread Rejection

Answer: no/uncertain.

Mechanism: spread is market-state dependent. Restart does not inherently fix a wide spread, but it can change whether spread is available.

Evidence supporting it: `LIVE_FEATURE_DEBUG reason=spread_too_wide` or spread bps in runtime event.

Missing evidence: live spread snapshots if not logged.

### Runtime Evidence Missing

Answer: uncertain, but suspicious.

Mechanism: no symbol-specific evidence means either the symbol was never processed, or diagnostics were insufficient. Restart could explain the first case but not the second.

Evidence supporting it: zero rows/events plus no journal lines for symbol.

Missing evidence: pre-signal state snapshot for every Top100 symbol.

## Minimum Next Runtime Diagnostics

If the analyzer still reports `runtime feature snapshot missing` or `runtime evidence missing`, add stage markers only:

- `PRE_SIGNAL_STATE_SNAPSHOT`
- `PRE_SIGNAL_REJECTION`
- `SESSION_STATE_RESET_SUMMARY`
- `TOP100_SYMBOL_STATE_CREATED`
- `TOP100_SYMBOL_SUBSCRIPTION_READY`
- `LIVE_FEATURES_AT_OFFLINE_SIGNAL_TIME`

These should be diagnostics only and must not change strategy, order logic, or thresholds.
