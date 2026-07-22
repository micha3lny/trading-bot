# Offline / Live Signal Parity Spec

This document captures the v67 signal definition used by offline analysis. It is a specification of current runtime behavior, not a strategy change.

## Runtime Source Of Truth

Live runtime updates state in `src/live_trading/v67_live_top100_expansion_paper_trader.py`:

- `update_state(...)` updates `first_5m_high`, `first_15m_high`, `or_high`, and `or_low` from usable live tick price snapshots.
- `compute_live_safe_features(...)` computes `first_5m_high_pct`, `first_15m_high_pct`, `or_range_pct`, `spread_bps`, `score`, and `ready`.
- There is no separate live gate requiring current price to break above `or_high`.

## Shared Offline Replay Definition

Offline analysis must use `live_signal_replay(...)` from `src/live_trading/analysis/common.py` when deciding whether a Top100 missed runner should have signaled.

Definitions:

- Market open is the first RTH candle timestamp in the input candle set.
- RTH 1m history timestamps are treated as bar-start timestamps by default.
- A bar-start candle at `13:44:00Z` is available at `13:45:00Z`.
- A bar-start candle at `13:45:00Z` is not available for a `13:45:00Z` decision.
- First 5m window is `[open, open + 5 minutes)`.
- First 15m window is `[open, open + 15 minutes)`.
- Opening range window is `[open, open + opening_range_seconds)`.
- Earliest legal signal time is after all required windows have finalized.
- Offline replay uses completed candle high/low only for finalized window features.
- Offline replay uses candle close as the live-equivalent current price.
- Candle high after the opening range is diagnostic only. It does not satisfy a live breakout gate because v67 has no such gate.

## No-Lookahead Invariants

At candidate timestamp `T`, offline replay may not use:

- Any candle whose availability time is after `T`.
- A final first15/opening-range value before the window is complete.
- A future daily high or future `first_time_above_*` diagnostic.
- High/low from an unfinished candle unless the replay is explicitly run in `bar_end` mode.

## Diagnostic Columns

`opening_range_break_time` and `did_break_or_high` are retained as runner diagnostics. They are not live-entry gates.

`possible_signal_time` is the live-equivalent signal time from `live_signal_replay(...)`.

`first_time_above_5pct` and `first_time_above_8pct` are diagnostics for large runner moves. They are not the same as the v67 first5/first15 gates.
