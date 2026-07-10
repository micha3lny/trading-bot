# SQLite Trade Reduction / Reconstruction Review

## Scope

This review covers the SQLite runtime trade reducer/reconstruction path only.
It does not change strategy, entry rules, exit rules, sizing, or live order
submission behavior.

## Entry Trade Rows

Entry rows are created in
`src/live_trading/v67_live_top100_expansion_paper_trader.py` around the BUY
submission path. The live trader builds an entry trade id:

`entry:<session_date>:<symbol>:<orderId>`

and persists it with status `ENTRY_PENDING` via
`SQLiteRuntimeStore.upsert_trade()`. These rows carry entry metadata such as
`entry_order_id`, `top100_rank`, `top100_score`, `live_entry_score`,
`signal_source`, `signal_time`, and `ready_since`.

## Reconstructed Trade Rows

Reconstructed rows were created in
`src/live_trading/storage/sqlite_store.py` inside
`SQLiteRuntimeStore.rebuild_symbol_trade_state()`. The reducer walks execution
rows by symbol, consumes BUY lots FIFO, and previously wrote a CLOSED trade
inside the per-BUY-lot/per-SELL-fill matching loop using:

`reconstructed:<entry_date>:<exit_date>:<symbol>:<buy_execution_id>:<sell_execution_id>`

That meant every matched execution pair could become a separate logical trade.

## Root Cause

The reducer consumed execution quantities correctly, but persisted the result at
the wrong granularity. It wrote one trade row per FIFO execution pair instead of
one row per logical entry/exit order group.

Consequences:

- Existing `entry:*` rows stayed `ENTRY_PENDING`.
- Separate `reconstructed:*` rows were created for the actual close.
- Entry partial fills became multiple reconstructed closed rows.
- A shared SELL execution could appear in multiple reconstructed rows when it
  matched multiple BUY partial fills.
- Dashboard/analysis exports saw only a small number of clean canonical CLOSED
  rows and many carry/reconstructed/unattributed rows.

## Logical Identity

The safe identity for a live logical trade is:

- entry identity: existing `entry:*` trade id if found, otherwise
  `entry_order_id`, otherwise `entry_perm_id`, otherwise BUY execution id.
- exit identity: exit order id if available, otherwise exit perm id, otherwise
  SELL execution id.

This preserves independent trades in the same symbol when they have different
entry orders, while merging partial fills from the same entry/exit order.

## Entry / Exit Partial Fill Aggregation

The fix changes `rebuild_symbol_trade_state()` to collect FIFO match components
first, then persist one canonical row per logical entry/exit group.

For each group:

- quantity = sum matched quantities
- entry_price = weighted average of BUY fill prices by matched quantity
- exit_price = weighted average of SELL fill prices by matched quantity
- gross_pnl = sum component gross PnL
- commission = sum component commissions
- net_pnl = sum component net PnL
- entry_fill_time = earliest BUY execution time
- exit_fill_time / closed_at = latest SELL execution time
- raw_json records component execution ids and group diagnostics

If an original `entry:*` row exists, the reducer updates that row to CLOSED
instead of creating an unrelated reconstructed row.

## Peak / Giveback

`peak_unrealized_pnl` previously used the max of raw snapshot PnL and
price-derived PnL. Raw snapshots can belong to stale/partial quantities, so a
canonical row could show `peak_price == exit_price` but still a large
`giveback_from_peak`.

The fix calculates peak/adverse PnL from canonical quantity whenever
price-derived PnL is available. Raw PnL is now fallback-only.

## Why Only A Few CLOSED Rows Appeared

For sessions like 2026-07-09, many live positions had `entry:*` rows still in
`ENTRY_PENDING`, while CLOSED state was represented by reconstructed rows. SQL
queries that require clean closed rows with filled entry/exit fields therefore
returned only the subset that had been finalized canonically.

## Idempotency

At the start of each symbol rebuild, the reducer clears old reconstructed rows
for that symbol. Re-running the reducer now recreates or updates the same
canonical logical rows rather than appending new per-execution-pair rows.

## Repair Script

Manual repair/backfill is available at:

`scripts/repair_sqlite_trade_reconstruction.py`

It is dry-run by default. Use `--apply` explicitly to rebuild selected symbols
or a date from the execution ledger.

Example dry-run:

`python scripts/repair_sqlite_trade_reconstruction.py --date 2026-07-09`

Example apply:

`python scripts/repair_sqlite_trade_reconstruction.py --date 2026-07-09 --apply`
