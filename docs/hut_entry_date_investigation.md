# HUT Entry Date Investigation

Date: 2026-07-02

Scope: dashboard/export investigation only. No trading logic or dashboard logic was changed.

## Executive Summary

The suspicious dashboard row:

| Symbol | Entry Date | Exit Date | Data Quality |
| --- | --- | --- | --- |
| HUT | 2026-06-02 | 2026-06-29 | carry/reconstructed style row reported by export |

could not be verified against production rows locally because this workstation checkout does not contain the production `data/runtime/trading_runtime.sqlite` file or the 3-day dashboard export. The code path review still identifies a high-probability source for this old Entry Date:

1. Runtime Dashboard closed PnL is currently allowed to display execution-derived closed rows from `closed_from_execution_realized_pnl()`.
2. That function groups executions by `symbol` and then enriches the row with the first matching row from the persisted `trades` dataframe by symbol only.
3. If a stale/reconstructed HUT row exists in `trades` with `entry_time=2026-06-02...`, that metadata can override the execution-derived entry time for the HUT execution-closed row on 2026-06-29.
4. The row can then look like a true 2026-06-02 to 2026-06-29 carry even when PnL itself is coming from 2026-06-29 executions.

This is suspicious attribution, not proof of a true broker-held overnight position. A real overnight HUT position would require independent broker/SQLite open-position evidence across those dates.

## Relevant Code Paths

### Runtime Dashboard Chooses Closed Source

File: `src/dashboard/runtime_queries.py`

`load_closed_positions()` builds three possible closed datasets:

| Dataset | Function | Notes |
| --- | --- | --- |
| Persisted trades | `closed_from_trades()` | Reads `trades.status in CLOSED_STATUSES` |
| Optional debug reconstructed rows | `closed_from_executions()` | Used only when `include_reconstructed=True` |
| Execution-realized PnL rows | `closed_from_execution_realized_pnl()` | Used when execution-closed row count is greater than persisted count |

Relevant lines:

| Lines | Behavior |
| --- | --- |
| `src/dashboard/runtime_queries.py:2392-2402` | Loads persisted `trades` rows through `closed_from_trades()` |
| `src/dashboard/runtime_queries.py:2426-2433` | Builds execution-realized closed rows with persisted trades passed in as metadata |
| `src/dashboard/runtime_queries.py:2446-2449` | If execution-realized rows exceed persisted rows, dashboard returns execution-realized rows |

Key consequence: if HUT appears in the execution-realized path, its PnL can be execution-truth while its entry metadata can still come from a stale persisted `trades` row.

### Persisted `trades` Entry Date Path

File: `src/dashboard/runtime_queries.py`

`closed_from_trades()` initially selects:

```sql
substr(entry_fill_time, 1, 10) AS entry_date
```

Relevant lines:

| Lines | Behavior |
| --- | --- |
| `src/dashboard/runtime_queries.py:1671-1680` | Selects `entry_fill_time AS entry_time` and `substr(entry_fill_time, 1, 10) AS entry_date` from `trades` |
| `src/dashboard/runtime_queries.py:1923-1926` | Recomputes `entry_date = date_part(entry_time) or session_date` |
| `src/dashboard/runtime_queries.py:1931-1945` | If `entry_date < exit_date`, marks `CARRIED_POSITION_CLOSED_TODAY`; if source is reconstructed, also marks `CARRY_BASIS_UNVERIFIED` |

If the dashboard row is coming directly from persisted `trades`, then `Entry Date=2026-06-02` comes from `trades.entry_fill_time` or from a fallback execution time assigned to `entry_time`.

### Execution-Realized Metadata Override Path

File: `src/dashboard/runtime_queries.py`

`closed_from_execution_realized_pnl()` groups executions by symbol and produces one execution-truth closed row per flat symbol.

Relevant lines:

| Lines | Behavior |
| --- | --- |
| `src/dashboard/runtime_queries.py:2096-2103` | Groups by `symbol`; only emits row if `buy_qty - sell_qty == 0` and `sell_qty > 0` |
| `src/dashboard/runtime_queries.py:2110-2116` | Computes execution-derived `entry_time = min(BUY execution time)` and `exit_time = max(SELL execution time)` |
| `src/dashboard/runtime_queries.py:2075-2084` | Builds `trade_by_symbol` by taking the first persisted trade row for each symbol |
| `src/dashboard/runtime_queries.py:2140-2146` | Merges metadata in priority order: `sqlite_trades`, exact order match, nearest order, lifecycle, Top100 |
| `src/dashboard/runtime_queries.py:2168-2171` | Sets output `entry_time` and `entry_date` from `metadata.entry_time` before using execution-derived `entry_time` |

The most suspicious line for this case is:

```python
"entry_time": metadata.get("entry_time") or entry_time,
"entry_date": date_part(metadata.get("entry_time") or entry_time) or session_date,
```

If `metadata` came from a stale HUT `trades` row, it can force `Entry Date=2026-06-02` even when the execution group used for PnL is from 2026-06-29.

### Reducer / Reconstructed Trade Path

File: `src/live_trading/storage/sqlite_store.py`

The SQLite execution reducer creates reconstructed closed trades by FIFO matching execution lots.

Relevant lines:

| Lines | Behavior |
| --- | --- |
| `src/live_trading/storage/sqlite_store.py:2010-2015` | BUY execution creates an open FIFO lot |
| `src/live_trading/storage/sqlite_store.py:2019-2027` | SELL consumes open FIFO lots |
| `src/live_trading/storage/sqlite_store.py:2029-2032` | `entry_date` comes from the matched BUY lot `session_date` or BUY execution time |
| `src/live_trading/storage/sqlite_store.py:2045` | Trade ID is derived from BUY execution date, SELL execution date, symbol, buy execution id, sell execution id |
| `src/live_trading/storage/sqlite_store.py:2168-2184` | Upserts reconstructed trade with `session_date=entry_date`, `entry_fill_time=buy_time`, `exit_fill_time=sell_time` |

If HUT has a reconstructed persisted trade from `2026-06-02` to `2026-06-29`, then the reducer paired a 2026-06-02 HUT BUY execution lot with a 2026-06-29 HUT SELL execution. That does not by itself prove the broker held HUT across those dates; it can also happen if historical executions are incomplete, duplicated, missing offsetting sells, or if the reducer used old historical BUYs during a repair/rebuild.

### Optional Debug Reconstruction Path

File: `src/dashboard/runtime_queries.py`

`closed_from_executions()` is older/debug execution-pair reconstruction and is currently only included when `include_reconstructed=True`.

Relevant lines:

| Lines | Behavior |
| --- | --- |
| `src/dashboard/runtime_queries.py:2236-2241` | Reconstructs BUY/SELL execution pairs |
| `src/dashboard/runtime_queries.py:2250-2256` | Attempts to recover entry time from trades, positions, runtime events |
| `src/dashboard/runtime_queries.py:2259-2262` | If recovered entry is before exit date, marks `CARRIED_ENTRY_TIME_RECOVERED` |
| `src/dashboard/runtime_queries.py:2297-2301` | Exports `entry_date = date_part(buy_time)` |

This path is less likely for normal dashboard display because the current normal mode prefers execution-realized PnL rows and does not include debug reconstruction by default.

## Evidence Needed From Production

Run these on the Raspberry / production checkout where the real DB and export exist.

### 1. Find HUT in the Dashboard Export

```bash
cd ~/trading-bot
source venv/bin/activate

python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("PATH_TO_THE_3_DAY_DASHBOARD_EXPORT.csv")
df = pd.read_csv(path)
mask = df.astype(str).apply(lambda col: col.str.upper().eq("HUT")).any(axis=1)
print(df[mask].to_string(index=False))
print("columns=", list(df.columns))
PY
```

Record:

| Field | Expected value to capture |
| --- | --- |
| Trade ID | If exported |
| Entry Execution ID | If exported |
| Exit Execution ID | If exported |
| Entry Order ID | If exported |
| Entry Time | Should show source timestamp |
| Exit Time | Should show source timestamp |
| Data Quality | Carry/reconstructed flags |
| Closed Source | If exported |
| Metadata Attribution Source | If exported |

### 2. HUT Persisted Trades

```bash
sqlite3 data/runtime/trading_runtime.sqlite <<'SQL'
.headers on
.mode column
SELECT
  trade_id,
  strategy_name,
  session_date,
  symbol,
  status,
  entry_fill_time,
  exit_fill_time,
  closed_at,
  entry_price,
  exit_price,
  quantity,
  gross_pnl,
  commission,
  net_pnl,
  substr(raw_json, 1, 500) AS raw_json_prefix
FROM trades
WHERE upper(symbol) = 'HUT'
ORDER BY COALESCE(exit_fill_time, closed_at, entry_fill_time), trade_id;
SQL
```

If a row exists with `entry_fill_time` on 2026-06-02 and `exit_fill_time/closed_at` on 2026-06-29, the dashboard Entry Date likely came from `trades.entry_fill_time`.

### 3. HUT Executions Around Both Dates

```bash
sqlite3 data/runtime/trading_runtime.sqlite <<'SQL'
.headers on
.mode column
SELECT
  execution_id,
  trade_id,
  order_id,
  perm_id,
  strategy_name,
  session_date,
  symbol,
  side,
  quantity,
  price,
  executed_at,
  recorded_at,
  commission,
  commission_source,
  realized_pnl,
  exit_reason,
  substr(raw_json, 1, 300) AS raw_json_prefix
FROM executions
WHERE upper(symbol) = 'HUT'
  AND (
    substr(COALESCE(executed_at, recorded_at), 1, 10) BETWEEN '2026-06-01' AND '2026-06-03'
    OR substr(COALESCE(executed_at, recorded_at), 1, 10) BETWEEN '2026-06-28' AND '2026-06-30'
  )
ORDER BY COALESCE(executed_at, recorded_at), execution_id;
SQL
```

Interpretation:

| Pattern | Meaning |
| --- | --- |
| BUY on 2026-06-02 and matching SELL on 2026-06-29, with no intermediate close | Could be real carry or incomplete execution ledger |
| SELL on 2026-06-29 but no BUY on 2026-06-29 in selected range | Execution-realized grouped row may be borrowing stale metadata |
| BUY and SELL both on 2026-06-29 | Dashboard should not imply 2026-06-02 entry unless metadata override happened |
| Multiple HUT BUY/SELL cycles | Symbol-only metadata join is unsafe |

### 4. HUT Orders and Entry Metadata

```bash
sqlite3 data/runtime/trading_runtime.sqlite <<'SQL'
.headers on
.mode column
SELECT
  order_key,
  trade_id,
  position_key,
  strategy_name,
  session_date,
  symbol,
  side,
  order_id,
  perm_id,
  status,
  ibkr_status,
  submitted_at,
  filled_at,
  substr(raw_json, 1, 500) AS raw_json_prefix
FROM orders
WHERE upper(symbol) = 'HUT'
ORDER BY COALESCE(submitted_at, filled_at), order_key;
SQL
```

This checks whether there was a true BUY order on 2026-06-02 that should be linked to a 2026-06-29 close.

### 5. Broker / SQLite Open Evidence

Historical broker positions are not normally available from the current IBKR API unless captured separately, so use persisted local states:

```bash
sqlite3 data/runtime/trading_runtime.sqlite <<'SQL'
.headers on
.mode column
SELECT
  position_key,
  strategy_name,
  session_date,
  symbol,
  status,
  quantity,
  avg_price,
  source,
  ibkr_quantity,
  active,
  updated_at,
  substr(raw_json, 1, 500) AS raw_json_prefix
FROM positions
WHERE upper(symbol) = 'HUT'
ORDER BY updated_at, position_key;
SQL
```

If there is no active/open HUT state across 2026-06-02 to 2026-06-29, and repeated daily checks showed broker open = 0 / SQLite active = 0, this supports the conclusion that the dashboard row is stale/carry attribution, not real held exposure.

## Why Dashboard Thinks This Is Carry

The dashboard marks carry when:

```python
entry_date < exit_date and selected_window_contains(exit_date)
```

For persisted trades this happens in `closed_from_trades()`:

| Lines | Behavior |
| --- | --- |
| `src/dashboard/runtime_queries.py:1925-1926` | Computes `entry_date` and `exit_date` |
| `src/dashboard/runtime_queries.py:1931-1945` | Adds `CARRIED_POSITION_CLOSED_TODAY`; if reconstructed source, adds `CARRY_BASIS_UNVERIFIED` |

For execution-realized rows this can happen indirectly because `closed_from_execution_realized_pnl()` sets `entry_date` from metadata before execution time:

| Lines | Behavior |
| --- | --- |
| `src/dashboard/runtime_queries.py:2140-2146` | Merges symbol metadata from persisted trades/orders/lifecycle/top100 |
| `src/dashboard/runtime_queries.py:2168-2171` | Uses `metadata.entry_time` before execution-derived `entry_time` |

Therefore HUT can look carried if either:

1. the persisted `trades` row itself says HUT entered on 2026-06-02 and closed on 2026-06-29, or
2. the execution-realized HUT row used 2026-06-29 execution PnL but borrowed stale `entry_time=2026-06-02` metadata from a HUT trade row.

## Current Most Likely Explanation

Without the production DB row, this is the ranked hypothesis list:

| Probability | Hypothesis | Evidence |
| --- | --- | --- |
| High | Stale persisted/reconstructed HUT trade row is supplying old `entry_time` metadata to an execution-realized row | `closed_from_execution_realized_pnl()` maps `trade_by_symbol` using symbol only and gives `sqlite_trades` first priority |
| Medium | SQLite reducer paired a stale 2026-06-02 BUY execution lot with a 2026-06-29 SELL during repair/rebuild | Reducer FIFO uses all executions for symbol and sets `session_date=entry_date` |
| Medium | Dashboard debug reconstruction recovered an old HUT entry from positions/runtime events | Possible only if `include_reconstructed=True` export path was used |
| Low | Broker actually held HUT from 2026-06-02 to 2026-06-29 | Conflicts with repeated broker open=0 / SQLite active=0 checks unless those checks missed HUT or were after forced cleanup |

## What Entry Date Should Mean

For clean same-day lifecycle trades:

| Case | Correct Entry Date |
| --- | --- |
| Persisted lifecycle trade with reliable BUY execution | Actual BUY execution date |
| Execution-realized row with same-day BUY and SELL | First BUY execution date for that execution group |
| True broker-held carry with verified open position | Original broker BUY date if independently verified |
| Carry/reconstructed/unverified row | Should not imply true broker hold; should show `unknown`, `unverified`, or clearly label basis as reconstructed |

For the HUT case, unless production queries prove a real broker-held overnight position, the safest interpretation is:

**Entry Date should be treated as unverified reconstructed basis, not as proof that HUT was held from 2026-06-02 to 2026-06-29.**

## Recommended Fix, Not Implemented

Do not implement until the production HUT evidence is reviewed.

1. In `closed_from_execution_realized_pnl()`, do not let symbol-only `sqlite_trades` metadata override `entry_time` / `entry_date` for execution-truth rows unless it can be matched by execution id, order id, perm id, or close-time proximity.
2. If metadata is symbol-only only, allow it to fill scores/order IDs with `metadata_attribution_confidence=low`, but keep `entry_time` from the execution group.
3. For rows with `CARRY_BASIS_UNVERIFIED`, render Entry Date as `UNVERIFIED 2026-06-02` or add a separate `Basis Entry Date` column instead of implying true broker holding period.
4. Add dashboard diagnostics:
   - `entry_date_source`
   - `entry_date_confidence`
   - `entry_date_warning`
5. Add a unit test:
   - Same symbol has stale persisted trade metadata from old date.
   - Execution-realized row for current date has same-day BUY/SELL.
   - Dashboard must keep execution-derived Entry Date and not borrow stale metadata date.

## Immediate Production Check Commands

Suggested single command block:

```bash
cd ~/trading-bot
source venv/bin/activate

python - <<'PY'
import sqlite3, json
from pathlib import Path

db = Path("data/runtime/trading_runtime.sqlite")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

print("=== HUT trades ===")
for r in conn.execute("""
SELECT trade_id, strategy_name, session_date, status, entry_fill_time, exit_fill_time, closed_at,
       entry_price, exit_price, quantity, gross_pnl, commission, net_pnl, raw_json
FROM trades WHERE upper(symbol)='HUT'
ORDER BY COALESCE(exit_fill_time, closed_at, entry_fill_time), trade_id
"""):
    d = dict(r)
    raw = d.pop("raw_json")
    print(d)
    try:
        j = json.loads(raw or "{}")
    except Exception:
        j = {}
    print("raw keys:", {k: j.get(k) for k in ("reconstruction_source","buy_execution_id","sell_execution_id","entry_date","exit_date","entry_executed_at","exit_executed_at")})

print("=== HUT executions around 2026-06-02 and 2026-06-29 ===")
for r in conn.execute("""
SELECT execution_id, trade_id, order_id, perm_id, strategy_name, session_date, side, quantity, price,
       executed_at, recorded_at, commission, commission_source, realized_pnl, exit_reason
FROM executions
WHERE upper(symbol)='HUT'
  AND (
    substr(COALESCE(executed_at, recorded_at), 1, 10) BETWEEN '2026-06-01' AND '2026-06-03'
    OR substr(COALESCE(executed_at, recorded_at), 1, 10) BETWEEN '2026-06-28' AND '2026-06-30'
  )
ORDER BY COALESCE(executed_at, recorded_at), execution_id
"""):
    print(dict(r))

print("=== HUT positions ===")
for r in conn.execute("""
SELECT position_key, strategy_name, session_date, status, quantity, avg_price, source, ibkr_quantity, active, updated_at, raw_json
FROM positions WHERE upper(symbol)='HUT'
ORDER BY updated_at, position_key
"""):
    d = dict(r)
    raw = d.pop("raw_json")
    print(d)
    try:
        j = json.loads(raw or "{}")
    except Exception:
        j = {}
    print("raw keys:", {k: j.get(k) for k in ("entry_time","entry_price","open_lot_execution_ids","broker_position_reducer_suppressed","stale_open_lot_suppressed")})
PY
```
