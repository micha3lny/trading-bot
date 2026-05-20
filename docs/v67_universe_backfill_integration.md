# v67 Universe 1m Automatic Backfill

## Goal

After market close and EOD liquidation, automatically backfill fresh 1m candles for the entire validated daytrading universe.

This removes dependence on stale TOP100-only telemetry and allows:

- missed runners analysis across entire universe
- next-day TOP100 recalculation from fresh data
- ranking decay analysis
- identifying runners that were never observed live
- replay/debugging from parquet history
- weekend/manual historical rebuilds

---

## Current validated universe

Recovered final production universe:

- source symbols scanned: 4439
- rejected junk / broken symbols removed
- final validated universe: 2463 symbols

Universe file now used across runtime:

```text
data/universe/v68_final_daytrading_universe.csv
```

Companion symbols list:

```text
data/universe/v68_final_daytrading_symbols.txt
```

Previous deprecated universe:

```text
data/universe/v64_universe_alpha_ranked.csv
```

was removed from active runtime defaults.

---

## Runtime migration completed

Updated components:

- v67 live trader
- v65 observer
- v68 parquet collector
- analytics backfill
- v59 backtest universe loader

All runtime defaults now point to:

```text
data/universe/v68_final_daytrading_universe.csv
```

---

## Architecture

```text
IBKR
  ↓
LIVE BOT (TOP100 runtime)
  ↓
CONTROL API :8767
  ↓
History collector queue
  ↓
Detached subprocess
  ↓
v68_universe_1m_parquet_collector
  ↓
Parquet / CSV storage
  ↓
Analytics / ranking rebuild
```

---

## Control API

Runtime control server:

```text
127.0.0.1:8767
```

Full endpoint reference:

```text
docs/control-api.md
```

Implemented endpoints:

### Flatten single symbol

```bash
curl -X POST http://127.0.0.1:8767/flatten_symbol \
  -H "Content-Type: application/json" \
  -d '{"symbol":"NVDA"}'
```

### Flatten all positions

```bash
curl -X POST http://127.0.0.1:8767/flatten_all_positions
```

### Run history collector

```bash
curl -X POST http://127.0.0.1:8767/run_history_collector \
  -H "Content-Type: application/json" \
  -d '{
    "start_date":"2026-05-15",
    "end_date":"2026-05-15",
    "session_type":"RTH",
    "max_tasks":300
  }'
```

Plan missing full-universe candles without connecting to IBKR:

```bash
curl -X POST http://127.0.0.1:8767/run_history_collector \
  -H "Content-Type: application/json" \
  -d '{
    "start_date":"2026-01-01",
    "end_date":"2026-05-15",
    "session_type":"RTH",
    "max_tasks":3000,
    "plan_only":true,
    "force":true
}'
```

By default, even `force:true` must not start a large collector run during the live RTH window. This protects the live trading client from extra IBKR load. If an operator intentionally wants to override this safety gate, the request must include `"allow_live_session": true`.

The collector logs a coverage line before any IBKR work:

```text
HISTORY_COLLECTOR_START symbols=2463 total_symbols=2463 tasks=... complete=... skipped_existing=... missing=... pending=...
HISTORY_COLLECTOR_DONE total_symbols=2463 processed=... skipped_existing=... complete=... partial=... no_data=... failed=... retries=...
```

`tasks` is `symbols x trading weekdays`. `max_tasks` limits only how many missing tasks this run will process. Re-running the collector for the same range is safe: complete parquet files and complete status rows are skipped, so each batch continues rebuilding the backlog.

v67 runtime can queue the overnight collector automatically at `20:15,23:00,03:00,07:00 UTC`. It uses the same single-process guard plus a collector lock file at `data/runtime/history_collector.lock`, so multiple collector instances should not run at the same time.

Status:

```bash
curl http://127.0.0.1:8767/history_collector/status
```

Cancel:

```bash
curl -X POST http://127.0.0.1:8767/history_collector/cancel
```

Manual EOD flatten test trigger:

```bash
curl -X POST http://127.0.0.1:8767/eod/flatten
```

---

## Collector runtime behavior

Collector is launched as detached subprocess.

Example runtime logs:

```text
CONTROL_API_HISTORY_COLLECTOR_QUEUED
CONTROL_API_HISTORY_COLLECTOR_START
CONTROL_API_HISTORY_COLLECTOR_PROCESS_STARTED
HISTORY_COLLECTOR_START symbols=2463
HISTORY_SYMBOL_OK NVDA rows=390
HISTORY_PROGRESS_SAVED
HISTORY_COLLECTOR_DONE
```

Collector supports:

- queued execution
- concurrency
- reconnect-safe operation
- partial retry handling
- no-data detection
- outside-session execution

---

## Weekend/session guard fix

Original collector window logic incorrectly treated weekend UTC times as active session windows.

Problem observed:

```text
reason=market_session_active
```

on Sunday evening.

Control API guard logic was patched.

Collector now correctly works:

- weekends
- post-market
- overnight UTC windows
- manual rebuild windows

---

## Current working collector command

Example successful production execution:

```text
python -m src.live_trading.data.v68_universe_1m_parquet_collector \
  --start-date 2026-05-15 \
  --end-date 2026-05-15 \
  --session-type RTH \
  --client-id 168 \
  --max-tasks 300 \
  --allow-outside-window
```

Observed runtime:

```text
symbols=2463
```

with successful candle downloads.

---

## Important operational notes

### IBKR port

Gateway/TWS runtime:

```text
4002
```

### Control API port

```text
8767
```

### Live runtime still scans

```text
TOP100
```

for entries.

Full universe is currently used for:

- historical collection
- analytics
- replay
- ranking rebuild
- future expansion work

---

## Current status

Working:

- live trading runtime
- hard block after restart
- manual flatten APIs
- detached history collector
- collector queue
- full universe recovery
- parquet collector runtime
- weekend collector execution
- universe migration to 2463 symbols

Pending future work:

- automatic scheduled collector trigger
- analytics auto-generation
- dynamic intraday universe expansion
- parquet compaction
- missed-runner dashboards
- live replay tooling

---

## Recommended production flow

```text
MARKET CLOSE
    ↓
EOD liquidation
    ↓
entries blocked
    ↓
history collector queued
    ↓
2463-symbol 1m backfill
    ↓
analytics generation
    ↓
next-day ranking rebuild
```
