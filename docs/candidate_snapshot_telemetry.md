# Candidate Snapshot Telemetry

Candidate snapshot telemetry is an optional, read-only observability sidecar for the v67 live entry loop. It records the state used to evaluate the runtime Top100 without changing candidate ranking, signal eligibility, order decisions, or order submission.

The feature is disabled by default.

## Snapshot Types

### LIGHT

LIGHT contains one row for every expected runtime Top100/watchlist symbol in every completed decision scan. A row is retained even when the contract, ticker, usable price, or `SymbolState` is missing, or when the symbol never reaches candidate ranking.

LIGHT is finalized after the entry-selection and order-dispatch loop so it includes the actual outcome of that scan. It contains identity, Top100 metadata, pipeline presence, current market values, current BUY features, ranking, global gates, sizing, selection outcome, and order outcome.

Record key:

```text
session_date + process_start_id + scan_id + symbol
```

`scan_uid` is:

```text
process_start_id + ":" + scan_id
```

### FULL

FULL contains completed 1-minute candles and the corresponding base feature state. It is emitted for controlled reasons:

- `new_completed_1m_bar`
- `initialization_state_change`
- `completion_state_change`
- `state_created`
- `explicit_feature_state_checkpoint`

Completed-candle rows use the completed bar and the feature state captured before the first tick of the next minute. Intraminute changes remain visible in LIGHT.

FULL uses `feature_state_revision`, `feature_state_hash`, and `full_snapshot_ref` for traceability. A completed candle is emitted at most once per symbol and process tracker.

FULL dedupe key:

```text
session_date + process_start_id + symbol + candle_timestamp + emit_reason + feature_state_revision
```

Premarket volume semantics remain explicitly marked as `UNKNOWN` until separately validated. Telemetry does not add premarket volume or VWAP to BUY logic.

## Storage Layout

Parquet chunks are the live source of truth:

```text
data/live/recorder/YYYY-MM-DD/top100_candidate_snapshots/
  candidate_snapshot_manifest.json
  light/
    part-PROCESS_ID-000000.parquet
    part-PROCESS_ID-000001.parquet
  full/
    part-PROCESS_ID-000000.parquet
    part-PROCESS_ID-000001.parquet
```

One bounded queue item contains one complete scan batch. The trading thread uses `put_nowait()`. A dedicated writer thread writes compressed Parquet chunks through temporary files and atomic rename. If the queue is full, the complete scan batch is dropped and recorded in the manifest; the trading loop does not wait or retry.

Snapshot rows are not written to the main runtime SQLite database or journald. Journald receives only periodic aggregate `CANDIDATE_SNAPSHOT_HEALTH` lines and exceptional writer start/stop errors.

## Manifest

`candidate_snapshot_manifest.json` records:

- schema and session date
- run/process IDs
- expected, enqueued, written, and dropped scan batches
- expected, written, and dropped rows
- first and last scan UID
- expected and written scan IDs by process
- missing scan ranges
- expected and written symbols by scan
- min/max/average written symbols per scan
- queue depth and high watermark
- last/max write latency
- last write error and writer error count
- session completeness

Completeness statuses:

- `COMPLETE`
- `PARTIAL_QUEUE_DROP`
- `PARTIAL_WRITE_ERROR`
- `PARTIAL_PROCESS_RESTART`
- `PARTIAL_MISSING_SYMBOL_ROWS`
- `IN_PROGRESS`

Check health during a session:

```bash
journalctl -u v67-trader.service --since today --no-pager \
  | grep CANDIDATE_SNAPSHOT_HEALTH
```

Check the durable manifest:

```bash
python -m json.tool \
  data/live/recorder/YYYY-MM-DD/top100_candidate_snapshots/candidate_snapshot_manifest.json
```

Important fields for operational review are `session_completeness`, `dropped_scan_batches`, `dropped_rows`, `missing_scan_ranges`, `queue_high_watermark`, `last_write_latency_ms`, and `last_write_error`.

## CLI Flags

```text
--candidate-snapshot-telemetry-enabled
--no-candidate-snapshot-telemetry-enabled
--candidate-snapshot-dir PATH
--candidate-snapshot-queue-size 16
--candidate-snapshot-chunk-rows 5000
--candidate-full-state-checkpoint-seconds 60
```

When `--candidate-snapshot-dir` is empty, the recorder root from `--recorder-dir` is used. Telemetry remains disabled unless `--candidate-snapshot-telemetry-enabled` is explicitly supplied.

## CSV Export

CSV is generated only after the session or on demand:

```bash
python scripts/export_candidate_snapshots.py \
  --date YYYY-MM-DD \
  --recorder-root data/live/recorder \
  --kind both \
  --output-dir data/analysis
```

Use `--kind light` or `--kind full` to export one dataset. The exporter streams Parquet chunks and deduplicates record keys; it does not modify source chunks.

## Safe Enable Procedure

Do not restart an active trading session solely to enable telemetry.

1. Pull and run focused tests after the completed session.
2. Add the enable flag and optional sizing flags to the v67 systemd service command.
3. Run `systemctl daemon-reload` after the session.
4. Start or restart v67 in the normal pre-session maintenance window.
5. Confirm one periodic `CANDIDATE_SNAPSHOT_HEALTH` line and inspect the manifest.
6. Watch queue drops, writer errors, disk usage, CPU, and I/O wait during the first session.

Recommended initial flags:

```text
--candidate-snapshot-telemetry-enabled
--candidate-snapshot-dir data/live/recorder
--candidate-snapshot-queue-size 16
--candidate-snapshot-chunk-rows 5000
--candidate-full-state-checkpoint-seconds 60
```

## Disable And Rollback

Remove `--candidate-snapshot-telemetry-enabled`, or replace it with `--no-candidate-snapshot-telemetry-enabled`, then restart v67 during a maintenance window. Existing Parquet chunks and manifests remain untouched and can still be exported.

Code rollback does not require deleting telemetry data. Revert the deployment commit or deploy the preceding commit, retain the snapshot directories, and restart only in the normal maintenance window.

## Retention

There is no automatic retention or deletion in this implementation. No session directories, Parquet chunks, manifests, or CSV exports are removed automatically. Storage cleanup requires a separate, explicitly reviewed procedure.
