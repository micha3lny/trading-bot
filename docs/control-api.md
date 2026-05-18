# Control API

Runtime server:

```text
127.0.0.1:8767
```

The API is local-only and is meant for operational control of the running v67 paper trader.

## Health

```bash
curl http://127.0.0.1:8767/health
```

Returns active managed positions, entry block status, pending flatten commands, and history collector state.

## Pause / Resume Entries

```bash
curl -X POST http://127.0.0.1:8767/pause_entries
curl -X POST http://127.0.0.1:8767/resume_entries
```

`pause_entries` blocks new BUY orders. Existing positions are still managed by exits and EOD flatten.

## Flatten Positions

Single symbol:

```bash
curl -X POST "http://127.0.0.1:8767/flatten_symbol?symbol=NVDA"
```

All active managed positions:

```bash
curl -X POST http://127.0.0.1:8767/flatten_all_positions
```

Dry run:

```bash
curl -X POST "http://127.0.0.1:8767/flatten_all_positions?dry_run=true"
```

## EOD Flatten

The bot automatically starts EOD flatten at `--eod-flatten-utc`, default `19:45` UTC.

Manual EOD trigger for tests:

```bash
curl -X POST http://127.0.0.1:8767/eod/flatten
```

Force retry mode:

```bash
curl -X POST http://127.0.0.1:8767/eod/flatten \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

Status:

```bash
curl http://127.0.0.1:8767/eod/status
```

EOD flatten blocks new entries, sends exits for active positions, and verifies the result against the IBKR portfolio.

## History Collector

Run collector:

```bash
curl -X POST http://127.0.0.1:8767/run_history_collector \
  -H "Content-Type: application/json" \
  -d '{"start_date":"2026-05-18","end_date":"2026-05-18","session_type":"RTH","max_tasks":300}'
```

Force collector outside the normal window:

```bash
curl -X POST http://127.0.0.1:8767/run_history_collector \
  -H "Content-Type: application/json" \
  -d '{"start_date":"2026-05-18","end_date":"2026-05-18","session_type":"RTH","max_tasks":300,"force":true}'
```

Status:

```bash
curl http://127.0.0.1:8767/history_collector/status
```

Cancel queued/running collector:

```bash
curl -X POST http://127.0.0.1:8767/history_collector/cancel
```

Hard cancel:

```bash
curl -X POST http://127.0.0.1:8767/history_collector/cancel \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```
