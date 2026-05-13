# v67 Universe 1m Automatic Backfill

## Goal

After market close and EOD liquidation, automatically backfill 1m candles for the full universe (~2100 symbols).

This removes dependence on stale TOP100-only telemetry and allows:

- missed runners analysis across entire universe
- next-day TOP100 recalculation from fresh data
- ranking decay analysis
- identifying runners that were never observed live

## Pipeline

```text
MARKET CLOSE
    ↓
EOD liquidation
    ↓
strategy shutdown
    ↓
automatic universe 1m backfill
    ↓
analytics generation
    ↓
next-day ranking preparation
```

## Current status

Implemented:

- `src/live_trading/analytics/v67_universe_1m_backfill.py`
- full-universe historical 1m downloader
- CSV persistence
- deduplication
- IBKR reconnect-safe logic

Current live recorder still captures 1m candles only for TOP100 live watchlist.

## Recommended integration

Inside `v67_live_top100_expansion_paper_trader.py`:

- after EOD flatten completes
- after managed positions become zero
- spawn subprocess:

```bash
python -m src.live_trading.analytics.v67_universe_1m_backfill --date YYYY-MM-DD
```

Recommended as detached subprocess:

```python
subprocess.Popen(...)
```

so the main strategy process can exit independently.

## Future improvements

- parquet storage
- universe snapshots
- daily ranking persistence
- intraday dynamic runner capture
- async parallel IBKR batching
