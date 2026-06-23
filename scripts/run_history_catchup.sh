#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$REPO_ROOT/src/live_trading/data/v68_universe_1m_parquet_collector.py" && -d "$HOME/trading-bot" ]]; then
  REPO_ROOT="$HOME/trading-bot"
fi

cd "$REPO_ROOT"

if [[ ! -f "venv/bin/activate" ]]; then
  log "ERROR missing venv/bin/activate in $REPO_ROOT"
  exit 1
fi

source "venv/bin/activate"

SESSION_DATE="${1:-${SESSION_DATE:-}}"
if [[ -z "$SESSION_DATE" ]]; then
  SESSION_DATE="$(python -c 'from datetime import date; from src.live_trading.market_calendar import previous_us_equity_trading_day; print(previous_us_equity_trading_day(date.today()).isoformat())')"
fi

SESSION_TYPE="${SESSION_TYPE:-RTH}"
CLIENT_ID="${CLIENT_ID:-168}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
REQUEST_SLEEP_SECONDS="${REQUEST_SLEEP_SECONDS:-0.7}"
BATCH_SIZE="${BATCH_SIZE:-25}"
BATCH_SLEEP_SECONDS="${BATCH_SLEEP_SECONDS:-10.0}"

log "HISTORY_CATCHUP_FORCED date=$SESSION_DATE session_type=$SESSION_TYPE repo=$REPO_ROOT"

python -m src.live_trading.data.v68_universe_1m_parquet_collector \
  --date "$SESSION_DATE" \
  --session-type "$SESSION_TYPE" \
  --client-id "$CLIENT_ID" \
  --max-tasks 0 \
  --max-attempts "$MAX_ATTEMPTS" \
  --retry-failed \
  --allow-outside-window \
  --request-sleep-seconds "$REQUEST_SLEEP_SECONDS" \
  --batch-size "$BATCH_SIZE" \
  --batch-sleep-seconds "$BATCH_SLEEP_SECONDS"
