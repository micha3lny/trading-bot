#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$REPO_ROOT/src/live_trading/ranking/daily_top100_builder.py" && -d "$HOME/trading-bot" ]]; then
  REPO_ROOT="$HOME/trading-bot"
fi

cd "$REPO_ROOT"

if [[ ! -f "venv/bin/activate" ]]; then
  log "ERROR missing venv/bin/activate in $REPO_ROOT"
  exit 1
fi

source "venv/bin/activate"

RANKING_DATE="${1:-${RANKING_DATE:-}}"
if [[ -z "$RANKING_DATE" ]]; then
  RANKING_DATE="$(python -c 'from datetime import date,timedelta; d=date.today()-timedelta(days=1); print(next((d-timedelta(days=i)).isoformat() for i in range(7) if (d-timedelta(days=i)).weekday() < 5))')"
fi

TOP_N="${TOP_N:-100}"
UNIVERSE="${UNIVERSE:-data/universe/v68_final_daytrading_universe.csv}"
HISTORY_DIR="${HISTORY_DIR:-data/history/universe_1m}"
OUTPUT_DIR="${OUTPUT_DIR:-data/universe}"
DATED_OUTPUT="${DATED_OUTPUT:-$OUTPUT_DIR/daily_top100_${RANKING_DATE}.csv}"
LATEST_OUTPUT="${LATEST_OUTPUT:-$OUTPUT_DIR/daily_top100_latest.csv}"
SQLITE_PATH="${SQLITE_PATH:-data/runtime/rankings.sqlite}"

log "DAILY_TOP100_PREMARKET_START repo=$REPO_ROOT ranking_date=$RANKING_DATE top_n=$TOP_N"
log "universe=$UNIVERSE"
log "history_dir=$HISTORY_DIR"
log "dated_output=$DATED_OUTPUT"
log "latest_output=$LATEST_OUTPUT"

set +e
python -m src.live_trading.ranking.daily_top100_builder \
  --date "$RANKING_DATE" \
  --universe "$UNIVERSE" \
  --history-dir "$HISTORY_DIR" \
  --output "$DATED_OUTPUT" \
  --latest-output "$LATEST_OUTPUT" \
  --sqlite-path "$SQLITE_PATH" \
  --top-n "$TOP_N"
RC=$?
set -e

if [[ "$RC" -ne 0 ]]; then
  log "DAILY_TOP100_PREMARKET_FAILED builder_rc=$RC"
  exit "$RC"
fi

if [[ ! -f "$DATED_OUTPUT" ]]; then
  log "ERROR dated output missing: $DATED_OUTPUT"
  exit 1
fi

ROW_COUNT="$(python -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "$DATED_OUTPUT")"
if [[ "$ROW_COUNT" -lt 100 ]]; then
  log "ERROR dated output invalid rows=$ROW_COUNT required=100"
  exit 1
fi

if [[ ! -f "$LATEST_OUTPUT" ]]; then
  log "ERROR latest output missing after valid build: $LATEST_OUTPUT"
  exit 1
fi

LATEST_COUNT="$(python -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "$LATEST_OUTPUT")"
if [[ "$LATEST_COUNT" -lt 100 ]]; then
  log "ERROR latest output invalid rows=$LATEST_COUNT required=100"
  exit 1
fi

log "DAILY_TOP100_PREMARKET_DONE rows=$ROW_COUNT latest_rows=$LATEST_COUNT"
