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
  RANKING_DATE="$(python -c 'from datetime import date; from src.live_trading.market_calendar import previous_us_equity_trading_day; print(previous_us_equity_trading_day(date.today()).isoformat())')"
fi

TOP_N="${TOP_N:-100}"
UNIVERSE="${UNIVERSE:-data/universe/v68_final_daytrading_universe.csv}"
HISTORY_DIR="${HISTORY_DIR:-data/history/universe_1m}"
OUTPUT_DIR="${OUTPUT_DIR:-data/universe}"
DATED_OUTPUT="${DATED_OUTPUT:-$OUTPUT_DIR/daily_top100_${RANKING_DATE}.csv}"
LATEST_OUTPUT="${LATEST_OUTPUT:-$OUTPUT_DIR/daily_top100_latest.csv}"
DIAGNOSTICS_OUTPUT="${DIAGNOSTICS_OUTPUT:-$OUTPUT_DIR/daily_top100_${RANKING_DATE}_diagnostics.csv}"
SQLITE_PATH="${SQLITE_PATH:-data/runtime/rankings.sqlite}"
MAX_MISSING_HISTORY_FOR_LATEST="${MAX_MISSING_HISTORY_FOR_LATEST:-}"
MAX_PARTIAL_HISTORY_FOR_TOP100="${MAX_PARTIAL_HISTORY_FOR_TOP100:-0}"
MAX_FAILED_HISTORY_FOR_TOP100="${MAX_FAILED_HISTORY_FOR_TOP100:-0}"
PRIOR_SESSIONS="${TRADING_BOT_TOP100_PRIOR_SESSIONS:-${PRIOR_SESSIONS:-5}}"
PRIOR_READ_SLOW_SECONDS="${TRADING_BOT_TOP100_PRIOR_READ_SLOW_SECONDS:-${PRIOR_READ_SLOW_SECONDS:-2.0}}"
SKIP_HISTORY_READINESS_GATE="${SKIP_HISTORY_READINESS_GATE:-0}"
STRICT_HISTORY_READINESS_GATE="${STRICT_HISTORY_READINESS_GATE:-0}"

log "DAILY_TOP100_PREMARKET_START repo=$REPO_ROOT ranking_date=$RANKING_DATE top_n=$TOP_N"
log "universe=$UNIVERSE"
log "history_dir=$HISTORY_DIR"
log "dated_output=$DATED_OUTPUT"
log "latest_output=$LATEST_OUTPUT"
log "diagnostics_output=$DIAGNOSTICS_OUTPUT"
log "max_missing_history_for_latest=$MAX_MISSING_HISTORY_FOR_LATEST"
log "max_partial_history_for_top100=$MAX_PARTIAL_HISTORY_FOR_TOP100"
log "max_failed_history_for_top100=$MAX_FAILED_HISTORY_FOR_TOP100"
log "strict_history_readiness_gate=$STRICT_HISTORY_READINESS_GATE"
log "prior_sessions=$PRIOR_SESSIONS"
log "prior_read_slow_seconds=$PRIOR_READ_SLOW_SECONDS"

if [[ "$SKIP_HISTORY_READINESS_GATE" != "1" ]]; then
  set +e
  python scripts/check_history_readiness.py \
    --date "$RANKING_DATE" \
    --universe "$UNIVERSE" \
    --history-dir "$HISTORY_DIR" \
    --max-missing "$MAX_MISSING_HISTORY_FOR_LATEST" \
    --max-partial "$MAX_PARTIAL_HISTORY_FOR_TOP100" \
    --max-failed "$MAX_FAILED_HISTORY_FOR_TOP100"
  READINESS_RC=$?
  set -e
  if [[ "$READINESS_RC" -ne 0 ]]; then
    if [[ "$STRICT_HISTORY_READINESS_GATE" == "1" ]]; then
      log "DAILY_TOP100_PREMARKET_FAILED readiness_rc=$READINESS_RC strict_history_readiness_gate=1"
      exit "$READINESS_RC"
    fi
    log "DAILY_TOP100_PREMARKET_READINESS_WARNING readiness_rc=$READINESS_RC strict_history_readiness_gate=0 continuing=1"
  fi
fi

BUILDER_MISSING_ARGS=()
if [[ -n "$MAX_MISSING_HISTORY_FOR_LATEST" ]]; then
  BUILDER_MISSING_ARGS=(--max-missing-history-for-latest "$MAX_MISSING_HISTORY_FOR_LATEST")
fi

set +e
python -m src.live_trading.ranking.daily_top100_builder \
  --date "$RANKING_DATE" \
  --universe "$UNIVERSE" \
  --history-dir "$HISTORY_DIR" \
  --output "$DATED_OUTPUT" \
  --latest-output "$LATEST_OUTPUT" \
  --diagnostics-output "$DIAGNOSTICS_OUTPUT" \
  --sqlite-path "$SQLITE_PATH" \
  "${BUILDER_MISSING_ARGS[@]}" \
  --prior-sessions "$PRIOR_SESSIONS" \
  --prior-read-slow-seconds "$PRIOR_READ_SLOW_SECONDS" \
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
