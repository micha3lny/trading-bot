#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/micha3lny/trading-bot}"
USER_NAME="${USER_NAME:-micha3lny}"
PYTHON_BIN="$REPO_DIR/venv/bin/python"
SERVICE_PATH="/etc/systemd/system/v68-history-collector.service"
TIMER_PATH="/etc/systemd/system/v68-history-collector.timer"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python venv at $PYTHON_BIN" >&2
  exit 1
fi

sudo tee "$SERVICE_PATH" >/dev/null <<EOF
[Unit]
Description=v68 Universe 1m Parquet History Collector
After=network-online.target ibgateway.service
Wants=network-online.target

[Service]
Type=oneshot
User=$USER_NAME
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_BIN -m src.live_trading.data.v68_universe_1m_parquet_collector --start-date 2026-01-01 --session-type RTH --client-id 168 --max-tasks 300 --request-sleep-seconds 0.7 --batch-size 25 --batch-sleep-seconds 10

# Each run is intentionally bounded. Missing candles are resumed in later timer runs.
TimeoutStartSec=75min
EOF

sudo tee "$TIMER_PATH" >/dev/null <<EOF
[Unit]
Description=Run v68 Universe 1m Parquet History Collector repeatedly outside US market hours

[Timer]
# Outside regular US session. UTC window: 20:15 -> 15:00 next day.
# Runs in bounded chunks so one collector run does not monopolize IBKR API for hours.
OnCalendar=*-*-* 20:15:00 UTC
OnCalendar=*-*-* 21:15:00 UTC
OnCalendar=*-*-* 22:15:00 UTC
OnCalendar=*-*-* 23:15:00 UTC
OnCalendar=*-*-* 00:15:00 UTC
OnCalendar=*-*-* 01:15:00 UTC
OnCalendar=*-*-* 02:15:00 UTC
OnCalendar=*-*-* 03:15:00 UTC
OnCalendar=*-*-* 04:15:00 UTC
OnCalendar=*-*-* 05:15:00 UTC
OnCalendar=*-*-* 06:15:00 UTC
OnCalendar=*-*-* 07:15:00 UTC
OnCalendar=*-*-* 08:15:00 UTC
OnCalendar=*-*-* 09:15:00 UTC
OnCalendar=*-*-* 10:15:00 UTC
OnCalendar=*-*-* 11:15:00 UTC
OnCalendar=*-*-* 12:15:00 UTC
OnCalendar=*-*-* 13:15:00 UTC
OnCalendar=*-*-* 14:15:00 UTC
Persistent=true
Unit=v68-history-collector.service

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now v68-history-collector.timer

echo "Installed and enabled v68-history-collector.timer"
systemctl list-timers --all | grep -E 'v68-history-collector|NEXT|LEFT' || true
