#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/micha3lny/trading-bot}"
USER_NAME="${USER_NAME:-micha3lny}"
HOST="${DASHBOARD_HOST:-0.0.0.0}"
PORT="${DASHBOARD_PORT:-8501}"
SERVICE_NAME="${SERVICE_NAME:-trading-dashboard}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
RUN_SCRIPT="$REPO_DIR/scripts/run_dashboard.sh"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "Missing executable dashboard runner at $RUN_SCRIPT" >&2
  echo "Run: chmod +x $RUN_SCRIPT" >&2
  exit 1
fi

sudo tee "$SERVICE_PATH" >/dev/null <<EOF
[Unit]
Description=Trading Bot Streamlit Runtime Dashboard
After=network-online.target trading-bot.service
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
Environment=STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ExecStart=$RUN_SCRIPT --server.address $HOST --server.port $PORT --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo "Installed and started ${SERVICE_NAME}.service"
systemctl --no-pager --full status "$SERVICE_NAME" || true
