#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d venv ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

streamlit run src/dashboard/runtime_dashboard.py "$@"
