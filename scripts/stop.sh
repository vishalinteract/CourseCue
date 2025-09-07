#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .api.pid ]]; then
  kill "$(cat .api.pid)" 2>/dev/null || true
  rm -f .api.pid
  echo "[scripts] API stopped"
else
  echo "[scripts] No running API pid file"
fi
# stop Streamlit if you want:
pkill -f "streamlit run demo_app.py" 2>/dev/null || true
