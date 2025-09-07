#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) ensure venv
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# 2) activate and install deps
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt >/dev/null || true
python -m pip install fastapi uvicorn streamlit requests pandas scikit-learn >/dev/null

# 3) start API in background
PYTHONPATH=. python -m uvicorn src.serve.app:app \
  --host 127.0.0.1 --port 8000 --reload --reload-dir src \
  --log-level info &
API_PID=$!
echo $API_PID > .api.pid
echo "[scripts] API started :8000 (pid $API_PID)"

# 4) start Streamlit (foreground)
python -m streamlit run demo_app.py --server.port 8501
# when UI exits (Ctrl+C), stop API
kill ${API_PID} 2>/dev/null || true
rm -f .api.pid
