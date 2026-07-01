#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "secret.json" ]; then cp secret.json secret.json.bak; fi
git fetch origin
git reset --hard origin/main
if [ -f "secret.json.bak" ]; then cp secret.json.bak secret.json; rm secret.json.bak; fi
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete
chmod +x *.sh master_runner.py build_states.py run_backtest.py 2>/dev/null || true
if [ -d ".venv" ]; then .venv/bin/pip install -q --upgrade -r requirements.txt; fi
echo "Update abgeschlossen."
