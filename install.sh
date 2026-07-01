#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "statebot Install"
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --quiet
mkdir -p artifacts/db artifacts/results artifacts/tracker artifacts/tmp logs
chmod +x *.sh master_runner.py build_states.py run_backtest.py 2>/dev/null || true
if [ ! -f "secret.json" ]; then
    cp secret.json.example secret.json
    echo "secret.json erstellt — API-Keys eintragen!"
fi
echo "Installation abgeschlossen."
echo "Naechste Schritte:"
echo "  1. secret.json konfigurieren"
echo "  2. ./run_pipeline.sh"
