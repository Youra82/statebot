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
# PATH: venv/bin in ~/.bashrc eintragen damit 'python' direkt funktioniert
BASHRC="$HOME/.bashrc"
PATH_LINE="export PATH=\"$SCRIPT_DIR/.venv/bin:\$PATH\""
if ! grep -qF "$SCRIPT_DIR/.venv/bin" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# statebot venv" >> "$BASHRC"
    echo "$PATH_LINE" >> "$BASHRC"
    echo "PATH erweitert — 'python' zeigt jetzt auf .venv/bin/python"
    echo "Fuehre aus: source ~/.bashrc"
else
    echo "PATH bereits konfiguriert."
fi

echo "Installation abgeschlossen."
echo "Naechste Schritte:"
echo "  1. source ~/.bashrc   (einmalig, damit 'python' sofort gilt)"
echo "  2. secret.json konfigurieren"
echo "  3. python build_states.py --pairs 'BTC/USDT:USDT|1d' --start_date '2021-01-01'"
