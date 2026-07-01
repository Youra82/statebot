#!/bin/bash
# run_pipeline.sh — statebot Haupt-Pipeline
#
# Modi:
#   ./run_pipeline.sh              → interaktiv (manueller Build)
#   ./run_pipeline.sh daily        → täglich via Cron (inkrementelles Update)
#   ./run_pipeline.sh monthly      → monatlich via Cron (erzwungener Recluster)
#   ./run_pipeline.sh check        → Calibration-Drift prüfen (kein Update)
#
# Crontab VPS:
#   0 1 * * *   cd /path/statebot && .venv/bin/python maintenance.py >> logs/maintenance.log 2>&1
#   0 2 1 * *   cd /path/statebot && .venv/bin/python maintenance.py --force_recluster >> logs/maintenance.log 2>&1
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then echo "[ERROR] .venv fehlt. ./install.sh zuerst!"; exit 1; fi
export PYTHONPATH="$SCRIPT_DIR/src"

MODE="${1:-interactive}"

# ─── Nicht-interaktive Cron-Modi ─────────────────────────────────────────────
if [ "$MODE" = "daily" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M UTC')] statebot daily maintenance"
    mkdir -p logs
    $PYTHON maintenance.py >> logs/maintenance.log 2>&1
    echo "  fertig."
    exit 0
fi

if [ "$MODE" = "monthly" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M UTC')] statebot monthly recluster"
    mkdir -p logs
    $PYTHON maintenance.py --force_recluster >> logs/maintenance.log 2>&1
    echo "  fertig."
    exit 0
fi

if [ "$MODE" = "check" ]; then
    $PYTHON maintenance.py --check_only
    exit 0
fi

echo ""
echo "========================================================"
echo "  statebot Pipeline  (interaktiv)"
echo "========================================================"
echo ""

# ─── Pairs ────────────────────────────────────────────────────────────────────
PAIRS_FROM_SETTINGS=$($PYTHON -c "
import json
try:
    with open('settings.json') as f:
        s = json.load(f)
    st = s.get('live_trading_settings',{}).get('active_strategies',[])
    print(','.join(f\"{a['symbol']}|{a['timeframe']}\" for a in st if a.get('enabled',True)))
except: print('')
" 2>/dev/null)

echo "Pairs aus settings.json: $PAIRS_FROM_SETTINGS"
read -p "Pairs benutzen? (ENTER = ja, eigene: BTC/USDT:USDT|1d): " PAIRS_INPUT
[ -z "$PAIRS_INPUT" ] && PAIRS_INPUT="$PAIRS_FROM_SETTINGS"
[ -z "$PAIRS_INPUT" ] && echo "[ERROR] Keine Pairs." && exit 1

# ─── Clustering ───────────────────────────────────────────────────────────────
read -p "Anzahl Cluster [20]: " N_CLUSTERS
N_CLUSTERS="${N_CLUSTERS:-20}"
read -p "Start-Datum (YYYY-MM-DD, ENTER = Standard): " START_DATE
read -p "Inkrementell? (j/n) [n]: " DO_INCR

FEAT_CMD="$PYTHON build_states.py --pairs \"$PAIRS_INPUT\" --n_clusters $N_CLUSTERS"
[ -n "$START_DATE" ]  && FEAT_CMD="$FEAT_CMD --start_date $START_DATE"
[ "$DO_INCR" = "j" ]  && FEAT_CMD="$FEAT_CMD --incremental"

echo ""
echo "Phase 1-3: Features + Clustering + Transitions..."
eval "$FEAT_CMD"
echo ""

# ─── Backtest ─────────────────────────────────────────────────────────────────
read -p "Backtest durchfuehren? (j/n): " DO_BT
if [ "$DO_BT" = "j" ]; then
    read -p "Start-Kapital [1000]: " CAP; CAP="${CAP:-1000}"
    read -p "SL % [1.5]: " SL; SL="${SL:-1.5}"
    read -p "R:R [2.0]: " RR; RR="${RR:-2.0}"
    read -p "Risiko % [1.0]: " RISK; RISK="${RISK:-1.0}"
    $PYTHON run_backtest.py --capital "$CAP" --sl-pct "$SL" --rr "$RR" --risk "$RISK"
fi

# ─── Ergebnisse ──────────────────────────────────────────────────────────────
echo ""
$PYTHON -m statebot.analysis.show_results

echo ""
echo "========================================================"
echo "  Pipeline abgeschlossen."
echo "  Live starten: python master_runner.py"
echo "  State-Übersicht: python -m statebot.analysis.show_results --states --symbol BTC/USDT:USDT --timeframe 1d"
echo "========================================================"
echo ""
