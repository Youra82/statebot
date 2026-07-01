#!/bin/bash
# run_pipeline.sh — statebot Interaktive Pipeline
#
# Modi:
#   ./run_pipeline.sh          → interaktiv (Build + Backtest)
#   ./run_pipeline.sh daily    → via Cron: inkrementelles Update
#   ./run_pipeline.sh monthly  → via Cron: erzwungener Recluster
#   ./run_pipeline.sh check    → Calibration-Drift prüfen (kein Update)
#
# Crontab VPS:
#   0 1 * * *   cd ~/statebot && ./run_pipeline.sh daily  >> logs/maintenance.log 2>&1
#   0 2 1 * *   cd ~/statebot && ./run_pipeline.sh monthly >> logs/maintenance.log 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# ── Venv prüfen ─────────────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}FEHLER: .venv nicht gefunden. Erst install.sh ausführen!${NC}"
    exit 1
fi
source "$SCRIPT_DIR/.venv/bin/activate"
echo -e "${GREEN}✔ Virtuelle Umgebung wurde erfolgreich aktiviert.${NC}"

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

# ── Header ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "       statebot — 4-Ebenen Markt-Zustandsbot"
echo "======================================================="
echo ""

# ── 1. Alte DB löschen? ──────────────────────────────────────────────────────
DB_PATH="$SCRIPT_DIR/artifacts/db/states.db"
if [ -f "$DB_PATH" ]; then
    read -p "Alte State-Datenbank vor dem Start löschen (Neustart)? (j/n) [Standard: n]: " RESET_DB
    RESET_DB="${RESET_DB//[$'\r\n ']/}"
    if [[ "$RESET_DB" == "j" || "$RESET_DB" == "J" || "$RESET_DB" == "y" || "$RESET_DB" == "Y" ]]; then
        rm -f "$DB_PATH"
        rm -f "$SCRIPT_DIR/artifacts/results"/backtest_wf_*.json
        echo -e "${GREEN}✔ Alte State-DB + Backtest-Ergebnisse gelöscht — Neustart.${NC}"
    else
        echo -e "${GREEN}✔ Bestehende State-DB wird beibehalten.${NC}"
    fi
else
    echo -e "${CYAN}ℹ  Keine bestehende State-DB gefunden — wird neu erstellt.${NC}"
fi

# ── 2. Coins / Timeframes ────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Coins und Timeframes:${NC}"
echo "  Leer lassen → automatisch aus active_strategies in settings.json übernehmen"
echo ""
read -p "Coin(s) eingeben (z.B. BTC ETH SOL) [leer=auto]: " COINS_INPUT
read -p "Timeframe(s) eingeben (z.B. 1d 4h) [leer=auto]: " TF_INPUT

COINS_INPUT="${COINS_INPUT//[$'\r\n']/}"
TF_INPUT="${TF_INPUT//[$'\r\n']/}"

if [ -n "$COINS_INPUT" ] && [ -n "$TF_INPUT" ]; then
    echo -e "${CYAN}ℹ  Explizite Auswahl: Coins=$COINS_INPUT | Timeframes=$TF_INPUT${NC}"
elif [ -n "$COINS_INPUT" ]; then
    echo -e "${CYAN}ℹ  Coins: $COINS_INPUT | Timeframes: aus active_strategies${NC}"
elif [ -n "$TF_INPUT" ]; then
    echo -e "${CYAN}ℹ  Coins: aus active_strategies | Timeframes: $TF_INPUT${NC}"
else
    echo -e "${GREEN}✔ Coins und Timeframes werden aus active_strategies übernommen.${NC}"
fi

export STATEBOT_OVERRIDE_COINS="$COINS_INPUT"
export STATEBOT_OVERRIDE_TFS="$TF_INPUT"

# ── 3. Start-Datum ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}--- Empfehlung: Optimaler Rückblick-Zeitraum ---${NC}"
printf "  %-12s  %s\n" "Zeitfenster" "Empfohlenes Start-Datum"
printf "  %-12s  %s\n" "──────────" "──────────────────────────"
printf "  %-12s  %s\n" "5m, 15m"    "vor 90-180 Tagen   → 2025-01-01"
printf "  %-12s  %s\n" "30m, 1h"    "vor 1-2 Jahren     → 2024-01-01"
printf "  %-12s  %s\n" "4h"         "vor 2-4 Jahren     → 2022-01-01"
printf "  %-12s  %s\n" "1d"         "vor 3-5 Jahren     → 2021-01-01"
echo ""
read -p "Start-Datum (YYYY-MM-DD) oder 'a' für Automatik [Standard: a]: " START_INPUT
START_INPUT="${START_INPUT//[$'\r\n ']/}"

START_ARG=""
if [[ "$START_INPUT" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    START_ARG="--start_date $START_INPUT"
    echo -e "${CYAN}ℹ  Start-Datum: $START_INPUT${NC}"
else
    echo -e "${GREEN}✔ Automatischer Zeitraum nach Timeframe.${NC}"
fi

# ── 4. Cluster-Anzahl ─────────────────────────────────────────────────────────
echo ""
read -p "Anzahl Cluster (State-Gruppen) [Standard: 20]: " N_CLUSTERS
N_CLUSTERS="${N_CLUSTERS//[$'\r\n ']/}"
N_CLUSTERS="${N_CLUSTERS:-20}"
if [[ ! "$N_CLUSTERS" =~ ^[0-9]+$ ]]; then N_CLUSTERS=20; fi
echo -e "${CYAN}ℹ  n_clusters = $N_CLUSTERS${NC}"

# ── 5. Inkrementell? (nur wenn DB existiert) ──────────────────────────────────
INCR_ARG=""
if [ -f "$DB_PATH" ]; then
    echo ""
    read -p "Inkrementell updaten? (nur neue Kerzen herunterladen) (j/n) [Standard: n]: " DO_INCR
    DO_INCR="${DO_INCR//[$'\r\n ']/}"
    if [[ "$DO_INCR" == "j" || "$DO_INCR" == "J" ]]; then
        INCR_ARG="--incremental"
        echo -e "${GREEN}✔ Inkrementeller Modus — nur neue Bars werden geladen.${NC}"
    fi
fi

# ── 6. Backtest? ──────────────────────────────────────────────────────────────
echo ""
read -p "Backtest nach dem Build durchführen? (j/n) [Standard: j]: " RUN_BT
RUN_BT="${RUN_BT//[$'\r\n ']/}"
RUN_BT="${RUN_BT:-j}"

CAPITAL=1000
SL=1.5
RR=2.0
RISK=1.0
DO_SWEEP=n
if [[ "$RUN_BT" == "j" || "$RUN_BT" == "J" || "$RUN_BT" == "y" || "$RUN_BT" == "Y" ]]; then
    read -p "Startkapital in USDT [Standard: 1000]: " CAP_INPUT
    CAP_INPUT="${CAP_INPUT//[$'\r\n ']/}"
    if [[ "$CAP_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then CAPITAL=$CAP_INPUT; fi

    read -p "Risiko pro Trade % [Standard: 1.0]: " RISK_INPUT
    RISK_INPUT="${RISK_INPUT//[$'\r\n ']/}"
    if [[ "$RISK_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then RISK=$RISK_INPUT; fi

    echo ""
    read -p "SL/RR automatisch optimieren (Sweep)? (j/n) [Standard: n]: " DO_SWEEP
    DO_SWEEP="${DO_SWEEP//[$'\r\n ']/}"
    DO_SWEEP="${DO_SWEEP:-n}"

    if [[ "$DO_SWEEP" == "j" || "$DO_SWEEP" == "J" ]]; then
        echo -e "${CYAN}ℹ  Sweep-Modus: SL 1.0-4.0% × RR 1.5-3.0 → Top-5 je Coin${NC}"
    else
        read -p "Stop-Loss % [Standard: 1.5]: " SL_INPUT
        SL_INPUT="${SL_INPUT//[$'\r\n ']/}"
        if [[ "$SL_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then SL=$SL_INPUT; fi

        read -p "Risk:Reward-Ratio [Standard: 2.0]: " RR_INPUT
        RR_INPUT="${RR_INPUT//[$'\r\n ']/}"
        if [[ "$RR_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then RR=$RR_INPUT; fi
    fi
fi

# ── Pairs zusammenbauen ──────────────────────────────────────────────────────
PAIRS=$($PYTHON - <<'PYEOF'
import os, json

coins_raw = os.environ.get('STATEBOT_OVERRIDE_COINS', '').strip()
tfs_raw   = os.environ.get('STATEBOT_OVERRIDE_TFS', '').strip()

try:
    with open('settings.json') as f:
        s = json.load(f)
    active = s.get('live_trading_settings', {}).get('active_strategies', [])
    auto_coins = list(dict.fromkeys(
        x['symbol'] for x in active if x.get('symbol') and x.get('enabled', True)
    ))
    auto_tfs = list(dict.fromkeys(
        x['timeframe'] for x in active if x.get('timeframe') and x.get('enabled', True)
    ))
except Exception:
    auto_coins = ['BTC/USDT:USDT']
    auto_tfs   = ['1d']

def to_symbol(coin):
    coin = coin.strip().upper()
    if '/' not in coin:
        return f"{coin}/USDT:USDT"
    return coin

coins = [to_symbol(c) for c in coins_raw.split()] if coins_raw else auto_coins
tfs   = [t.strip() for t in tfs_raw.split()] if tfs_raw else auto_tfs

for sym in coins:
    for tf in tfs:
        print(f"{sym} {tf}")
PYEOF
)

if [ -z "$PAIRS" ]; then
    echo -e "${RED}FEHLER: Keine Pairs gefunden. Coins/Timeframes eingeben oder settings.json konfigurieren.${NC}"
    exit 1
fi

# ── Pipeline starten ─────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Pipeline startet..."
echo "======================================================="
echo ""
echo -e "${CYAN}Scan-Paare:${NC}"
echo "$PAIRS" | while read -r sym tf; do
    echo "  → $sym ($tf)"
done
echo ""

# ── Schritt 1: State Space aufbauen ──────────────────────────────────────────
echo -e "${YELLOW}[Schritt 1/3] State Space aufbauen (Features → Clustering → Transitions)...${NC}"
echo ""

export STATEBOT_PIPELINE=1
echo "$PAIRS" | while IFS=' ' read -r sym tf; do
    echo -e "${CYAN}  Scanne: $sym ($tf)${NC}"
    $PYTHON build_states.py \
        --pairs "${sym}|${tf}" \
        --n_clusters "$N_CLUSTERS" \
        $START_ARG $INCR_ARG
    echo ""
done
unset STATEBOT_PIPELINE

echo -e "${GREEN}✔ State Space aufgebaut.${NC}"
echo ""

# ── Schritt 2: Backtest ───────────────────────────────────────────────────────
if [[ "$RUN_BT" == "j" || "$RUN_BT" == "J" || "$RUN_BT" == "y" || "$RUN_BT" == "Y" ]]; then
    echo -e "${YELLOW}[Schritt 2/3] Backtest läuft...${NC}"
    echo ""
    if [[ "$DO_SWEEP" == "j" || "$DO_SWEEP" == "J" ]]; then
        BT_ARGS="--sweep --min-trades 10 --top-n 1"
    else
        BT_ARGS="--sl-pct $SL --rr $RR"
    fi

    echo "$PAIRS" | while IFS=' ' read -r sym tf; do
        echo -e "${CYAN}  Backtest: $sym ($tf)${NC}"
        $PYTHON run_backtest.py \
            --symbol "$sym" --timeframe "$tf" \
            --capital "$CAPITAL" \
            --risk "$RISK" \
            $BT_ARGS
        echo ""
    done
    echo -e "${GREEN}✔ Backtest abgeschlossen.${NC}"
    echo ""
fi

# ── Schritt 3: Zusammenfassung ────────────────────────────────────────────────
echo -e "${YELLOW}[Schritt 3/3] Ergebnisse...${NC}"
echo ""
$PYTHON -m statebot.analysis.show_results 2>/dev/null || true
echo ""

echo "======================================================="
echo -e "  ${GREEN}Pipeline abgeschlossen!${NC}"
echo ""
echo "  Nächste Schritte:"
echo "    1. Backtest-Analyse:  python -m statebot.analysis.attribution --file ..."
echo "    2. State-Scorecard:   python -m statebot.analysis.attribution --scorecard --files ..."
echo "    3. Live starten:      python master_runner.py"
echo "======================================================="
echo ""

deactivate
