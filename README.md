# statebot — Market State Intelligence Engine

Ein probabilistischer Krypto-Trading-Bot der auf **Market State Analysis** basiert.
Statt direkte Preismuster zu suchen, klassifiziert statebot den aktuellen Marktzustand
und berechnet die Wahrscheinlichkeit einer Aufwärtsbewegung über eine 7-stufige Pipeline
aus Clustering, Markov-Übergängen, KNN und Bayesian Fusion.

---

## Kernidee

Klassische Systeme fragen: *"Welche vergangenen Kerzen sehen dem aktuellen Moment ähnlich?"*

statebot fragt: *"In welchem mathematischen Marktzustand sind wir — und wie entwickeln sich
solche Zustände statistisch weiter?"*

Der Unterschied ist fundamental:

- **KNN auf Rohdaten**: sucht Ähnlichkeit in allen Zeitreihen gleichzeitig
- **statebot**: clustert zuerst in 20 diskrete States, sucht KNN dann nur innerhalb desselben
  States — deutlich weniger Rauschen, höhere Signalqualität

---

## Systemarchitektur

### 7-stufige Vorhersage-Pipeline

```
OHLCV-Daten
    │
    ▼
[1] Feature-Extraktion (22 Dimensionen)
    │   Trend, Momentum, Volatilität, Fraktal, Energie, Kerze, Orderflow
    │
    ▼
[2] State-Zuordnung (KMeans, 20 Cluster)
    │   + Qualitätsprüfung (Silhouette Score)
    │   + Membership Score (Nähe zum Zentroid)
    │
    ▼
[3] KNN innerhalb desselben States (k=20, dynamisch)
    │   → P(up | ähnliche Bars im gleichen State)
    │
    ▼
[4] Markov-Übergangsmatrix (Order 1+2)
    │   → P(up | aktueller State)  ← PRIOR
    │   Look-Ahead-sicher: aus Train-Daten berechnet
    │
    ▼
[5] Bayesian Fusion: Prior × Likelihood
    │   fuse_prior_likelihood(p_markov, p_knn)
    │   reliability-gewichtet (Markov: 0.85, KNN: 0.90)
    │
    ▼
[6] Externe Evidenz-Pipeline (optional)
    │   Funding Rate, OI, Fear & Greed → weiterer Bayes-Update
    │
    ▼
[7] Multi-Target + Eigenmodes
        Expected Close/High/Low %, PCA-Eigenmodes
        → P(bayes) → Entscheidung: Long / Short / Neutral
```

### Feature-Vektor (22 Dimensionen)

| Kategorie | Features |
|---|---|
| **Trend** | ema20_dist, ema50_dist, ema200_dist, ema_cross |
| **Momentum** | rsi, roc, macd_hist |
| **Volatilität** | atr_ratio, atr_change, hv20 |
| **Fraktal** | hurst (R/S-Methode), petrosian_fd |
| **Marktenergie** | permutation_entropy, shannon_entropy |
| **Kerze** | body_ratio, upper_wick, lower_wick, gap |
| **Orderflow** | cvd_slope |
| **Volumen** | rel_volume, obv_slope, delta_ratio |

**Hurst-Exponent**: Misst Trendpersistenz.
- H > 0.55 → Trend-Regime (Persistenz)
- H < 0.45 → Range-Regime (Mean-Reversion)
- H ≈ 0.50 → Neutral (Random Walk)

Wird direkt als Regime-Klassifikation (TREND / RANGE / NEUTRAL) verwendet.

---

## Module-Übersicht

### Engine (`src/statebot/engine/`)

| Datei | Funktion |
|---|---|
| `features.py` | 22-dimensionaler Feature-Vektor, WARMUP=215 Bars |
| `store.py` | SQLite-Interface (4 Tabellen: feature_vectors, state_definitions, transitions, scan_log) |
| `clusterer.py` | KMeans 20 States, Silhouette Quality Score, Membership Score, State-Benennung |
| `transitions.py` | Markov Order-1 und Order-2 Übergangsmatrix |
| `matcher.py` | KNN innerhalb State, dynamisches K mit Gap-Erkennung, Quality Stars (1-5) |
| `predictor.py` | Vollständige 7-Stufen-Pipeline, PredictionSnapshot-Dict, PCA-Eigenmodes |
| `signal_pipeline.py` | Bayes-Fusion (Prior×Likelihood), externe Evidenz-Pipeline, Reliability-Weighting |
| `htf_filter.py` | HTF Supertrend Veto (Wilder ATR), Auto-HTF-Mapping |
| `ensemble.py` | Temporales KNN-Ensemble (30d / 180d / all) |
| `calibrator.py` | Kalibrierungs-Tracking pro Signal-Quelle |
| `changepoint.py` | ADWIN-basierte Drift-Erkennung |
| `validity.py` | OOS-Gültigkeitsprüfung (Brier, Sample-Größe) |

### Strategy (`src/statebot/strategy/`)

| Datei | Funktion |
|---|---|
| `signal.py` | `get_state_signal()` — vollständige Signal-Pipeline inkl. HTF-Veto + Structure Protection |
| `run.py` | Entry Point pro Coin/Timeframe, lädt Config aus settings.json |

### Utils (`src/statebot/utils/`)

| Datei | Funktion |
|---|---|
| `trade_manager.py` | Live-Trading: Fetch OHLCV, Signal, Order platzieren, SL/TP, Tracking |
| `exchange.py` | CCXT-Wrapper (Bitget) |
| `telegram.py` | Telegram-Benachrichtigungen |
| `guardian.py` | Positionsschutz, Max-Loss-Prüfung |

### Analysis (`src/statebot/analysis/`)

| Datei | Funktion |
|---|---|
| `backtester.py` | PnL-Backtest (Candle-Auflösung, Pending Orders, Trailing Stop) |
| `attribution.py` | Trade Attribution, Research/Stability/Scorecard Report |
| `ab_test.py` | A/B-Test mit Bootstrap-CI für State-Filtering |
| `show_results.py` | Kalibrierungs-Kurven, Status-Anzeige |

---

## Besondere Eigenschaften

### HTF Supertrend Veto
Gegentrend-Trades werden hart blockiert wenn der Supertrend auf einem höheren Timeframe
gegensätzlich zeigt. Auto-Mapping: 1h → 4h, 4h → 1d, 1d → 3d etc.

### Structure Protection
SL wird hinter das High/Low der vorherigen Kerze gezogen wenn dieses weiter entfernt liegt
als der konfigurierte feste %-SL. TP wird entsprechend angepasst um das R:R zu erhalten.
Verhindert SL-Hunting durch enge Stops innerhalb der Vorkerze.

### Look-Ahead-sicheres Backtesting
Der Markov-Prior wird im Backtester ausschließlich aus Training-Daten berechnet
(`labeled_in_state` aus `train_rows`). Einzige bekannte Einschränkung: das Clustering
verwendet alle Daten — dieser Bias ist minimal (strukturelle Features, keine Richtungsinfo)
und im Code dokumentiert.

### PredictionSnapshot als zentrale Schnittstelle
Jede Vorhersage gibt einen vollständigen Dict zurück der alle Modell-Metriken enthält:
`state_id`, `regime`, `p_prior`, `p_knn`, `p_bayes`, `membership`, `quality_score`,
`confidence`, `stars`, `k_used`, `reliability_markov`, `reliability_knn`, u.a.
Dieser Snapshot wird pro Trade gespeichert und ermöglicht spätere Attribution ohne
erneutes Backtesting.

### Drei-Achsen-Evaluationsrahmen
```
State-Qualität = f(Profitability, Calibration, Invarianz)

Profitability:  E[V], Profit Factor, Win Rate
Calibration:    Brier Score (<0.22 gut), ECE (<0.05 gut), Reliability Curve
Invarianz:      pct_coins in Pareto-80%-Zone (IRM-Proxy: Cross-Coin Stabilität)
```

### State Scorecard
Kombiniert alle drei Achsen pro State:
```
CORE Edge State = E[V] > 0
               AND Invarianz >= 50% der Coins
               AND Brier < 0.22
```

### A/B-Test mit Paired Bootstrap
Sauberer statistischer Test ob State-Filtering strukturellen Vorteil bringt.
Paired Bootstrap (nicht Independent Bootstrap) weil "after" ein Subset von "before" ist.
Ergibt formales Verdict: GAIN / NEUTRAL / LOSS mit 95%-CI für ΔE[V].

---

## Verzeichnisstruktur

```
statebot/
├── build_states.py         — OHLCV → Features → Clustering → DB (einmaliger Aufbau)
├── master_runner.py        — Startet alle aktiven Strategien aus settings.json
├── maintenance.py          — Tägliches Feature-Update + Drift-Check
├── run_pipeline.sh         — Kompletter Pipeline-Run (interaktiv/daily/monthly)
├── install.sh              — Einmaliges Setup
├── update.sh               — VPS-Update (safe: secret.json wird gesichert)
├── settings.json           — Konfiguration (Risiko, KNN, Clustering, Live-Strategien)
├── secret.json.example     — Template für API-Keys (secret.json NICHT committen)
├── requirements.txt
├── artifacts/
│   ├── db/                 — states.db (SQLite, gitignore)
│   ├── results/            — Backtest-JSONs (gitignore)
│   └── tracker/            — Live-Trade-Tracking (gitignore)
└── src/statebot/
    ├── engine/             — Kern-Logik (Features, Cluster, KNN, Markov, Predictor)
    ├── strategy/           — Signal-Generierung + Live Entry Point
    ├── utils/              — Exchange, Telegram, Trade-Manager
    └── analysis/           — Backtester, Attribution, A/B-Test
```

---

## Konfiguration (`settings.json`)

```json
{
  "risk_settings": {
    "risk_per_entry_pct": 1.0,    // % des Kapitals pro Trade
    "leverage": 5,
    "margin_mode": "isolated",
    "rr_ratio": 2.0,              // Risk:Reward Verhältnis
    "sl_pct": 1.5,                // Stop-Loss in %
    "use_structure_protection": true
  },
  "htf_filter": {
    "enabled": true,
    "timeframe": null,            // null = Auto-Mapping
    "factor": 3.0,                // Supertrend ATR-Multiplikator
    "period": 10                  // Supertrend ATR-Periode
  },
  "knn_settings": {
    "threshold_long": 0.62,       // p_bayes >= 0.62 → Long
    "threshold_short": 0.38,      // p_bayes <= 0.38 → Short
    "min_confidence": 0.45,       // Minimale KNN-Konfidenz
    "min_stars": 2,               // Qualitätsschwelle (1-5)
    "use_bayes": true,
    "use_markov_order2": true,
    "use_temporal_ensemble": true
  },
  "live_trading_settings": {
    "active_strategies": [
      {"symbol": "BTC/USDT:USDT", "timeframe": "1d", "enabled": true}
    ]
  }
}
```

---

## Qualitätssterne (1–5)

Jede Vorhersage bekommt 1-5 Sterne basierend auf:
- **Abweichung von p_bayes = 0.5** (Signalstärke)
- **KNN-Konfidenz** (Konsistenz der k Nachbarn)
- **State-Datenbasis** (wie viele historische Bars im State)
- **Membership Score** (wie nahe liegt der aktuelle Bar am Cluster-Zentroid)

`min_stars: 2` in settings.json → schwache Signale werden nicht gehandelt.

---

## Vom Anfang zum Livebetrieb

### Voraussetzungen

- Python 3.10+
- Bitget-Account mit Futures-Handel freigeschaltet
- API-Key mit Trade-Berechtigung (keine Withdraw-Berechtigung nötig)
- Telegram-Bot (optional, aber empfohlen für Benachrichtigungen)
- Linux-VPS für Livebetrieb (oder Windows mit Task-Scheduler)

---

### Phase 0 — Installation (einmalig, ~15 Minuten)

```bash
# 1. Repository klonen
git clone https://github.com/Youra82/statebot.git
cd statebot

# 2. Installieren (erstellt .venv, Verzeichnisse, kopiert secret.json.example)
bash install.sh

# 3. API-Keys eintragen
nano secret.json
```

`secret.json` Struktur:
```json
{
  "statebot": [
    {
      "name": "Bitget-Main",
      "exchange": "bitget",
      "apiKey": "DEIN_API_KEY",
      "secret":  "DEIN_SECRET",
      "password": "DEIN_PASSPHRASE",
      "use_testnet": false
    }
  ],
  "telegram": {
    "bot_token": "DEIN_BOT_TOKEN",
    "chat_id":   "DEINE_CHAT_ID"
  }
}
```

Installation prüfen:
```bash
.venv/bin/python -c "from statebot.engine.predictor import predict; print('OK')"
```

---

### Phase 1 — State Space aufbauen (einmalig pro Coin, ~5 Min/Coin)

`build_states.py` lädt OHLCV von Bitget, berechnet alle 22 Features und clustert
die Marktdaten in 20 diskrete States. Die Ergebnisse werden in `artifacts/db/states.db`
gespeichert (SQLite).

**Empfohlene Coin-Auswahl** (Diversität über verschiedene Marktmechanismen):

**Hinweis**: Bitget Perpetuals (USDT-Swap) haben erst ab ca. 2021 historische OHLCV-Daten.
Alle Coins mit `--start_date "2021-01-01"` starten.

```bash
# Liquid / Macro-sensitiv (Basisanker)
python build_states.py --pairs "BTC/USDT:USDT|1d" --start_date "2021-01-01"
python build_states.py --pairs "ETH/USDT:USDT|1d" --start_date "2021-01-01"

# Ecosystem-Coins (korreliert aber andere Mikrostruktur)
python build_states.py --pairs "SOL/USDT:USDT|1d" --start_date "2021-01-01"
python build_states.py --pairs "BNB/USDT:USDT|1d" --start_date "2021-01-01"

# Sentiment-getrieben / Niedrigere BTC-Korrelation
python build_states.py --pairs "XRP/USDT:USDT|1d" --start_date "2021-01-01"
python build_states.py --pairs "DOGE/USDT:USDT|1d" --start_date "2021-01-01"
python build_states.py --pairs "ADA/USDT:USDT|1d"  --start_date "2021-01-01"
python build_states.py --pairs "LINK/USDT:USDT|1d" --start_date "2021-01-01"
```

Ergebnis prüfen:
```bash
python -c "
from statebot.engine.store import StateStore
s = StateStore('artifacts/db/states.db')
rows = s.conn.execute('SELECT market, COUNT(*) as n FROM feature_vectors GROUP BY market').fetchall()
for r in rows: print(f'  {r[0]}: {r[1]} Bars')
s.close()
"
```
Erwartung: 2000+ Bars pro Coin (je nach Start-Datum).

---

### Phase 2 — Backtest TRAIN-Periode (2021–2023)

Die Train-Periode dient zur **Pareto-Ableitung**: welche States sind in historischen Daten
profitabel? Sie enthält Bull-Run 2021, Bear-Crash 2022 (Luna, FTX) und Recovery 2023.

**Hinweis**: Bitget Perpetuals haben erst ab ~2021 Daten. Frühere Daten liefern 0 Kerzen.

```bash
for COIN in "BTC/USDT:USDT" "ETH/USDT:USDT" "SOL/USDT:USDT" "BNB/USDT:USDT" \
            "XRP/USDT:USDT" "DOGE/USDT:USDT" "ADA/USDT:USDT" "LINK/USDT:USDT"; do
    echo "=== TRAIN: $COIN ==="
    python -m statebot.analysis.backtester \
        --mode pnl \
        --symbol "$COIN" \
        --timeframe 1d \
        --start-date 2021-01-01 \
        --end-date   2023-06-01 \
        --capital 1000 \
        --risk 1.0 \
        --sl-pct 1.5 \
        --rr 2.0
done

# Train-Outputs umbenennen
cd artifacts/results
for f in backtest_pnl_*_1d.json; do mv "$f" "train_${f}"; done
cd ../..
```

Outputs: `artifacts/results/train_backtest_pnl_BTCUSDTUSDT_1d.json` etc.

---

### Phase 3 — Backtest TEST-Periode (2023–2025, Out-of-Sample)

Die Test-Periode enthält: Recovery 2023, neuer Bull-Run 2024, Korrekturen 2025.
Signale aus dieser Periode wurden **nie für Training verwendet**.

```bash
for COIN in "BTC/USDT:USDT" "ETH/USDT:USDT" "SOL/USDT:USDT" "BNB/USDT:USDT" \
            "XRP/USDT:USDT" "DOGE/USDT:USDT" "ADA/USDT:USDT" "LINK/USDT:USDT"; do
    echo "=== TEST: $COIN ==="
    python -m statebot.analysis.backtester \
        --mode pnl \
        --symbol "$COIN" \
        --timeframe 1d \
        --start-date 2023-06-01 \
        --end-date   2025-06-01 \
        --capital 1000 \
        --risk 1.0 \
        --sl-pct 1.5 \
        --rr 2.0
done

# Test-Outputs umbenennen
cd artifacts/results
for f in backtest_pnl_*_1d.json; do mv "$f" "test_${f}"; done
cd ../..
```

---

### Phase 4 — Analyse und Entscheidung

**Schritt 4a — Research Report pro Coin**

Zeigt: ECE, Brier Score, Drift, Komponenten-Beitrag (HTF / Structure SL / Trailing),
Regime-Breakdown, Reliability Curve, Sterne-Qualität.

```bash
python -m statebot.analysis.attribution \
    --file artifacts/results/test_backtest_pnl_BTCUSDTUSDT_1d.json \
    --report
```

**Schritt 4b — Stability Report (Cross-Coin Aggregat)**

Zeigt: Median PF, StdDev PF, ECE, Brier über alle Coins.

```bash
python -m statebot.analysis.attribution \
    --stability \
    --files "artifacts/results/test_backtest_pnl_*_1d.json"
```

Stop-Kriterium: Median PF < 1.0 oder StdDev PF > 0.50 → Parameter anpassen,
mehr Coins laden, oder anderen Timeframe wählen.

**Schritt 4c — State Scorecard (Kernentscheidung)**

Zeigt für jeden der 20 States: E[V], PF, WR, Brier, Invarianz (Anteil Coins in 80%-Zone),
Gesamtscore 0-3. CORE States = Score 3 (alle drei Achsen gut).

```bash
python -m statebot.analysis.attribution \
    --scorecard \
    --files "artifacts/results/test_backtest_pnl_*_1d.json" \
    --field state_id \
    --save
```

**Schritt 4d — A/B-Test: State Filtering**

Testet ob negative States (PnL < 0 im Training) entfernt werden sollten.
Bootstrap-CI gibt formales Verdict: GAIN / NEUTRAL / LOSS.

```bash
python -m statebot.analysis.ab_test \
    --train artifacts/results/train_backtest_pnl_BTCUSDTUSDT_1d.json \
    --test  artifacts/results/test_backtest_pnl_BTCUSDTUSDT_1d.json \
    --field state_id \
    --bootstrap 5000
```

**Entscheidungsmatrix:**

| Ergebnis | Bedeutung | Aktion |
|---|---|---|
| Stability PF > 1.2, StdDev < 0.30 | System funktioniert cross-coin | Weiter zu Phase 5 |
| AB-Test: GAIN | State Filtering hilft | `min_stars` erhöhen auf 3 |
| AB-Test: NEUTRAL | States already filtered by KNN/Markov | Standard-Config |
| AB-Test: LOSS | States zu gekoppelt | Kein hard Filter |
| ECE > 0.10 | Kalibrierung schlecht | Thresholds anpassen |
| Brier > 0.25 | Modell nicht besser als Zufall | Mehr Daten, anderen TF |

---

### Phase 5 — Konfiguration (`settings.json` anpassen)

Auf Basis der Analyse-Ergebnisse:

```json
{
  "knn_settings": {
    "threshold_long":  0.62,
    "threshold_short": 0.38,
    "min_confidence":  0.45,
    "min_stars": 2,
    "min_state_quality": 0.20
  },
  "live_trading_settings": {
    "active_strategies": [
      {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1d",
        "enabled": true,
        "risk_overrides": {},
        "knn_overrides": {}
      }
    ]
  }
}
```

Nur Coins aktivieren die im Test-Backtest **PF > 1.2 UND Brier < 0.25** erreicht haben.

---

### Phase 6 — Deployment auf VPS (Linux)

```bash
# Auf VPS: Repository klonen
git clone https://github.com/Youra82/statebot.git
cd statebot
bash install.sh

# secret.json manuell erstellen (NICHT per git)
nano secret.json   # API-Keys eintragen

# State Space aufbauen (identisch zu lokal)
python build_states.py --pairs "BTC/USDT:USDT|1d" --start_date "2018-01-01"
python build_states.py --pairs "ETH/USDT:USDT|1d" --start_date "2018-01-01"

# Einmal manuell testen
python master_runner.py
```

**Crontab einrichten:**

```bash
crontab -e
```

```cron
# Signal-Check täglich 00:30 UTC (nach Tageskerzen-Schluss 00:00 UTC)
30 0 * * * cd /pfad/zu/statebot && .venv/bin/python master_runner.py >> logs/run.log 2>&1

# Inkrementelles Feature-Update täglich 01:00
0 1 * * * cd /pfad/zu/statebot && .venv/bin/python maintenance.py >> logs/maintenance.log 2>&1

# Monatlicher Recluster (1. des Monats 02:00)
0 2 1 * * cd /pfad/zu/statebot && .venv/bin/python maintenance.py --force_recluster >> logs/maintenance.log 2>&1
```

**Update einspielen (safe — secret.json wird gesichert):**

```bash
bash update.sh
```

---

### Phase 7 — Pilot-Betrieb (2–4 Wochen, kleines Kapital)

Ziel: Live-Verhalten mit Backtest-Erwartungen vergleichen.

```bash
# Täglich prüfen: Signale und Trades
tail -50 logs/run.log

# Kalibrierungs-Drift prüfen
python maintenance.py --check_drift

# Nach 2 Wochen: Welche States wurden tatsächlich gehandelt?
# (prüfen ob CORE-States dominieren oder auch negative States aktiv sind)
```

**Stop-Kriterien für Pilot:**
- 3+ aufeinanderfolgende Losses auf einem Coin → Coin vorübergehend deaktivieren
- Kein Signal in 14 Tagen → prüfen ob State Space veraltet (`maintenance.py --force_recluster`)
- ECE-Drift-Alert → Recluster + neue Backtests

---

### Phase 8 — Livebetrieb (nach erfolgreichem Pilot)

**Kriterien für Go-Live:**
- Live Win-Rate nicht mehr als ±15% vom Backtest abweichend
- Kein systematischer Slippage-Bias erkennbar
- State-Verteilung im Live-Betrieb ähnlich zur Backtest-Verteilung

**Skalierung:**
- Kapital schrittweise erhöhen (+50% pro Woche, nicht sofort 10×)
- Weitere Coins aus Phase 1-4 hinzufügen wenn deren Test-PF > 1.2
- Monatlich: neuer Stability Report als Regressions-Baseline

---

### Zeitplan (realistisch)

| Phase | Dauer | Hauptblocker |
|---|---|---|
| 0 Setup | 30 Min | API-Keys beschaffen |
| 1 State Space (8 Coins) | 1–2 Std | API-Ratelimit von Bitget |
| 2+3 Backtests | 1–3 Std | API-Ratelimit |
| 4 Analyse | 30 Min | Entscheidung lesen |
| 5 Konfiguration | 15 Min | Entscheidung treffen |
| 6 VPS Deployment | 1 Std | SSH-Zugang, VPS-Setup |
| 7 Pilot | 2–4 Wochen | Zeit + Marktbedingungen |
| 8 Live | laufend | — |

**Hinweis zu API-Ratelimits**: Bitget limitiert historische OHLCV-Abfragen.
Wenn `build_states.py` oder der Backtester auf Ratelimit-Fehler läuft:
kurze Pause zwischen Coin-Aufrufen einbauen oder `--start_date` auf ein neueres Datum setzen.

---

## Häufige Probleme

**"Keine OHLCV-Daten"** → `secret.json` prüfen, API-Key hat möglicherweise keine
Lese-Berechtigung für Futures-Daten.

**"Kein Signal"** → Normal wenn `min_stars=2` und Qualität niedrig ist. Prüfen:
`python -m statebot.analysis.show_results --status`

**"State Space veraltet"** → `python maintenance.py --force_recluster`

**"ECE > 0.15 nach Recluster"** → Thresholds `threshold_long` und `threshold_short`
anpassen (z.B. 0.65 / 0.35 für strengere Selektion).

---

## Technische Details

- **Sprache**: Python 3.10+
- **Datenbank**: SQLite (`artifacts/db/states.db`)
- **Exchange**: Bitget Futures (via CCXT)
- **Clustering**: scikit-learn KMeans, n_clusters=20
- **KNN**: Custom implementation, dynamisches K, Gap-Erkennung
- **Markov**: Order 1 + Order 2 Übergangsmatrix
- **Bayes**: Log-Odds Fusion mit Reliability-Weighting
- **Bootstrap**: Paired Bootstrap, n=2000, 95%-CI
- **Abhängigkeiten**: ccxt, pandas, numpy, scikit-learn, scipy

## Lizenz

Privates Projekt — kein öffentlicher Einsatz ohne Genehmigung.

---

*statebot — Market State Intelligence Engine*
