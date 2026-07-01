#!/usr/bin/env python3
# build_states.py — Historische Daten laden, Features berechnen, clustern, Transitionen lernen
#
# Ablauf:
#   1. OHLCV-Daten von Bitget laden
#   2. 22 Features berechnen
#   3. Feature-Vektoren + Multi-Target Labels in SQLite speichern
#   4. KMeans Clustering → State-IDs zuweisen
#   5. Markov-Übergangsmatrix bauen
#   6. Zusammenfassung ausgeben

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.utils.exchange    import Exchange
from statebot.engine.features   import compute_features, get_feature_vector, WARMUP_BARS
from statebot.engine.store      import StateStore
from statebot.engine.clusterer  import build_state_labels
from statebot.engine.transitions import build_transition_matrix, build_transition_matrix_order2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'states.db')

HISTORY_DAYS_MAP = {
    '15m': 90,
    '1h':  365,
    '4h':  1095,
    '1d':  2190,
}


def _load_accounts() -> tuple[dict, dict]:
    path = os.path.join(PROJECT_ROOT, 'secret.json')
    if not os.path.exists(path):
        logger.critical("secret.json nicht gefunden!")
        sys.exit(1)
    with open(path) as f:
        s = json.load(f)
    accounts = s.get('statebot', [])
    if not accounts:
        logger.critical("Kein 'statebot'-Account in secret.json")
        sys.exit(1)
    return accounts[0], s.get('telegram', {})


def build_features_for_pair(exchange, store, market, tf, start_date, incremental):
    days    = HISTORY_DAYS_MAP.get(tf, 365)
    end_dt  = datetime.now(timezone.utc)

    if incremental and store.get_last_bar_time(market, tf):
        last_bar = store.get_last_bar_time(market, tf)
        start_dt = datetime.fromisoformat(last_bar.replace('Z', '+00:00')) - timedelta(days=2)
        logger.info(f"[{market}/{tf}] Inkrementell ab {start_dt.date()}")
    elif start_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        logger.info(f"[{market}/{tf}] Historisch ab {start_dt.date()}")
    else:
        start_dt = end_dt - timedelta(days=days)
        logger.info(f"[{market}/{tf}] Standard {days}d ab {start_dt.date()}")

    df = exchange.fetch_historical_ohlcv(market, tf,
                                          start_dt.strftime('%Y-%m-%d'),
                                          end_dt.strftime('%Y-%m-%d'))
    if df is None or len(df) < WARMUP_BARS + 10:
        logger.warning(f"[{market}/{tf}] Zu wenige Daten ({len(df) if df is not None else 0})")
        return 0

    logger.info(f"[{market}/{tf}] {len(df)} Kerzen → Feature-Berechnung...")
    df_feat = compute_features(df)
    closes  = df_feat['close'].values
    highs   = df_feat['high'].values
    lows    = df_feat['low'].values

    inserted = 0
    for i in range(len(df_feat) - 1):
        fvec = get_feature_vector(df_feat, i)
        if fvec is None:
            continue

        curr_close = float(closes[i])
        if curr_close <= 0:
            continue

        next_close = float(closes[i + 1])
        # next_high/low = Maximum/Minimum in der NÄCHSTEN Kerze
        next_high  = float(highs[i + 1])
        next_low   = float(lows[i + 1])

        next_close_pct = (next_close - curr_close) / curr_close * 100
        next_high_pct  = (next_high  - curr_close) / curr_close * 100
        next_low_pct   = (next_low   - curr_close) / curr_close * 100

        idx   = df_feat.index[i]
        bar_time = str(idx)

        store.upsert_vector(market, tf, bar_time, fvec,
                             round(next_close_pct, 4),
                             round(next_high_pct,  4),
                             round(next_low_pct,   4))
        inserted += 1

    total = store.get_count(market, tf)
    logger.info(f"[{market}/{tf}] {inserted} Vektoren gespeichert | Gesamt: {total}")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="statebot Feature-Builder + Clustering")
    parser.add_argument('--pairs',       type=str,   default=None)
    parser.add_argument('--start_date',  type=str,   default=None)
    parser.add_argument('--incremental', action='store_true', default=False)
    parser.add_argument('--n_clusters',  type=int,   default=20)
    parser.add_argument('--skip_cluster', action='store_true', default=False,
                        help='Nur Features bauen, kein Clustering')
    parser.add_argument('--reset',       action='store_true', default=False)
    args = parser.parse_args()

    account, _ = _load_accounts()
    exchange   = Exchange(account)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    store = StateStore(DB_PATH)

    if args.reset:
        confirm = input("Feature-Store LOESCHEN? (j/n): ").strip().lower()
        if confirm == 'j':
            store.close()
            os.remove(DB_PATH)
            store = StateStore(DB_PATH)
            logger.info("Store geleert.")
        else:
            logger.info("Abgebrochen.")

    # Pairs
    pairs_input = args.pairs
    if not pairs_input:
        settings_path = os.path.join(PROJECT_ROOT, 'settings.json')
        if os.path.exists(settings_path):
            with open(settings_path) as f:
                settings = json.load(f)
            strats = settings.get('live_trading_settings', {}).get('active_strategies', [])
            pairs_input = ','.join(
                f"{s['symbol']}|{s['timeframe']}" for s in strats
                if s.get('symbol') and s.get('timeframe') and s.get('enabled', True)
            )
    if not pairs_input:
        logger.critical("Keine Pairs. Nutze --pairs oder settings.json")
        sys.exit(1)

    pairs = []
    for entry in pairs_input.split(','):
        entry = entry.strip()
        if '|' in entry:
            m, tf = entry.split('|', 1)
            pairs.append((m.strip(), tf.strip()))

    # Phase 1: Features bauen
    logger.info(f"\n{'='*50}\nPhase 1: Feature-Vektoren bauen\n{'='*50}")
    for market, tf in pairs:
        try:
            build_features_for_pair(exchange, store, market, tf,
                                     args.start_date, args.incremental)
        except Exception as e:
            logger.error(f"Fehler {market}/{tf}: {e}", exc_info=True)

    if args.skip_cluster:
        logger.info("--skip_cluster: Clustering übersprungen.")
        store.close()
        return

    # Phase 2: Clustering
    logger.info(f"\n{'='*50}\nPhase 2: KMeans Clustering (n={args.n_clusters})\n{'='*50}")
    for market, tf in pairs:
        try:
            n = build_state_labels(store, market, tf, n_clusters=args.n_clusters)
            store.update_scan_log(market, tf, n, n_clusters=args.n_clusters)
        except Exception as e:
            logger.error(f"Clustering-Fehler {market}/{tf}: {e}", exc_info=True)

    # Phase 3: Transitions (Order-1 + Order-2)
    logger.info(f"\n{'='*50}\nPhase 3: Markov-Übergangsmatrizen\n{'='*50}")
    for market, tf in pairs:
        try:
            build_transition_matrix(store, market, tf)
            n2 = build_transition_matrix_order2(store, market, tf)
            logger.info(f"  Order-2: {n2} Triplets  [{market}/{tf}]")
        except Exception as e:
            logger.error(f"Transitions-Fehler {market}/{tf}: {e}", exc_info=True)

    # Zusammenfassung
    logger.info(f"\n{'='*50}\nZusammenfassung\n{'='*50}")
    for row in store.get_summary():
        logger.info(
            f"  {row['market']:<25} {row['timeframe']:<6} | "
            f"Total: {row['total']:>6} | Geclustert: {row['clustered']:>6} | "
            f"States: {row.get('n_clusters', '?')} | Zuletzt: {(row.get('latest_bar') or '')[:10]}"
        )

    store.close()
    logger.info("\nbuild_states.py abgeschlossen.")


if __name__ == "__main__":
    main()
