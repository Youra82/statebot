#!/usr/bin/env python3
# maintenance.py — Automatische Lifecycle-Verwaltung
#
# Täglich via Cron:
#   1. Neue Kerzen → inkrementelle Feature-Vektoren (kein Look-Ahead)
#   2. Calibration Drift prüfen → Telegram-Alert wenn Qualität sinkt
#   3. Monatlich (auto-detektiert): Recluster + Übergangsmatrizen neu aufbauen
#
# Prinzip: Kein Look-Ahead-Bias.
#   Nur Daten bis GESTERN werden verarbeitet.
#   Clustering und Transitions verwenden ausschließlich vergangene Bars.
#
# Aufruf:
#   python maintenance.py                    → auto (täglich oder monatlich)
#   python maintenance.py --force_recluster  → erzwungener Recluster
#   python maintenance.py --check_only       → nur Drift-Report, kein Update
#
# Crontab (VPS):
#   0 1 * * * cd /path/to/statebot && .venv/bin/python maintenance.py >> logs/maintenance.log 2>&1

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.utils.exchange       import Exchange
from statebot.utils.telegram       import send_message
from statebot.engine.features      import compute_features, get_feature_vector, WARMUP_BARS
from statebot.engine.store         import StateStore
from statebot.engine.clusterer     import build_state_labels
from statebot.engine.transitions   import (build_transition_matrix,
                                            build_transition_matrix_order2)
from statebot.engine.changepoint   import ChangepointManager
from statebot.engine.validity      import run_oos_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

DB_PATH          = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'states.db')
MAINT_LOG_PATH   = os.path.join(PROJECT_ROOT, 'artifacts', 'maintenance_log.json')

HISTORY_DAYS_MAP = {
    '15m': 90,
    '1h':  365,
    '4h':  1095,
    '1d':  2190,
}


# ─── Maintenance-Log (JSON) ───────────────────────────────────────────────────

def _load_log() -> dict:
    if os.path.exists(MAINT_LOG_PATH):
        try:
            with open(MAINT_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_log(log: dict):
    os.makedirs(os.path.dirname(MAINT_LOG_PATH), exist_ok=True)
    with open(MAINT_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)


def _log_key(market: str, tf: str, event: str) -> str:
    return f"{market.replace('/', '_')}_{tf}_{event}"


def _days_since(log: dict, key: str) -> int | None:
    ts = log.get(key)
    if ts is None:
        return None
    last = datetime.fromisoformat(ts)
    return (datetime.now(timezone.utc) - last).days


# ─── Laden ───────────────────────────────────────────────────────────────────

def _load_config() -> tuple[dict, dict, list[tuple[str, str]]]:
    path = os.path.join(PROJECT_ROOT, 'settings.json')
    with open(path) as f:
        settings = json.load(f)

    secret_path = os.path.join(PROJECT_ROOT, 'secret.json')
    if not os.path.exists(secret_path):
        logger.critical("secret.json nicht gefunden")
        sys.exit(1)
    with open(secret_path) as f:
        s = json.load(f)
    accounts  = s.get('statebot', [])
    telegram  = s.get('telegram', {})
    if not accounts:
        logger.critical("Kein statebot-Account in secret.json")
        sys.exit(1)

    strats = settings.get('live_trading_settings', {}).get('active_strategies', [])
    pairs  = [(a['symbol'], a['timeframe'])
               for a in strats if a.get('enabled', True)]
    return settings, telegram, pairs


# ─── Inkrementelles Feature-Update ───────────────────────────────────────────

def run_incremental_update(exchange: Exchange, store: StateStore,
                            market: str, tf: str) -> int:
    """
    Lädt neue Kerzen ab dem letzten gespeicherten Bar.
    Kein Look-Ahead: verwendet nur Bars die vor HEUTE enden (closed candles).
    Gibt Anzahl neu eingefügter Vektoren zurück.
    """
    last_bar = store.get_last_bar_time(market, tf)
    end_dt   = datetime.now(timezone.utc) - timedelta(days=1)   # gestern = letzter geschlossener Tag

    if last_bar:
        start_dt = (datetime.fromisoformat(last_bar.replace('Z', '+00:00'))
                    - timedelta(days=3))   # 3 Tage Puffer für fehlende Bars
    else:
        days     = HISTORY_DAYS_MAP.get(tf, 365)
        start_dt = end_dt - timedelta(days=days)

    logger.info(f"[{market}/{tf}] Inkrementelles Update: {start_dt.date()} → {end_dt.date()}")
    df = exchange.fetch_historical_ohlcv(market, tf,
                                          start_dt.strftime('%Y-%m-%d'),
                                          end_dt.strftime('%Y-%m-%d'))
    if df is None or len(df) < WARMUP_BARS + 5:
        logger.warning(f"[{market}/{tf}] Zu wenige Daten für Update")
        return 0

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
        next_close_pct = (float(closes[i + 1]) - curr_close) / curr_close * 100
        next_high_pct  = (float(highs[i + 1])  - curr_close) / curr_close * 100
        next_low_pct   = (float(lows[i + 1])   - curr_close) / curr_close * 100
        bar_time = str(df_feat.index[i])

        store.upsert_vector(market, tf, bar_time, fvec,
                             round(next_close_pct, 4),
                             round(next_high_pct,  4),
                             round(next_low_pct,   4))
        inserted += 1

    logger.info(f"[{market}/{tf}] +{inserted} neue Vektoren | Gesamt: {store.get_count(market, tf)}")
    return inserted


# ─── Recluster ───────────────────────────────────────────────────────────────

def run_recluster(store: StateStore, market: str, tf: str,
                  n_clusters: int = 20) -> dict:
    """
    Neuberechnung aller Cluster und Übergangsmatrizen.
    Gibt Statistiken über Cluster-Drift zurück.
    """
    # State-Definitionen VOR dem Recluster (für Drift-Berechnung)
    old_defs = store.get_state_definitions(market, tf)
    old_names = {d['state_id']: d['name'] for d in old_defs}
    old_up    = {d['state_id']: d.get('up_prob', 0.5) for d in old_defs}

    logger.info(f"[{market}/{tf}] Recluster mit n_clusters={n_clusters}...")
    n_clustered = build_state_labels(store, market, tf, n_clusters=n_clusters)
    store.update_scan_log(market, tf, n_clustered, n_clusters=n_clusters)

    logger.info(f"[{market}/{tf}] Übergangsmatrizen neu aufbauen...")
    build_transition_matrix(store, market, tf)
    n_triplets = build_transition_matrix_order2(store, market, tf)
    logger.info(f"[{market}/{tf}] Order-2: {n_triplets} Triplets")

    # Drift berechnen (Veränderung der up_prob pro State)
    new_defs = store.get_state_definitions(market, tf)
    drift_states = []
    for d in new_defs:
        sid = d['state_id']
        if sid in old_up:
            delta = abs(d.get('up_prob', 0.5) - old_up[sid])
            if delta > 0.10:   # > 10% Veränderung der Up-Wahrscheinlichkeit
                drift_states.append({
                    'state_id': sid,
                    'name': d['name'],
                    'up_prob_old': old_up[sid],
                    'up_prob_new': d.get('up_prob', 0.5),
                    'delta': delta,
                })

    return {
        'n_clustered': n_clustered,
        'n_triplets':  n_triplets,
        'drift_states': drift_states,
    }


# ─── Calibration Drift Erkennung ──────────────────────────────────────────────

def check_calibration_drift(store: StateStore, market: str, tf: str,
                              drift_threshold: float = 1.5) -> list[str]:
    """
    Vergleicht kurzfristige vs. langfristige Brier Scores pro Signal.
    Gibt Warnmeldungen zurück wenn Qualität signifikant sinkt.

    drift_threshold=1.5 → Alarm wenn 7-Tage-BS > 30-Tage-BS × 1.5
    """
    warnings = []
    cal = store.calibrator
    signal_names = ['knn', 'markov', 'funding', 'oi']

    for sig in signal_names:
        bs_7d  = cal.get_brier_score(sig, market, tf, lookback_days=7)
        bs_30d = cal.get_brier_score(sig, market, tf, lookback_days=30)

        if bs_7d is None or bs_30d is None:
            continue   # Zu wenige Daten → kein Urteil

        if bs_7d > bs_30d * drift_threshold:
            pct_worse = (bs_7d / bs_30d - 1.0) * 100
            warnings.append(
                f"{sig}: Qualität gesunken ({pct_worse:.0f}% schlechter)  "
                f"[7d BS={bs_7d:.3f} vs 30d BS={bs_30d:.3f}]"
            )

        # Reliability in letzten 7 Tagen
        rel_now = cal.get_reliability(sig, market, tf, static_fallback=None)
        if rel_now is not None and rel_now < 0.40:
            warnings.append(
                f"{sig}: Reliability kritisch niedrig ({rel_now:.2f})  "
                f"[Signalgeber möglicherweise unbrauchbar]"
            )

    return warnings


# ─── Cluster-Alter Report ─────────────────────────────────────────────────────

def _cluster_age_report(log: dict, pairs: list[tuple[str, str]]) -> str:
    lines = []
    for market, tf in pairs:
        key     = _log_key(market, tf, 'last_cluster')
        n_days  = _days_since(log, key)
        age_str = f"{n_days}d" if n_days is not None else "nie"
        lines.append(f"  {market}/{tf}: letzter Recluster vor {age_str}")
    return '\n'.join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="statebot Maintenance")
    parser.add_argument('--force_recluster', action='store_true',
                        help='Erzwinge Recluster unabhängig vom Intervall')
    parser.add_argument('--check_only', action='store_true',
                        help='Nur Drift-Report, kein Update')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  statebot maintenance.py")
    logger.info(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    settings, telegram, pairs = _load_config()
    maint_cfg = settings.get('maintenance', {})
    recluster_interval = int(maint_cfg.get('recluster_interval_days', 30))
    drift_threshold    = float(maint_cfg.get('drift_alert_threshold', 1.5))
    n_clusters         = int(settings.get('clustering', {}).get('n_clusters', 20))

    log   = _load_log()
    store = StateStore(DB_PATH)

    # ── Kontext-Report ─────────────────────────────────────────────────────────
    logger.info(f"Pairs: {pairs}")
    logger.info(f"Recluster-Intervall: {recluster_interval} Tage")
    logger.info(_cluster_age_report(log, pairs))

    if args.check_only:
        for market, tf in pairs:
            logger.info(f"\n── Calibration Drift {market}/{tf} ──")
            warnings = check_calibration_drift(store, market, tf, drift_threshold)
            if warnings:
                for w in warnings:
                    logger.warning(f"  ! {w}")
            else:
                logger.info("  OK – kein Drift erkannt")
        store.close()
        return

    account = _load_config()[0]   # nur erstes Account
    secret_path = os.path.join(PROJECT_ROOT, 'secret.json')
    with open(secret_path) as f:
        s = json.load(f)
    exchange = Exchange(s.get('statebot', [])[0])

    # ── Phase 1: Inkrementelles Feature-Update ─────────────────────────────────
    logger.info("\n── Phase 1: Inkrementelles Feature-Update ──")
    total_new = 0
    for market, tf in pairs:
        try:
            n = run_incremental_update(exchange, store, market, tf)
            total_new += n
            log[_log_key(market, tf, 'last_update')] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.error(f"Feature-Update fehlgeschlagen {market}/{tf}: {e}", exc_info=True)

    # ── Phase 2: Recluster (wenn fällig) ──────────────────────────────────────
    logger.info("\n── Phase 2: Cluster-Check ──")
    recluster_results = {}

    for market, tf in pairs:
        key       = _log_key(market, tf, 'last_cluster')
        n_days    = _days_since(log, key)
        due       = args.force_recluster or (n_days is None) or (n_days >= recluster_interval)

        if due:
            reason = "erzwungen" if args.force_recluster else \
                     ("erstmalig" if n_days is None else f"fällig nach {n_days}d")
            logger.info(f"[{market}/{tf}] Recluster ({reason})...")
            try:
                result = run_recluster(store, market, tf, n_clusters=n_clusters)
                log[key] = datetime.now(timezone.utc).isoformat()
                recluster_results[f"{market}/{tf}"] = result
                logger.info(
                    f"[{market}/{tf}] Recluster: {result['n_clustered']} Samples, "
                    f"{result['n_triplets']} Triplets, "
                    f"{len(result['drift_states'])} States mit Drift"
                )
                if result['drift_states']:
                    for ds in result['drift_states'][:5]:
                        logger.info(
                            f"    State {ds['state_id']} ({ds['name']}): "
                            f"up_prob {ds['up_prob_old']:.2f} → {ds['up_prob_new']:.2f}  "
                            f"(Δ={ds['delta']:.2f})"
                        )
            except Exception as e:
                logger.error(f"Recluster fehlgeschlagen {market}/{tf}: {e}", exc_info=True)
        else:
            logger.info(f"[{market}/{tf}] Kein Recluster nötig (zuletzt vor {n_days}d)")

    # ── Phase 3: Calibration Drift ─────────────────────────────────────────────
    logger.info("\n── Phase 3: Calibration Drift ──")
    all_warnings = []
    for market, tf in pairs:
        warnings = check_calibration_drift(store, market, tf, drift_threshold)
        if warnings:
            for w in warnings:
                logger.warning(f"  [{market}/{tf}] {w}")
            all_warnings.extend([f"[{market}/{tf}] {w}" for w in warnings])
        else:
            logger.info(f"  [{market}/{tf}] OK – kein Drift")

    # ── Phase 4: CUSUM Change Point Detection ─────────────────────────────────
    logger.info("\n── Phase 4: Change Point Detection (CUSUM) ──")
    cusum_cfg = settings.get('changepoint', {})
    for market, tf in pairs:
        try:
            key_base = f"{market.replace('/', '_')}_{tf}"
            for sig_name in ['knn', 'markov']:
                cusum = ChangepointManager(
                    name=f"{key_base}_{sig_name}",
                    cusum_params={
                        'target': cusum_cfg.get('target_brier', 0.22),
                        'k':      cusum_cfg.get('k', 0.01),
                        'h':      cusum_cfg.get('h', 5.0),
                    }
                )
                cusum.restore_state(log)   # persistierten Zustand laden

                # Brier Score der letzten 7 Tage als aktuelle Beobachtung
                bs = store.calibrator.get_brier_score(sig_name, market, tf, lookback_days=7)
                if bs is not None:
                    result = cusum.update(bs)
                    if result['any_alarm']:
                        msg = (f"Change Point [{market}/{tf}] {sig_name}: "
                               f"BS(7d)={bs:.3f} | CUSUM-Alarm")
                        logger.warning(f"  ⚡ {msg}")
                        all_warnings.append(msg)
                        if market not in recluster_results:
                            logger.info(f"  → Erzwinge Recluster wegen CUSUM-Alarm")
                            try:
                                r = run_recluster(store, market, tf, n_clusters=n_clusters)
                                log[_log_key(market, tf, 'last_cluster')] = \
                                    datetime.now(timezone.utc).isoformat()
                                recluster_results[f"{market}/{tf}"] = r
                            except Exception as e:
                                logger.error(f"  CUSUM-Recluster fehlgeschlagen: {e}")
                    else:
                        logger.info(f"  [{market}/{tf}] {sig_name}: "
                                    f"BS(7d)={bs:.3f}  {cusum.cusum.status_line()}")
                cusum.persist_state(log)   # Zustand für nächsten Cron-Run speichern
        except Exception as e:
            logger.error(f"CUSUM-Fehler {market}/{tf}: {e}", exc_info=True)

    # ── Phase 5: OOS Validity Check ───────────────────────────────────────────
    logger.info("\n── Phase 5: OOS Walk-Forward Validity ──")
    validity_cfg = settings.get('validity', {})
    oos_days     = int(validity_cfg.get('oos_lookback_days', 60))
    oos_k        = int(settings.get('knn_settings', {}).get('k', 20))

    for market, tf in pairs:
        try:
            oos = run_oos_validation(store, market, tf,
                                      last_n_days=oos_days, k=oos_k)
            if not oos['is_valid']:
                msg = (f"OOS Invalid [{market}/{tf}]: {oos['reason']}  "
                       f"(n={oos['n_tested']}, BS={oos.get('brier_score', 'n/a')})")
                logger.warning(f"  ✗ {msg}")
                all_warnings.append(msg)
                # Erzwinge Recluster wenn Modell OOS invalid und noch nicht getan
                if f"{market}/{tf}" not in recluster_results:
                    logger.info(f"  → Erzwinge Recluster wegen OOS-Invalidiät")
                    try:
                        r = run_recluster(store, market, tf, n_clusters=n_clusters)
                        log[_log_key(market, tf, 'last_cluster')] = \
                            datetime.now(timezone.utc).isoformat()
                        recluster_results[f"{market}/{tf}"] = r
                    except Exception as e:
                        logger.error(f"  OOS-Recluster fehlgeschlagen: {e}")
            else:
                logger.info(
                    f"  ✓ [{market}/{tf}]: BS={oos['brier_score']:.3f}  "
                    f"Acc={oos['accuracy']*100:.1f}%  n={oos['n_tested']}"
                )
            log[_log_key(market, tf, 'last_oos_brier')] = str(oos.get('brier_score', ''))
        except Exception as e:
            logger.error(f"OOS-Fehler {market}/{tf}: {e}", exc_info=True)

    # ── Telegram-Alerts ────────────────────────────────────────────────────────
    if telegram:
        msgs = []

        if recluster_results:
            for pair_tf, res in recluster_results.items():
                drift_count = len(res['drift_states'])
                msgs.append(
                    f"🔄 Recluster {pair_tf}: {res['n_clustered']} Samples | "
                    f"{drift_count} States mit >10% Drift"
                )
                if drift_count > 0:
                    for ds in res['drift_states'][:3]:
                        msgs.append(
                            f"  • State {ds['state_id']} ({ds['name']}): "
                            f"{ds['up_prob_old']:.2f}→{ds['up_prob_new']:.2f}"
                        )

        if all_warnings:
            msgs.append(f"\n⚠️ Calibration Drift erkannt:")
            msgs.extend([f"  • {w}" for w in all_warnings[:5]])

        if total_new > 0 and not msgs:
            msgs.append(f"✅ statebot update: +{total_new} neue Vektoren")

        if msgs:
            body = '\n'.join(msgs)
            try:
                send_message(telegram, f"statebot Maintenance\n{body}")
                logger.info("Telegram-Alert gesendet.")
            except Exception as e:
                logger.warning(f"Telegram fehlgeschlagen: {e}")

    # ── Abschluss ──────────────────────────────────────────────────────────────
    _save_log(log)
    store.close()
    logger.info(f"\n✓ maintenance.py abgeschlossen | +{total_new} Vektoren")
    if recluster_results:
        logger.info(f"  Recluster: {list(recluster_results.keys())}")
    if all_warnings:
        logger.info(f"  Warnungen: {len(all_warnings)}")


if __name__ == "__main__":
    main()
