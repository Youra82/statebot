# engine/validity.py — Walk-Forward OOS Validity Check
#
# Beantwortet: "Ab wann wird das aktuelle Modell historisch ungültig?"
#
# Unterschied zu Calibration Drift:
#   Drift:    "Die Qualität sinkt" (Messung über Zeit)
#   Validity: "Das Modell ist auf den letzten N Tagen schlechter als Zufall"
#             (Binäre Entscheidung: gültig / ungültig)
#
# Walk-Forward Logik (kein Look-Ahead):
#   1. Centroids wurden aus historischen Daten bis T_train berechnet
#   2. OOS-Test verwendet nur Bars NACH T_train (die echte Zukunft des Modells)
#   3. Für jeden OOS-Bar: assign_state() → KNN (nur Bars vor diesem Bar) → p_up
#   4. Vergleich mit actual outcome → Brier Score
#
# Retrain-Grenze (compute_retrain_boundary):
#   Scannt rückwärts: ab welchem Datum verschlechtert sich OOS-Performance?
#   Das gibt den konkreten Zeitpunkt, ab dem das Modell "altert".

import logging
from datetime import datetime, timezone, timedelta

import numpy as np

from statebot.engine.features  import FEATURE_COLS
from statebot.engine.clusterer import assign_state_to_vector
from statebot.engine.matcher   import knn_within_state

logger = logging.getLogger(__name__)


# ─── OOS-Validierung ──────────────────────────────────────────────────────────

def run_oos_validation(store, market: str, tf: str,
                        last_n_days: int = 60,
                        min_samples: int = 10,
                        k: int = 20) -> dict:
    """
    Testet das aktuelle Modell auf den letzten N Tagen (echter OOS).

    Nutzt die aktuell gespeicherten Centroids und State-Definitions —
    also den Modellstand vom letzten Recluster.

    Returns:
        is_valid      — True wenn Modell noch reliable
        brier_score   — OOS Brier Score (niedriger = besser)
        accuracy      — Direktionaler Treffer in %
        n_tested      — Anzahl OOS-Bars getestet
        reason        — Erklärung wenn not valid
        cutoff_date   — Datum ab dem OOS getestet wurde
    """
    state_defs = store.get_state_definitions(market, tf)
    if not state_defs:
        return _invalid_result("Keine State-Definitions vorhanden", 0)

    centroids = np.array([sd['centroid'] for sd in state_defs], dtype=np.float64)

    # OOS-Zeitfenster
    now      = datetime.now(timezone.utc)
    cutoff   = now - timedelta(days=last_n_days)
    cutoff_s = cutoff.strftime('%Y-%m-%d')

    # Alle Vektoren im OOS-Fenster
    all_rows  = store.get_all_vectors(market, tf)
    oos_rows  = [r for r in all_rows
                  if r.get('next_close_pct') is not None
                  and r['bar_time'][:10] >= cutoff_s]
    train_rows = [r for r in all_rows if r['bar_time'][:10] < cutoff_s]

    if len(oos_rows) < min_samples:
        return _invalid_result(
            f"Zu wenige OOS-Bars: {len(oos_rows)} < {min_samples}", len(oos_rows)
        )

    errors:    list[float] = []
    corrects:  list[int]   = []
    n_no_state = 0

    for oos_bar in oos_rows:
        feat_arr = np.array([oos_bar.get(c, np.nan) for c in FEATURE_COLS], dtype=np.float64)
        if np.any(np.isnan(feat_arr)):
            continue

        # State bestimmen (Inferenz mit aktuellen Centroids)
        state_id = assign_state_to_vector(feat_arr, centroids)

        # NUR Training-Rows dieses States als Referenz (kein Look-Ahead)
        state_train = [r for r in train_rows
                        if r.get('state_id') == state_id
                        and r.get('next_close_pct') is not None]
        if len(state_train) < 5:
            n_no_state += 1
            continue

        result = knn_within_state(feat_arr, state_train, k=k)
        if result is None:
            continue

        p_up   = result['p_up']
        actual = 1.0 if float(oos_bar['next_close_pct']) > 0 else 0.0
        errors.append((p_up - actual) ** 2)
        corrects.append(1 if (p_up >= 0.5 and actual == 1) or
                              (p_up < 0.5  and actual == 0) else 0)

    if len(errors) < min_samples:
        return _invalid_result(
            f"Zu wenige auswertbare Bars (kein State oder KNN): {len(errors)}", len(errors)
        )

    brier     = float(np.mean(errors))
    accuracy  = float(np.mean(corrects))
    n_tested  = len(errors)

    # Gültigkeitskriterien
    MAX_BRIER = 0.28   # BS=0.25 = Zufalls-Baseline; 0.28 = leicht schlechter als Zufall
    MIN_ACC   = 0.44   # Direktionale Trefferquote mindestens 44%

    is_valid  = (brier <= MAX_BRIER) and (accuracy >= MIN_ACC)
    reason    = ""
    if brier > MAX_BRIER:
        reason = f"Brier Score {brier:.3f} > {MAX_BRIER} (Zufalls-Grenze)"
    elif accuracy < MIN_ACC:
        reason = f"Direktionale Genauigkeit {accuracy*100:.1f}% < {MIN_ACC*100:.0f}%"

    logger.info(
        f"OOS Validity [{market}/{tf}]: n={n_tested}, "
        f"BS={brier:.3f}, Acc={accuracy*100:.1f}%, "
        f"{'VALID ✓' if is_valid else 'INVALID ✗  ' + reason}"
    )

    return {
        'is_valid':    is_valid,
        'brier_score': brier,
        'accuracy':    accuracy,
        'n_tested':    n_tested,
        'n_no_state':  n_no_state,
        'reason':      reason,
        'cutoff_date': cutoff_s,
        'max_brier':   MAX_BRIER,
        'min_acc':     MIN_ACC,
    }


def _invalid_result(reason: str, n: int) -> dict:
    return {
        'is_valid':    False,
        'brier_score': None,
        'accuracy':    None,
        'n_tested':    n,
        'n_no_state':  0,
        'reason':      reason,
        'cutoff_date': None,
        'max_brier':   0.28,
        'min_acc':     0.44,
    }


# ─── Retrain-Grenze ───────────────────────────────────────────────────────────

def compute_retrain_boundary(store, market: str, tf: str,
                              windows: list[int] | None = None,
                              k: int = 20) -> dict:
    """
    Scannt rückwärts: ab welchem Zeitpunkt verschlechtert sich OOS-Performance?

    Gibt Brier Scores für verschiedene Lookback-Fenster zurück.
    Der älteste Lookback mit akzeptablem Brier Score ist die "Gültigkeitsgrenze".

    windows — Lookback-Fenster in Tagen, von kurz nach lang
    """
    if windows is None:
        windows = [14, 30, 60, 90, 180]

    results = []
    for days in windows:
        r = run_oos_validation(store, market, tf, last_n_days=days, k=k)
        results.append({
            'lookback_days': days,
            'brier_score':   r['brier_score'],
            'accuracy':      r['accuracy'],
            'is_valid':      r['is_valid'],
            'n_tested':      r['n_tested'],
        })

    # Grenze: letzter Punkt an dem das Modell noch valid war
    boundary_days = None
    for r in reversed(results):
        if r['is_valid']:
            boundary_days = r['lookback_days']
            break

    logger.info(f"Retrain-Grenze [{market}/{tf}]: "
                f"{'Modell gültig bis ' + str(boundary_days) + 'd' if boundary_days else 'Modell überall invalid'}")

    return {
        'market':         market,
        'timeframe':      tf,
        'boundary_days':  boundary_days,  # None = Modell ist sofort invalid
        'window_results': results,
    }
