# engine/ensemble.py — Multi-Horizon Temporal KNN Ensemble
#
# Kernidee:
#   Verschiedene Zeitfenster im historischen Datenstrom modellieren
#   verschiedene Aspekte des Markts:
#
#   short (30d):  Aktuelles Micro-Regime — passt sich schnell an
#   mid  (180d):  Zyklisches Regime     — Balance Recency/History
#   long (alle):  Strukturelles Regime  — langfristige Basis
#
# Warum log-odds statt einfaches Mittel:
#   p=0.80, p=0.60 → simple average = 0.70
#   log-odds fusion → 0.72  (stärker gewichtet in Richtung sicherer Signal)
#   Das ist mathematisch korrekt für unabhängige Evidenzquellen.
#
# Inverse Uncertainty Weighting:
#   w_final = config_weight × confidence × sample_size_factor
#   Horizonte mit wenigen Daten oder niedriger Konfidenz
#   haben automatisch weniger Einfluss.

import logging
from datetime import datetime, timezone, timedelta

import numpy as np

from statebot.engine.matcher import knn_within_state

logger = logging.getLogger(__name__)


DEFAULT_HORIZONS = [
    {'days': 30,   'weight': 1.00, 'label': 'short'},
    {'days': 180,  'weight': 0.70, 'label': 'mid'},
    {'days': None, 'weight': 0.40, 'label': 'long'},   # None = alle Daten
]


def _filter_rows_by_horizon(state_rows: list[dict], days: int | None) -> list[dict]:
    """Filtert state_rows auf die letzten N Tage (None = alle)."""
    if days is None:
        return state_rows
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
    return [r for r in state_rows if r.get('bar_time', '')[:10] >= cutoff]


def _logit(p: float) -> float:
    eps = 1e-7
    p   = max(eps, min(1 - eps, p))
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def temporal_knn_ensemble(
    current_features: np.ndarray,
    state_rows: list[dict],
    k: int = 20,
    feature_weights: np.ndarray | None = None,
    gap_factor: float = 0.0,
    horizons: list[dict] | None = None,
    min_samples_per_horizon: int = 5,
) -> dict | None:
    """
    Multi-Horizon Temporal Ensemble:

    1. Filtert state_rows in Zeitfenster (short / mid / long)
    2. Führt KNN für jedes Fenster durch (knn_within_state)
    3. Fusioniert Ergebnisse im log-odds Raum mit Unsicherheits-Gewichtung

    Finale Wahrscheinlichkeit:
        logit(p_ensemble) = Σ(w_i × logit(p_i)) / Σ(w_i)
        w_i = config_weight × confidence_i × sample_factor_i

    Gibt None zurück wenn kein Horizont genug Daten hat.

    Returns:
        Alles wie knn_within_state(), plus:
        horizons_detail — Liste mit Ergebnissen pro Horizont
        n_horizons_used — Anzahl genutzter Horizonte
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    horizon_results = []
    total_weight    = 0.0
    lo_weighted_sum = 0.0

    # Akkumulatoren für Multi-Target (weighted avg)
    exp_close_acc = 0.0
    exp_high_acc  = 0.0
    exp_low_acc   = 0.0
    exp_weight    = 0.0

    for h in horizons:
        rows = _filter_rows_by_horizon(state_rows, h.get('days'))

        if len(rows) < min_samples_per_horizon:
            horizon_results.append({
                'label':  h['label'],
                'days':   h['days'],
                'status': 'too_few_samples',
                'n':      len(rows),
            })
            continue

        result = knn_within_state(
            current_features, rows,
            k=k,
            feature_weights=feature_weights,
            gap_factor=gap_factor,
        )
        if result is None:
            horizon_results.append({'label': h['label'], 'status': 'knn_none'})
            continue

        p_up       = result['p_up']
        confidence = result['confidence']
        n_samples  = result['k_used']

        # Gewichtung: config × Konfidenz × Datenbasis-Faktor
        sample_factor = min(1.0, n_samples / 10.0)   # voll bei ≥10 Nachbarn
        w_final = float(h['weight']) * confidence * sample_factor

        if w_final < 1e-6:
            horizon_results.append({'label': h['label'], 'status': 'zero_weight'})
            continue

        # Log-odds Fusion
        lo_weighted_sum += w_final * _logit(p_up)
        total_weight    += w_final

        # Multi-Target linear gewichtet
        exp_close_acc += w_final * result['expected_close_pct']
        exp_high_acc  += w_final * result['expected_high_pct']
        exp_low_acc   += w_final * result['expected_low_pct']
        exp_weight    += w_final

        horizon_results.append({
            'label':       h['label'],
            'days':        h['days'],
            'status':      'applied',
            'n':           len(rows),
            'k_used':      n_samples,
            'p_up':        p_up,
            'confidence':  confidence,
            'w_config':    h['weight'],
            'w_final':     w_final,
            'gap_detected': result.get('gap_detected', False),
        })

    if total_weight < 1e-6:
        logger.debug("Temporal Ensemble: alle Horizonte ohne Gewicht → kein Signal")
        return None

    # Ensemble-Ergebnis
    p_ensemble    = _sigmoid(lo_weighted_sum / total_weight)
    exp_close_pct = exp_close_acc / exp_weight if exp_weight > 0 else 0.0
    exp_high_pct  = exp_high_acc  / exp_weight if exp_weight > 0 else 0.0
    exp_low_pct   = exp_low_acc   / exp_weight if exp_weight > 0 else 0.0

    n_active = sum(1 for h in horizon_results if h.get('status') == 'applied')

    # Ensemble-Konfidenz = gewichteter Mittelwert der Einzel-Konfidenzen
    applied     = [h for h in horizon_results if h.get('status') == 'applied']
    conf_vals   = [h['confidence'] * h['w_final'] for h in applied]
    w_vals      = [h['w_final'] for h in applied]
    ensemble_confidence = sum(conf_vals) / sum(w_vals) if w_vals else 0.0

    # k_used = gewichtetes Mittel der k_used Werte
    k_used_avg = int(round(
        sum(h['k_used'] * h['w_final'] for h in applied) / sum(w_vals)
        if w_vals else k
    ))

    # neighbor_returns: aus dem stärksten Horizont (höchstes w_final)
    best_horizon = max(applied, key=lambda h: h['w_final'], default=None)
    neighbor_ret = []

    return {
        # KNN-Interface-kompatible Felder
        'p_up':               p_ensemble,
        'p_down':             1.0 - p_ensemble,
        'expected_close_pct': exp_close_pct,
        'expected_high_pct':  exp_high_pct,
        'expected_low_pct':   exp_low_pct,
        'confidence':         ensemble_confidence,
        'k_used':             k_used_avg,
        'k_requested':        k,
        'avg_distance':       0.0,   # nicht sinnvoll über Horizonte
        'neighbor_returns':   neighbor_ret,
        'gap_detected':       any(h.get('gap_detected') for h in applied),
        # Ensemble-spezifische Felder
        'horizons_detail':    horizon_results,
        'n_horizons_used':    n_active,
        'total_weight':       total_weight,
        'ensemble_mode':      True,
    }


def format_ensemble_detail(result: dict) -> list[str]:
    """Formatiert Horizont-Details für den Terminal-Report."""
    lines = []
    for h in result.get('horizons_detail', []):
        if h.get('status') != 'applied':
            lines.append(f"  {h['label']:<8}  {'[' + h.get('status','?') + ']'}")
        else:
            days_str = f"{h['days']}d" if h.get('days') else "all"
            lines.append(
                f"  {h['label']:<8}  {days_str:<5}  "
                f"P(up)={h['p_up']*100:.1f}%  "
                f"conf={h['confidence']:.2f}  "
                f"w={h['w_final']:.2f}  "
                f"k={h['k_used']}"
                + ("  [Gap]" if h.get('gap_detected') else "")
            )
    return lines
