# engine/matcher.py — KNN innerhalb desselben Marktzustands
#
# Kernunterschied zu knnbot:
#   Wir suchen NICHT in allen historischen Daten,
#   sondern NUR in Bars mit demselben State-ID.
#   Das reduziert Rauschen dramatisch.
#
# Multi-Target: Vorhersage von Close, High UND Low.

import numpy as np
import logging

from statebot.engine.features import FEATURE_COLS

logger = logging.getLogger(__name__)


def _robust_normalize(matrix: np.ndarray,
                       query: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Median + IQR Normierung (outlier-resistent)."""
    median = np.nanmedian(matrix, axis=0)
    q75    = np.nanpercentile(matrix, 75, axis=0)
    q25    = np.nanpercentile(matrix, 25, axis=0)
    iqr    = np.where((q75 - q25) > 1e-8, q75 - q25, 1.0)

    norm_matrix = (matrix - median) / iqr
    norm_query  = (query  - median) / iqr if query is not None else None
    return norm_matrix, norm_query, median, iqr


def _find_gap_cutoff(sorted_distances: np.ndarray, max_k: int,
                      min_k: int = 5, gap_factor: float = 2.5) -> int:
    """
    Findet den natürlichen Stopp-Punkt durch Distanz-Gap-Erkennung.

    Prinzip: Sortiere Distanzen aufsteigend. Wenn d[i] > d[i-1] × gap_factor
    UND wir bereits min_k Nachbarn haben → STOP. Dadurch keine schlechten
    Nachbarn mehr, die nur wegen des festen K mitgenommen werden.

    Beispiel:  0.04, 0.05, 0.05, 0.06, 0.07, 0.18 ← Gap → STOP bei k=5
    """
    k = max_k
    for i in range(1, max_k):
        if i >= min_k and sorted_distances[i] > sorted_distances[i - 1] * gap_factor:
            k = i
            break
    return k


def knn_within_state(
    current_features: np.ndarray,
    state_rows: list[dict],
    k: int = 20,
    feature_weights: np.ndarray | None = None,
    gap_factor: float = 0.0,   # 0 = deaktiviert; z.B. 2.5 = Gap-Erkennung aktiv
) -> dict | None:
    """
    KNN nur innerhalb derselben State-ID.
    Optionale Gap-Erkennung: stoppt automatisch bei großem Distanzsprung.

    Returns:
        p_up              — P(next close > current close)
        p_down            — 1 - p_up
        expected_close_pct — erwartete Close-Rendite (%)
        expected_high_pct  — erwartete High-Rendite (%)
        expected_low_pct   — erwartete Low-Rendite (%)
        confidence         — gewichtete Distanz-Konfidenz
        k_used             — tatsächlich genutzter K (nach Gap-Erkennung)
        k_requested        — ursprünglich angefordertes K
        avg_distance       — mittlere Distanz zu Nachbarn
        neighbor_returns   — Liste der k_used nächsten Close-Renditen
        gap_detected       — True wenn Gap-Stopp ausgelöst wurde
    """
    valid = [r for r in state_rows
             if r.get('next_close_pct') is not None
             and not any(r.get(c) is None for c in FEATURE_COLS)]

    k_requested = k
    k_candidate = min(k, len(valid))
    if k_candidate < 3:
        return None

    feat_matrix = np.array([[r[c] for c in FEATURE_COLS] for r in valid], dtype=np.float64)
    close_ret   = np.array([r['next_close_pct'] for r in valid], dtype=np.float64)
    high_ret    = np.array([r.get('next_high_pct',  r['next_close_pct']) for r in valid], dtype=np.float64)
    low_ret     = np.array([r.get('next_low_pct',   r['next_close_pct']) for r in valid], dtype=np.float64)

    norm_matrix, norm_query, _, _ = _robust_normalize(feat_matrix, current_features)
    norm_query  = np.nan_to_num(norm_query,  nan=0.0)
    norm_matrix = np.nan_to_num(norm_matrix, nan=0.0)

    if feature_weights is not None and len(feature_weights) == norm_matrix.shape[1]:
        diff = (norm_matrix - norm_query) * feature_weights
    else:
        diff = norm_matrix - norm_query

    distances  = np.linalg.norm(diff, axis=1)
    idx_sorted = np.argsort(distances)[:k_candidate]
    all_dists  = distances[idx_sorted]

    # Gap-Erkennung (nur wenn gap_factor > 0)
    if gap_factor > 0 and k_candidate >= 5:
        k_actual  = _find_gap_cutoff(all_dists, k_candidate, gap_factor=gap_factor)
        gap_detected = k_actual < k_candidate
    else:
        k_actual     = k_candidate
        gap_detected = False

    idx_selected = idx_sorted[:k_actual]
    k_dists  = distances[idx_selected]
    k_close  = close_ret[idx_selected]
    k_high   = high_ret[idx_selected]
    k_low    = low_ret[idx_selected]

    inv_dists = 1.0 / (k_dists + 1e-6)
    weights   = inv_dists / inv_dists.sum()

    exp_close = float(np.dot(weights, k_close))
    exp_high  = float(np.dot(weights, k_high))
    exp_low   = float(np.dot(weights, k_low))
    p_up      = float(np.sum(weights[k_close > 0]))

    max_dist   = max(k_dists.max(), 1e-6)
    avg_dist   = float(k_dists.mean())
    confidence = float(1.0 - avg_dist / (max_dist + avg_dist))

    return {
        'p_up':               p_up,
        'p_down':             1.0 - p_up,
        'expected_close_pct': exp_close,
        'expected_high_pct':  exp_high,
        'expected_low_pct':   exp_low,
        'confidence':         confidence,
        'k_used':             k_actual,
        'k_requested':        k_requested,
        'avg_distance':       avg_dist,
        'neighbor_returns':   k_close.tolist(),
        'gap_detected':       gap_detected,
    }


def quality_stars(p_bayes: float, confidence: float, k_used: int, n_state_samples: int) -> int:
    """Qualitätssterne 1–5 für das Signal."""
    # Datenbasis-Faktor
    data_factor = min(k_used / 15.0, 1.0) * min(n_state_samples / 50.0, 1.0)
    # Wahrscheinlichkeits-Abweichung von 0.5
    prob_strength = abs(p_bayes - 0.5) * 2.0   # 0 = 50%, 1 = 100%
    # Kombination
    score = prob_strength * confidence * data_factor
    stars = int(score * 5) + 1
    return max(1, min(5, stars))
