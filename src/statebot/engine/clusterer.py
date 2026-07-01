# engine/clusterer.py — Markt-Zustandsclusterung
#
# Ablauf:
#   1. Feature-Matrix aus Store laden
#   2. Robust normieren (Median + IQR)
#   3. KMeans mit n_clusters (default 20)
#   4. Zustandsnamen aus Zentroid-Eigenschaften ableiten
#   5. State-IDs in feature_vectors schreiben
#   6. State-Definitions in store speichern

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_samples

from statebot.engine.features import FEATURE_COLS

logger = logging.getLogger(__name__)


# Heuristik-Schwellen für State-Benennung (relativ zu Median über alle States)
_HURST_IDX    = FEATURE_COLS.index('hurst')
_ENTROPY_IDX  = FEATURE_COLS.index('entropy')
_ATR_IDX      = FEATURE_COLS.index('atr_ratio')
_EMA_CROSS_IDX = FEATURE_COLS.index('ema_cross')
_RSI_IDX      = FEATURE_COLS.index('rsi')
_ROC_IDX      = FEATURE_COLS.index('roc')
_FD_IDX       = FEATURE_COLS.index('fd')
_HV20_IDX     = FEATURE_COLS.index('hv20')


def _label_state(centroid: np.ndarray, all_centroids: np.ndarray) -> str:
    """Leitet einen beschreibenden Namen aus dem Zentroid ab."""
    med_atr  = np.median(all_centroids[:, _ATR_IDX])
    med_ent  = np.median(all_centroids[:, _ENTROPY_IDX])
    med_hv20 = np.median(all_centroids[:, _HV20_IDX])

    h   = centroid[_HURST_IDX]
    ent = centroid[_ENTROPY_IDX]
    atr = centroid[_ATR_IDX]
    ec  = centroid[_EMA_CROSS_IDX]
    rsi = centroid[_RSI_IDX]
    roc = centroid[_ROC_IDX]
    hv  = centroid[_HV20_IDX]

    # Reihenfolge: spezifischste zuerst
    if ent > med_ent * 1.40:
        return "CHAOS"
    if atr < med_atr * 0.45:
        return "COMPRESSION"
    if h > 0.58:
        if ec > 0.005:
            return "TREND_UP"
        elif ec < -0.005:
            return "TREND_DOWN"
        else:
            return "TREND_FLAT"
    if h < 0.44:
        if atr > med_atr * 1.35:
            return "RANGE_VOLATILE"
        else:
            return "RANGE_QUIET"
    if rsi > 0.65 and roc > 0.02:
        return "MOMENTUM_UP"
    if rsi < 0.35 and roc < -0.02:
        return "MOMENTUM_DOWN"
    if hv > med_hv20 * 1.5:
        return "EXPANSION"
    if hv < med_hv20 * 0.6:
        return "CONTRACTION"
    return "NEUTRAL"


def fit_clusters(feature_matrix: np.ndarray,
                 n_clusters: int = 20,
                 random_state: int = 42) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """
    KMeans Clustering auf normalisierten Feature-Vektoren.
    Gibt (labels, centroids_raw, scaler) zurück.
    centroids_raw ist in den Original-Feature-Einheiten (nicht skaliert).
    """
    valid_mask = ~np.any(np.isnan(feature_matrix), axis=1)
    X_valid    = feature_matrix[valid_mask]

    if len(X_valid) < n_clusters * 3:
        n_clusters = max(2, len(X_valid) // 3)
        logger.warning(f"Zu wenige Samples → n_clusters auf {n_clusters} reduziert")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_valid)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

    logger.info(f"KMeans mit n_clusters={n_clusters} auf {len(X_valid)} Samples...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=10, max_iter=300)
    km.fit(X_scaled)

    # Silhouette-Score pro Cluster (State Quality Score)
    sil_per_sample = silhouette_samples(X_scaled, km.labels_)
    quality_scores = {}
    for s_id in range(n_clusters):
        mask = km.labels_ == s_id
        quality_scores[s_id] = float(np.mean(sil_per_sample[mask])) if mask.sum() > 1 else 0.0

    # Labels für alle Zeilen (NaN-Zeilen bekommen State -1)
    all_labels = np.full(len(feature_matrix), -1, dtype=int)
    all_labels[valid_mask] = km.labels_

    # Zentroide zurück in Original-Einheiten
    centroids_raw = scaler.inverse_transform(km.cluster_centers_)

    return all_labels, centroids_raw, scaler, quality_scores


def assign_state_to_vector(feature_vector: np.ndarray,
                            centroids_raw: np.ndarray) -> int:
    """
    Zustandszuweisung zur Inferenzzeit:
    Nächstes Zentroid per euklidischer Distanz (ohne Scaler nötig).
    Normiert beide Seiten durch robusten Vektor-Median.
    """
    feat = np.nan_to_num(feature_vector, nan=0.0)
    cent = np.nan_to_num(centroids_raw,  nan=0.0)
    # Spalten-Median als Normierungsmaßstab
    col_range = np.abs(np.median(cent, axis=0)) + 1e-6
    feat_n = feat / col_range
    cent_n = cent / col_range
    dists  = np.linalg.norm(cent_n - feat_n, axis=1)
    return int(np.argmin(dists))


def compute_membership_score(feature_vector: np.ndarray,
                              state_id: int,
                              centroids_raw: np.ndarray,
                              state_rows: list[dict]) -> float:
    """
    Wie gut liegt der HEUTIGE Punkt innerhalb seines Clusters?

    Beantwortet: Liegt der aktuelle Punkt im Kern (typisch) oder am Rand (atypisch)?

    Methode: Percentile-Rank
      1. Berechne Distanz des aktuellen Punktes zum Zentroid
      2. Berechne Distanzen aller Cluster-Mitglieder zum Zentroid
      3. membership = Anteil der Mitglieder, die WEITER entfernt sind

    Interpretation:
      0.95 → Punkt ist näher als 95% der Mitglieder (Kern des Clusters)
      0.28 → Punkt liegt am Rand (atypisch für diesen State)

    Unterschied zu quality_score (Silhouette):
      quality_score = Wie gut ist der Cluster INSGESAMT definiert?
      membership    = Wie gut passt DIESER EINE PUNKT in seinen Cluster?
    """
    centroid  = centroids_raw[state_id]
    col_range = np.abs(np.median(centroids_raw, axis=0)) + 1e-6

    feat_n = np.nan_to_num(feature_vector, nan=0.0) / col_range
    cent_n = centroid / col_range
    d_current = float(np.linalg.norm(feat_n - cent_n))

    member_dists = []
    for row in state_rows:
        row_arr = np.array([row.get(c, np.nan) for c in FEATURE_COLS], dtype=np.float64)
        if not np.any(np.isnan(row_arr)):
            row_n = row_arr / col_range
            member_dists.append(float(np.linalg.norm(row_n - cent_n)))

    if len(member_dists) < 3:
        return 0.5

    member_dists_arr = np.array(member_dists)
    membership = float(np.mean(member_dists_arr >= d_current))
    return membership


def build_state_labels(store, market: str, tf: str,
                        n_clusters: int = 20) -> int:
    """
    Vollständiger Clustering-Durchlauf:
    1. Feature-Matrix aus Store
    2. KMeans
    3. State-IDs in DB schreiben
    4. State-Definitions berechnen + speichern
    Gibt Anzahl geclusterter Samples zurück.
    """
    rows = store.get_all_vectors(market, tf)
    if not rows:
        logger.warning(f"Keine Vektoren für {market}/{tf}")
        return 0

    feat_matrix = np.array(
        [[r.get(c, np.nan) for c in FEATURE_COLS] for r in rows],
        dtype=np.float64
    )

    labels, centroids_raw, _, quality_scores = fit_clusters(feat_matrix, n_clusters)

    # State-IDs in DB schreiben (Batch)
    for i, row in enumerate(rows):
        state_id = int(labels[i]) if labels[i] >= 0 else None
        if state_id is not None:
            store.assign_state(market, tf, row['bar_time'], state_id)
    store.commit()

    # State-Definitions berechnen
    all_centroids = centroids_raw
    n_states = len(centroids_raw)

    for s_id in range(n_states):
        state_rows = [r for r, lbl in zip(rows, labels) if lbl == s_id
                      and r.get('next_close_pct') is not None]
        n_samp = len(state_rows)
        returns = [r['next_close_pct'] for r in state_rows]
        avg_ret = float(np.mean(returns)) if returns else 0.0
        std_ret = float(np.std(returns))  if returns else 0.0
        up_prob = float(np.mean([1 if r > 0 else 0 for r in returns])) if returns else 0.5

        name          = _label_state(centroids_raw[s_id], all_centroids)
        quality_score = quality_scores.get(s_id, 0.5)

        grade = ("HIGH" if quality_score >= 0.60 else
                 "MED"  if quality_score >= 0.35 else
                 "LOW"  if quality_score >= 0.20 else "POOR")
        logger.debug(f"  State {s_id:2d}  {name:<18}  n={n_samp:4d}  sil={quality_score:.3f}  [{grade}]")

        store.upsert_state_definition(
            market, tf, s_id, name,
            centroids_raw[s_id].tolist(),
            n_samp, avg_ret, std_ret, up_prob,
            quality_score=quality_score,
        )

    clustered = int(np.sum(labels >= 0))
    logger.info(f"Clustering abgeschlossen: {n_states} States, {clustered} Samples geclustert")
    return clustered
