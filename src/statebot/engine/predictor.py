# engine/predictor.py — Vollständige Vorhersage-Pipeline
#
# Probabilistische Rollenstruktur:
#   Markov → PRIOR      P(up | State)             ─┐  fuse_prior_likelihood()
#   KNN    → LIKELIHOOD P(ähnliche Bars)           ─┘
#   Extern → EVIDENCE   Funding, OI, Macro, ...   → SignalPipeline.fuse()
#
# Ebene 1: State bestimmen (feature_vector → nearest centroid)
# Ebene 2: State Quality Check (Silhouette → kein Signal wenn unscharf)
# Ebene 3: KNN innerhalb desselben States
# Ebene 4: Markov × KNN → proper Bayesian Fusion (Prior × Likelihood)
# Ebene 5: Externe Signal-Pipeline (Funding/OI/Macro wenn aktiv)
# Ebene 6: Eigenmode-Analyse (PCA auf Feature-Kovarianzmatrix)
# Ebene 7: Multi-Target-Output (Close, High, Low)

import logging
import numpy as np

from statebot.engine.features        import FEATURE_COLS, get_feature_vector, feature_dict_to_array
from statebot.engine.clusterer       import assign_state_to_vector, compute_membership_score
from statebot.engine.transitions     import (get_top_transitions, state_up_probability,
                                              state_up_probability_order2)
from statebot.engine.matcher         import knn_within_state, quality_stars
from statebot.engine.signal_pipeline import SignalPipeline, fuse_prior_likelihood
from statebot.engine.ensemble        import temporal_knn_ensemble, format_ensemble_detail

logger = logging.getLogger(__name__)


def _compute_eigenmodes(feature_matrix: np.ndarray) -> list[dict]:
    """PCA auf Feature-Kovarianzmatrix → Top-4 Eigenmodes."""
    if len(feature_matrix) < 10:
        return []
    try:
        X = feature_matrix - np.nanmean(feature_matrix, axis=0)
        X = np.nan_to_num(X, nan=0.0)
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sortierung: absteigend
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        total = max(eigenvalues.sum(), 1e-10)

        modes = []
        for i in range(min(4, len(eigenvalues))):
            ev   = eigenvectors[:, i]
            dom  = int(np.argmax(np.abs(ev)))
            modes.append({
                'mode':             i + 1,
                'explained_pct':    float(eigenvalues[i] / total * 100),
                'dominant_feature': FEATURE_COLS[dom] if dom < len(FEATURE_COLS) else '?',
            })
        return modes
    except Exception as e:
        logger.debug(f"Eigenmode-Berechnung fehlgeschlagen: {e}")
        return []


def predict(df, params: dict, store) -> dict | None:
    """
    Vollständige Vorhersage für die aktuelle Marktlage.

    Returns:
        state_id, state_name, regime-Info,
        p_prior, p_knn, p_bayes,
        expected_close/high/low_pct,
        expected_close/high/low (absolut),
        confidence, k_used, stars,
        top_transitions, eigenmodes,
        feature_vector, prediction_id
    """
    market    = params['market']['symbol']
    tf        = params['market']['timeframe']
    knn_cfg   = params.get('knn', {})
    k         = knn_cfg.get('k', 20)
    fw        = knn_cfg.get('feature_weights')
    fw_arr    = np.array(fw) if fw else None

    # ── Ebene 1: Feature-Vektor ────────────────────────────────────────────────
    from statebot.engine.features import compute_features
    df_feat = compute_features(df)
    fvec    = get_feature_vector(df_feat, -2)   # Letzte geschlossene Kerze
    if fvec is None:
        logger.info("Feature-Vektor enthält NaN (Warmup-Phase).")
        return None

    current_arr = feature_dict_to_array(fvec)
    current_close = float(df['close'].iloc[-2])
    prediction_id = str(df.index[-2]) if hasattr(df.index[-2], 'isoformat') else str(df.index[-2])

    # ── Ebene 2: State zuweisen ────────────────────────────────────────────────
    state_defs = store.get_state_definitions(market, tf)
    if not state_defs:
        logger.warning("Keine State-Definitionen gefunden. Zuerst build_states.py laufen lassen.")
        return None

    centroids = np.array([sd['centroid'] for sd in state_defs], dtype=np.float64)
    state_id  = assign_state_to_vector(current_arr, centroids)
    state_def = next((sd for sd in state_defs if sd['state_id'] == state_id), None)
    state_name = state_def['name'] if state_def else f"STATE_{state_id}"
    n_state_samples = state_def['n_samples'] if state_def else 0

    # Regime-Eigenschaften
    h_idx = FEATURE_COLS.index('hurst')
    ent_idx = FEATURE_COLS.index('entropy')
    atr_idx = FEATURE_COLS.index('atr_ratio')
    hurst_val  = float(current_arr[h_idx])
    entropy_val = float(current_arr[ent_idx])
    atr_val    = float(current_arr[atr_idx])
    regime = ("TREND" if hurst_val > 0.55
              else "RANGE" if hurst_val < 0.45
              else "NEUTRAL")
    volatility = ("HIGH" if atr_val > np.median(centroids[:, atr_idx]) * 1.3
                  else "LOW" if atr_val < np.median(centroids[:, atr_idx]) * 0.7
                  else "MED")

    # ── Ebene 2a: Cluster Quality (Silhouette — Cluster als Ganzes) ───────────
    quality_score = float(state_def.get('quality_score', 1.0) if state_def else 1.0)
    min_quality   = knn_cfg.get('min_state_quality', 0.20)
    quality_label = ("HIGH" if quality_score >= 0.60 else
                     "MED"  if quality_score >= 0.35 else
                     "LOW"  if quality_score >= min_quality else "POOR")
    if quality_score < min_quality:
        logger.info(
            f"State {state_id} ({state_name}) Silhouette={quality_score:.2f} < "
            f"{min_quality} — kein Signal (Cluster zu unscharf)"
        )
        return None

    # K proportional zur State-Qualität reduzieren wenn Cluster unscharf
    k_adjusted = k if quality_score >= 0.50 else max(5, round(k * (quality_score / 0.50)))
    if k_adjusted < k:
        logger.debug(f"State {state_id} Qualität={quality_score:.2f} → k={k_adjusted}")

    # ── Ebene 2b: Markov-Prior (Order-1 + optional Order-2) ───────────────────
    use_order2 = knn_cfg.get('use_markov_order2', True)
    p_prior_order1 = state_up_probability(store, market, tf, state_id)
    p_prior        = p_prior_order1
    markov_order   = 1

    if use_order2:
        prev_state = store.get_previous_state(market, tf)
        if prev_state is not None:
            p_order2 = state_up_probability_order2(
                store, market, tf, prev_state, state_id,
                min_samples=knn_cfg.get('markov_order2_min_samples', 10),
            )
            if p_order2 is not None:
                # Blend: Order-2 hat mehr Information wenn genug Daten
                p_prior    = 0.60 * p_order2 + 0.40 * p_prior_order1
                markov_order = 2
                logger.debug(f"Order-2 Markov: p1={p_prior_order1:.2f} p2={p_order2:.2f} → {p_prior:.2f}")

    top_trans = get_top_transitions(store, market, tf, state_id, top_n=3)

    # ── Ebene 3: KNN innerhalb State (mit dynamischem K) ──────────────────────
    min_vectors = knn_cfg.get('min_vectors', 30)
    if n_state_samples < min_vectors:
        logger.info(f"State {state_id} hat nur {n_state_samples} Samples (Min: {min_vectors}). Kein Signal.")
        return None

    state_rows = store.get_vectors_in_state(market, tf, state_id)

    # ── Ebene 2c: Membership Score (aktueller Punkt im Cluster) ───────────────
    membership = compute_membership_score(current_arr, state_id, centroids, state_rows)
    membership_label = ("KERN"  if membership >= 0.70 else
                        "MITTE" if membership >= 0.40 else "RAND")

    gap_factor    = float(knn_cfg.get('knn_gap_factor', 0.0))
    use_ensemble  = knn_cfg.get('use_temporal_ensemble', False)
    horizon_cfg   = knn_cfg.get('ensemble_horizons', None)
    ensemble_mode = False

    if use_ensemble:
        knn_result = temporal_knn_ensemble(
            current_arr, state_rows,
            k=k_adjusted, feature_weights=fw_arr,
            gap_factor=gap_factor,
            horizons=horizon_cfg,
        )
        if knn_result is not None:
            ensemble_mode = True
    else:
        knn_result = None

    if knn_result is None:
        knn_result = knn_within_state(current_arr, state_rows,
                                       k=k_adjusted, feature_weights=fw_arr,
                                       gap_factor=gap_factor)
    if knn_result is None:
        logger.info(f"KNN innerhalb State {state_id} lieferte kein Ergebnis.")
        return None

    p_knn = knn_result['p_up']
    if knn_result.get('gap_detected'):
        logger.debug(f"KNN Gap: k_requested={knn_result['k_requested']} → k_used={knn_result['k_used']}")

    # ── Ebene 4: Markov × KNN → Bayesian Fusion (Prior × Likelihood) ──────────
    # Markov ist der PRIOR, KNN ist die LIKELIHOOD.
    # fuse_prior_likelihood() macht proper Bayes im log-odds Raum.
    use_bayes     = knn_cfg.get('use_bayes', True)
    signal_config = knn_cfg.get('signal_pipeline', None)
    signal_trace  = []

    # Defaults — werden im use_bayes-Block überschrieben wenn Calibrator aktiv
    prior_reliability: float = 0.85
    knn_reliability:   float = 0.90

    if use_bayes:
        use_dynamic       = knn_cfg.get('use_dynamic_reliability', False)
        calibrator        = getattr(store, 'calibrator', None) if use_dynamic else None

        # Dynamische Reliabilities aus Calibrator (oder statisch)
        prior_reliability = (calibrator.get_reliability('markov', market, tf, 0.85)
                             if calibrator else 0.85)
        knn_reliability   = (calibrator.get_reliability('knn', market, tf, 0.90)
                             if calibrator else 0.90)

        p_posterior = fuse_prior_likelihood(p_prior, p_knn,
                                             prior_reliability=prior_reliability,
                                             knn_reliability=knn_reliability)

        # ── Ebene 5: Externe Evidence-Pipeline ────────────────────────────────
        # Funding, OI, Makro etc. als zusätzliche Evidenz auf dem Posterior
        pipeline = SignalPipeline.default(
            calibrator=calibrator,
            market=market,
            timeframe=tf,
            signal_config=signal_config,
        )
        pipeline_context = {
            'p_knn':         p_knn,
            'p_prior':       p_prior,
            'state_id':      state_id,
            'state_name':    state_name,
            'regime':        regime,
            'prediction_id': prediction_id,
        }
        fusion       = pipeline.fuse(pipeline_context, start_probability=p_posterior)
        p_bayes      = fusion['p_final']
        signal_trace = [
            {'name': 'markov', 'p_signal': p_prior,     'reliability': prior_reliability,
             'status': 'applied', 'role': 'prior'},
            {'name': 'knn',    'p_signal': p_knn,        'reliability': knn_reliability,
             'status': 'applied', 'role': 'likelihood',  'posterior': p_posterior},
        ] + fusion['signal_trace']
    else:
        p_bayes = p_knn

    # ── Ebene 5: Multi-Target (absolut) ───────────────────────────────────────
    exp_close_pct = knn_result['expected_close_pct']
    exp_high_pct  = knn_result['expected_high_pct']
    exp_low_pct   = knn_result['expected_low_pct']

    exp_close = current_close * (1 + exp_close_pct / 100)
    exp_high  = current_close * (1 + exp_high_pct  / 100)
    exp_low   = current_close * (1 + exp_low_pct   / 100)

    # ── Ebene 6: Eigenmodes ────────────────────────────────────────────────────
    labeled_rows = store.get_labeled_vectors(market, tf)
    if labeled_rows:
        feat_m = np.array([[r.get(c, np.nan) for c in FEATURE_COLS] for r in labeled_rows])
        eigenmodes = _compute_eigenmodes(feat_m)
    else:
        eigenmodes = []

    # ── Qualitätssterne ───────────────────────────────────────────────────────
    stars = quality_stars(p_bayes, knn_result['confidence'], knn_result['k_used'], n_state_samples)

    return {
        # State
        'state_id':          state_id,
        'state_name':        state_name,
        'n_state_samples':   n_state_samples,
        'regime':            regime,
        'volatility':        volatility,
        'hurst':             hurst_val,
        'entropy':           entropy_val,
        'atr_ratio':         atr_val,
        # Wahrscheinlichkeiten
        'p_prior':           p_prior,
        'p_knn':             p_knn,
        'p_bayes':           p_bayes,
        'use_bayes':         use_bayes,
        'signal_trace':      signal_trace,
        # KNN
        'confidence':        knn_result['confidence'],
        'k_used':            knn_result['k_used'],
        'avg_distance':      knn_result['avg_distance'],
        'neighbor_returns':  knn_result['neighbor_returns'],
        # Multi-Target (%)
        'expected_close_pct': exp_close_pct,
        'expected_high_pct':  exp_high_pct,
        'expected_low_pct':   exp_low_pct,
        # Multi-Target (absolut)
        'expected_close':    exp_close,
        'expected_high':     exp_high,
        'expected_low':      exp_low,
        'current_close':     current_close,
        # Qualität
        'stars':             stars,
        'quality_score':     quality_score,
        'quality_label':     quality_label,
        'membership':        membership,
        'membership_label':  membership_label,
        'k_adjusted':        k_adjusted,
        'k_requested':       knn_result.get('k_requested', k),
        'gap_detected':      knn_result.get('gap_detected', False),
        'markov_order':      markov_order,
        'ensemble_mode':     ensemble_mode,
        'ensemble_detail':   knn_result.get('horizons_detail', []),
        'n_horizons_used':   knn_result.get('n_horizons_used', 1),
        # Reliabilities (für Attribution)
        'reliability_markov': prior_reliability,
        'reliability_knn':    knn_reliability,
        # Transitions
        'top_transitions':   top_trans,
        # Eigenmodes
        'eigenmodes':        eigenmodes,
        # Für Self-Learning
        'feature_vector':    fvec,
        'prediction_id':     prediction_id,
    }


def format_prediction_report(pred: dict) -> str:
    """Formatiert die Vorhersage als lesbaren Terminal-Report."""
    if pred is None:
        return "Kein Signal."

    cc    = pred.get('current_close', 0)
    stars = '★' * pred.get('stars', 1) + '☆' * (5 - pred.get('stars', 1))

    def _line(label, value, width=26):
        return f"  {label:{width}}{value}"

    lines = [
        "=" * 56,
        f"  STATEBOT  |  Vorhersage",
        "=" * 56,
        "",
        "  MARKTZUSTAND",
        _line("State", f"{pred['state_id']}  ({pred['state_name']})"),
        _line("Cluster-Qualität", f"{pred.get('quality_score', 1.0):.2f}  [{pred.get('quality_label','?')}]"),
        _line("Membership heute",
              f"{pred.get('membership', 0.5):.2f}  [{pred.get('membership_label','?')}]"
              + ("  ← Randlage" if pred.get('membership_label') == 'RAND' else "")),
        _line("Regime", pred['regime']),
        _line("Hurst", f"{pred['hurst']:.3f}"),
        _line("Volatilität", pred['volatility']),
        _line("Entropie", f"{pred['entropy']:.3f}"),
        "",
        "  STATE TRANSITIONS",
    ]

    for t in pred.get('top_transitions', []):
        lines.append(_line(t['name'][:24], f"{t['probability']*100:.1f}%"))

    lines += [
        "",
        "  NÄCHSTE ZUSTANDS-WAHRSCHEINLICHKEIT",
    ]
    for t in pred.get('signal_trace', []):
        if t.get('status') != 'applied':
            continue
        role    = t.get('role', 'evidence')
        rel_str = f"r={t['reliability']:.2f}"
        role_tag = {'prior': '[PRIOR]', 'likelihood': '[LIKELI]'}.get(role, '[EVID] ')
        if role == 'likelihood':
            lines.append(_line(f"  {role_tag} {t['name'][:16]}",
                               f"{t['p_signal']*100:.1f}%  ({rel_str})  → {t['posterior']*100:.1f}%"))
        else:
            p_post_str = f"  → {t['posterior']*100:.1f}%" if 'posterior' in t else ""
            lines.append(_line(f"  {role_tag} {t['name'][:16]}",
                               f"{t['p_signal']*100:.1f}%  ({rel_str}){p_post_str}"))
    lines += [
        _line("P(up) Final" + (" [Bayes]" if pred.get('use_bayes') else " [KNN]"),
              f"{pred['p_bayes']*100:.1f}%  ←"),
        "",
        "  NEAREST STATE NEIGHBORS",
        _line("Markov Ordnung", f"{'2' if pred.get('markov_order') == 2 else '1'}  (P_prior={pred['p_prior']*100:.1f}%)"),
        _line("KNN Modus",
              ("Temporal Ensemble" if pred.get('ensemble_mode') else "Single Window")
              + (f"  [{pred.get('n_horizons_used', 1)} Horizonte]" if pred.get('ensemble_mode') else "")),
        _line("K angefordert / genutzt",
              f"{pred.get('k_requested', pred['k_used'])} / {pred['k_used']}"
              + ("  [Gap]" if pred.get('gap_detected') else "")),
        _line("State-Samples", pred['n_state_samples']),
        _line("Avg. Distanz", f"{pred['avg_distance']:.3f}"),
        "",
        "  MULTI-TARGET VORHERSAGE",
        _line("Expected Close", f"{pred['expected_close_pct']:+.2f}%  =  {pred['expected_close']:.4g}"),
        _line("Expected High",  f"{pred['expected_high_pct']:+.2f}%  =  {pred['expected_high']:.4g}"),
        _line("Expected Low",   f"{pred['expected_low_pct']:+.2f}%  =  {pred['expected_low']:.4g}"),
        _line("Konfidenz", f"{pred['confidence']:.2f}"),
        "",
    ]

    if pred.get('ensemble_mode') and pred.get('ensemble_detail'):
        lines += ["", "  TEMPORAL ENSEMBLE"]
        from statebot.engine.ensemble import format_ensemble_detail
        lines += format_ensemble_detail({'horizons_detail': pred['ensemble_detail']})

    if pred.get('eigenmodes'):
        lines.append("  EIGENMODES")
        for m in pred['eigenmodes'][:4]:
            lines.append(_line(
                f"Mode {m['mode']} ({m['dominant_feature'][:14]})",
                f"{m['explained_pct']:.1f}%"
            ))
        lines.append("")

    lines += [
        "  SIGNAL",
        _line("Qualität", stars),
        "=" * 56,
    ]
    return '\n'.join(lines)
