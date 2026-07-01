# strategy/signal.py — Signal-Generierung aus Vorhersage
#
# Nutzt predictor.predict() und leitet daraus Entry/SL/TP ab.
# Optional: Expected-Low als SL, Expected-High als TP (use_expected_targets)
#
# Ergänzungen (von pbot):
#   HTF-Supertrend Veto: Gegentrend-Trades werden hart geblockt.
#   Structure Protection: SL wird hinter das H/L der vorherigen Kerze gezogen,
#       wenn dieses weiter entfernt liegt als der feste %-SL.

import logging

from statebot.engine.predictor  import predict, format_prediction_report
from statebot.engine.htf_filter import get_htf_trend, apply_htf_veto

logger = logging.getLogger(__name__)


def get_state_signal(df, params: dict, store, htf_df=None) -> dict | None:
    """
    Vollständige Signal-Pipeline.
    Gibt None zurück wenn kein Signal, sonst:
      side, entry_price, sl_price, tp_price, sl_pct, rr_ratio,
      p_bayes, confidence, stars, prediction_id,
      expected_close/high/low, feature_vector, prediction (full dict)

    htf_df — optionaler Higher-Timeframe DataFrame für Supertrend-Veto.
    """
    pred = predict(df, params, store)
    if pred is None:
        return None

    # Terminal-Report ausgeben
    report = format_prediction_report(pred)
    logger.info(f"\n{report}")

    knn_cfg = params.get('knn', {})
    risk    = params.get('risk', {})

    threshold_long  = knn_cfg.get('threshold_long',  0.60)
    threshold_short = knn_cfg.get('threshold_short', 0.40)
    min_confidence  = knn_cfg.get('min_confidence',  0.40)
    min_stars       = knn_cfg.get('min_stars', 2)

    p_bayes    = pred['p_bayes']
    confidence = pred['confidence']
    stars      = pred['stars']

    if confidence < min_confidence:
        logger.info(f"Konfidenz {confidence:.2f} unter Minimum {min_confidence}")
        return None
    if stars < min_stars:
        logger.info(f"Qualität {stars} Sterne unter Minimum {min_stars}")
        return None

    if p_bayes >= threshold_long:
        side = 'long'
    elif p_bayes <= threshold_short:
        side = 'short'
    else:
        logger.info(f"P(up)={p_bayes:.2%} im neutralen Bereich → kein Signal")
        return None

    # ── HTF-Supertrend Veto ────────────────────────────────────────────────────
    htf_trend: int | None = None
    htf_cfg = params.get('htf_filter', {})
    if htf_cfg.get('enabled', False):
        htf_trend = get_htf_trend(
            htf_df,
            period=htf_cfg.get('period', 10),
            factor=htf_cfg.get('factor', 3.0),
        )
        if apply_htf_veto(side, htf_trend):
            return None

    current_price = float(df['close'].iloc[-1])
    rr_ratio = risk.get('rr_ratio', 2.0)

    # ── SL und TP bestimmen ────────────────────────────────────────────────────
    use_expected = risk.get('use_expected_targets', False)

    if use_expected and pred.get('expected_low') and pred.get('expected_high'):
        # Multi-Target-Modus: ML-Vorhersage für SL und TP
        if side == 'long':
            sl_price = pred['expected_low']   * (1 - 0.001)    # kleiner Puffer
            tp_price = pred['expected_high']  * (1 - 0.001)
        else:
            sl_price = pred['expected_high']  * (1 + 0.001)
            tp_price = pred['expected_low']   * (1 + 0.001)

        if sl_price <= 0 or tp_price <= 0:
            use_expected = False

    if not use_expected:
        # Standard-Modus: fixer % SL
        sl_pct_cfg = risk.get('sl_pct', 1.0) / 100.0
        if side == 'long':
            sl_price = current_price * (1 - sl_pct_cfg)
            tp_price = current_price * (1 + sl_pct_cfg * rr_ratio)
        else:
            sl_price = current_price * (1 + sl_pct_cfg)
            tp_price = current_price * (1 - sl_pct_cfg * rr_ratio)

    # ── Structure Protection ────────────────────────────────────────────────────
    # SL wird hinter das H/L der vorherigen Kerze gezogen wenn dieses weiter
    # entfernt liegt als der konfigurierte feste SL.
    # Verhindert SL-Hunting durch enge Stops direkt innerhalb der Vorkerze.
    structure_sl_applied = False
    if risk.get('use_structure_protection', True) and len(df) >= 2:
        prev_low  = float(df['low'].iloc[-2])
        prev_high = float(df['high'].iloc[-2])
        orig_sl   = sl_price

        if side == 'long' and prev_low < sl_price:
            sl_price = prev_low * (1 - 0.001)          # knapp unter prev_low
        elif side == 'short' and prev_high > sl_price:
            sl_price = prev_high * (1 + 0.001)         # knapp über prev_high

        if sl_price != orig_sl:
            structure_sl_applied = True
            new_sl_dist = abs(current_price - sl_price)
            tp_price = (current_price + new_sl_dist * rr_ratio
                        if side == 'long'
                        else current_price - new_sl_dist * rr_ratio)
            logger.debug(
                f"Structure Protection: SL {orig_sl:.5g} → {sl_price:.5g} "
                f"(prev {'low' if side=='long' else 'high'}: "
                f"{prev_low if side=='long' else prev_high:.5g})"
            )

    sl_dist   = abs(current_price - sl_price)
    sl_pct_eff = sl_dist / current_price * 100
    rr_eff    = abs(tp_price - current_price) / sl_dist if sl_dist > 0 else 0

    return {
        'side':                  side,
        'entry_price':           current_price,
        'sl_price':              sl_price,
        'tp_price':              tp_price,
        'sl_pct':                sl_pct_eff,
        'rr_ratio':              rr_eff,
        'p_bayes':               p_bayes,
        'p_prior':               pred['p_prior'],
        'p_knn':                 pred['p_knn'],
        'confidence':            confidence,
        'stars':                 stars,
        'state_id':              pred['state_id'],
        'state_name':            pred['state_name'],
        'regime':                pred['regime'],
        'k_used':                pred['k_used'],
        'membership':            pred.get('membership'),
        'quality_score':         pred.get('quality_score'),
        'htf_trend':             htf_trend,
        'structure_sl_applied':  structure_sl_applied,
        'expected_close':        pred.get('expected_close'),
        'expected_high':         pred.get('expected_high'),
        'expected_low':          pred.get('expected_low'),
        'expected_close_pct':    pred.get('expected_close_pct'),
        'eigenmodes':            pred.get('eigenmodes', []),
        'top_transitions':       pred.get('top_transitions', []),
        'feature_vector':        pred['feature_vector'],
        'prediction_id':         pred['prediction_id'],
        # Vollständiger Snapshot aller Modell-Metriken für Attribution
        'prediction_snapshot':   pred,
    }


def update_prediction_result(store, market: str, tf: str,
                              bar_time: str, actual_return_pct: float):
    """Self-Learning: Tatsächliche Rendite zurück in den Store schreiben."""
    try:
        store.conn.execute("""
            UPDATE feature_vectors
            SET next_close_pct = ?
            WHERE market = ? AND timeframe = ? AND bar_time = ?
        """, (actual_return_pct, market, tf, bar_time))
        store.conn.commit()
    except Exception as e:
        logger.error(f"Self-Learning Update fehlgeschlagen: {e}")
