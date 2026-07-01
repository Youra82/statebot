# analysis/backtester.py — Backtesting für statebot
#
# Zwei Modi:
#
#   run_walkforward_backtest()  — Original Walk-Forward (close-to-close, Brier-optimiert)
#       Benötigt nur die StateStore DB. Schnell, kein API-Zugriff nötig.
#
#   run_pnl_backtest()          — PnL-Simulation mit Candle-Auflösung (von pbot übernommen)
#       Benötigt raw OHLCV (load_ohlcv_data() oder extern laden).
#       Pending Orders (Signal bei Close → Entry bei Open der nächsten Kerze).
#       SL/TP-Checks via Candle H/L — kein Close-to-Close-Approximation.
#       Trailing Stop + Structure Protection + Slippage.

import os, sys, json, logging, argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.engine.store           import StateStore
from statebot.engine.features        import FEATURE_COLS
from statebot.engine.clusterer       import assign_state_to_vector, compute_membership_score
from statebot.engine.matcher         import knn_within_state, quality_stars
from statebot.engine.transitions     import state_up_probability
from statebot.engine.signal_pipeline import fuse_prior_likelihood

logger      = logging.getLogger(__name__)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'results')
DB_PATH     = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'states.db')

TRAIN_RATIO  = 0.70
FEE_PCT      = 0.06 / 100
SLIPPAGE_PCT = 0.05 / 100


# ═══════════════════════════════════════════════════════════════════════════════
# MODUS 1: Walk-Forward (close-to-close, original)
# ═══════════════════════════════════════════════════════════════════════════════

def run_walkforward_backtest(store: StateStore, market: str, tf: str,
                              k: int = 20,
                              threshold_long:  float = 0.62,
                              threshold_short: float = 0.38,
                              min_confidence:  float = 0.45,
                              min_stars:       int   = 2,
                              rr_ratio: float = 2.0,
                              sl_pct:   float = 1.5,
                              start_capital: float = 1000.0,
                              risk_per_trade_pct: float = 1.0,
                              leverage: int = 5,
                              allowed_states: list[int] | None = None) -> dict:
    rows = store.get_labeled_vectors(market, tf)
    n    = len(rows)
    if n < 60:
        logger.warning(f"Zu wenige Vektoren ({n}) für {market}/{tf}")
        return {"trades": [], "stats": {}}

    state_defs = store.get_state_definitions(market, tf)
    if not state_defs:
        logger.warning("Keine State-Definitionen — zuerst clustering durchführen")
        return {"trades": [], "stats": {}}

    centroids = np.array([sd['centroid'] for sd in state_defs], dtype=np.float64)
    logger.info(f"[Backtest WF] {market}/{tf} | {n} Vektoren | {len(state_defs)} States | Train={TRAIN_RATIO:.0%}")

    split    = max(20, int(n * (1 - TRAIN_RATIO)))
    trades   = []
    equity   = start_capital
    eq_curve = [equity]

    for test_idx in range(n - split, n - 1):
        train_rows = rows[:test_idx]
        curr_row   = rows[test_idx]

        curr_feat = np.array([curr_row.get(c, np.nan) for c in FEATURE_COLS], dtype=np.float64)
        if np.any(np.isnan(curr_feat)):
            eq_curve.append(equity)
            continue

        train_centroids = _recompute_centroids_from_rows(train_rows, centroids, len(state_defs))
        state_id = assign_state_to_vector(curr_feat, train_centroids)

        if allowed_states is not None and state_id not in allowed_states:
            eq_curve.append(equity)
            continue

        state_rows_train = [r for r in train_rows if r.get('state_id') == state_id]
        if len(state_rows_train) < 5:
            eq_curve.append(equity)
            continue

        knn_result = knn_within_state(curr_feat, state_rows_train, k=k)
        if knn_result is None:
            eq_curve.append(equity)
            continue

        # Prior aus Training-Daten berechnen (kein Look-Ahead via state_def['up_prob'])
        labeled_in_state = [r for r in state_rows_train if r.get('next_close_pct') is not None]
        p_prior = float(np.mean([1.0 if r['next_close_pct'] > 0 else 0.0
                                  for r in labeled_in_state])) if labeled_in_state else 0.5
        p_bayes   = fuse_prior_likelihood(p_prior, knn_result['p_up'])
        confidence = knn_result['confidence']
        stars      = quality_stars(p_bayes, confidence, knn_result['k_used'], len(state_rows_train))

        if confidence < min_confidence or stars < min_stars:
            eq_curve.append(equity)
            continue

        if p_bayes >= threshold_long:
            side = 'long'
        elif p_bayes <= threshold_short:
            side = 'short'
        else:
            eq_curve.append(equity)
            continue

        actual_ret_pct = curr_row.get('next_close_pct', 0) or 0
        sl_dist = sl_pct / 100.0
        tp_dist = sl_dist * rr_ratio
        actual_ret = actual_ret_pct / 100.0

        if side == 'long':
            outcome = 'WIN'  if actual_ret >= tp_dist  else \
                      'LOSS' if actual_ret <= -sl_dist  else 'TIMEOUT'
            pnl_pct = sl_pct * rr_ratio if outcome == 'WIN' else \
                      -sl_pct           if outcome == 'LOSS' else actual_ret_pct
        else:
            outcome = 'WIN'  if actual_ret <= -tp_dist else \
                      'LOSS' if actual_ret >= sl_dist   else 'TIMEOUT'
            pnl_pct = sl_pct * rr_ratio if outcome == 'WIN' else \
                      -sl_pct           if outcome == 'LOSS' else -actual_ret_pct

        cost_pct    = (FEE_PCT * 2 + SLIPPAGE_PCT) * 100
        net_pnl_pct = pnl_pct - cost_pct
        risk_amount = equity * (risk_per_trade_pct / 100.0)
        pos_size    = min(risk_amount / (sl_pct / 100.0), equity * leverage)
        actual_pnl  = pos_size * (net_pnl_pct / 100.0)
        equity     += actual_pnl

        trades.append({
            'bar_time':     curr_row['bar_time'],
            'side':         side,
            'state_id':     state_id,
            'outcome':      outcome,
            'pnl_pct':      net_pnl_pct,
            'pnl_usdt':     actual_pnl,
            'equity_after': equity,
            'p_bayes':      p_bayes,
            'confidence':   confidence,
            'stars':        stars,
        })
        eq_curve.append(equity)

    stats = _compute_stats(trades, eq_curve, start_capital)
    logger.info(
        f"[Backtest WF] {stats.get('total_trades',0)} Trades | "
        f"WR: {stats.get('win_rate',0):.1%} | "
        f"PnL: {stats.get('total_pnl_usdt',0):+.2f} USDT | "
        f"DD: {stats.get('max_drawdown_pct',0):.1f}%"
    )
    return {"trades": trades, "stats": stats, "equity_curve": eq_curve}


# ═══════════════════════════════════════════════════════════════════════════════
# MODUS 2: PnL-Simulation mit Candle-Auflösung (von pbot übernommen)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ohlcv_data(market: str, tf: str,
                    start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lädt OHLCV-Daten aus CSV-Cache oder Bitget API.
    Cache-Pfad: artifacts/data/cache/<symbol>_<tf>.csv
    """
    data_dir  = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'cache')
    os.makedirs(data_dir, exist_ok=True)
    sym_safe  = market.replace('/', '-').replace(':', '-')
    cache_file = os.path.join(data_dir, f"{sym_safe}_{tf}.csv")

    req_start = pd.to_datetime(start_date, utc=True)
    req_end   = pd.to_datetime(end_date,   utc=True)

    if os.path.exists(cache_file):
        try:
            data = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
            data.index = pd.to_datetime(data.index, utc=True)
            if data.index.min() <= req_start and data.index.max() >= req_end:
                return data.loc[req_start:req_end]
        except Exception:
            pass

    # Cache miss → API
    logger.info(f"Lade {market} ({tf}) von Bitget API ({start_date} – {end_date})...")
    try:
        sec_path = os.path.join(PROJECT_ROOT, 'secret.json')
        with open(sec_path) as f:
            secrets = json.load(f)

        from statebot.utils.exchange import Exchange
        acc = (secrets.get('statebot') or secrets.get('ltbbot') or [])[0]
        exchange = Exchange(acc)
        df = exchange.fetch_historical_ohlcv(market, tf, start_date, end_date)
        if not df.empty:
            df.to_csv(cache_file)
            return df.loc[req_start:req_end]
    except Exception as e:
        logger.error(f"OHLCV-Laden fehlgeschlagen: {e}")

    return pd.DataFrame()


def run_pnl_backtest(store: StateStore,
                     ohlcv_df: pd.DataFrame,
                     market: str,
                     tf: str,
                     k: int = 20,
                     threshold_long:  float = 0.62,
                     threshold_short: float = 0.38,
                     min_confidence:  float = 0.45,
                     min_stars:       int   = 2,
                     sl_pct:          float = 1.5,
                     rr_ratio:        float = 2.0,
                     atr_mult_sl:     float = 2.0,
                     use_structure_protection: bool = True,
                     trailing_act_rr:          float = 1.5,
                     trailing_callback_pct:    float = 0.5,
                     start_capital:      float = 1000.0,
                     risk_per_trade_pct: float = 1.0,
                     leverage:           int   = 5,
                     max_hold_bars:      int   = 10) -> dict:
    """
    PnL-Simulation mit realistischer Candle-Auflösung.

    Signal bei Bar-Close → Pending Entry bei Open der nächsten Kerze.
    SL/TP via Candle H/L (nicht nur Close-to-Close).
    Trailing Stop + Structure Protection + Slippage.

    METHODISCHE EINSCHRÄNKUNG — CLUSTERING:
      Die Centroid-Struktur (welche States existieren, wie sie definiert sind)
      wird einmalig offline mit ALLEN verfügbaren Daten gelernt und hier als
      statischer Zustandsraum verwendet. Das ist formal eine leichte Form von
      Look-Ahead: ein Modell in 2018 hätte die Cluster-Geometrie aus 2021
      noch nicht kennen können.

      Richtungsmodelle (KNN p_up, Markov-Prior) werden dagegen ausschließlich
      aus dem inkrementellen Train-Set berechnet und sind sauber gegen
      Look-Ahead abgesichert.

      Begründung für den Kompromiss: Clustering lernt strukturelle Merkmale
      (Hurst, ATR-Ratio, Entropie) — keine Richtungen. Der Bias ist deutlich
      schwächer als bei Label-basiertem Look-Ahead. Später ggf. mit
      periodischem Re-Clustering (alle 6 Monate) vergleichen.

    ohlcv_df — raw OHLCV, index=timestamp (datetime, UTC), Spalten: open/high/low/close.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        logger.error("run_pnl_backtest: ohlcv_df leer — keine Simulation möglich")
        return {"trades": [], "stats": {}}

    rows = store.get_labeled_vectors(market, tf)
    if not rows:
        logger.warning("Keine Feature-Vektoren im Store")
        return {"trades": [], "stats": {}}

    state_defs = store.get_state_definitions(market, tf)
    if not state_defs:
        logger.warning("Keine State-Definitionen vorhanden")
        return {"trades": [], "stats": {}}

    centroids = np.array([sd['centroid'] for sd in state_defs], dtype=np.float64)

    # OHLCV als dict: bar_time_key → row für schnellen Zugriff
    ohlcv_index = list(ohlcv_df.index)
    ohlcv_rows  = ohlcv_df.to_dict('records')
    # Baue bar_time→position Lookup (key = ISO date[:16])
    ohlcv_pos: dict[str, int] = {}
    for pos, ts in enumerate(ohlcv_index):
        key = str(ts)[:16].replace('T', ' ')
        ohlcv_pos[key] = pos

    # OOS-Split: letzte TRAIN_RATIO für Training, Rest für Test
    n     = len(rows)
    split = max(20, int(n * (1 - TRAIN_RATIO)))
    test_start_idx = n - split

    logger.info(
        f"[PnL-Backtest] {market}/{tf} | {n} Vektoren | "
        f"Test-Bars: {split} | {len(ohlcv_rows)} OHLCV-Kerzen"
    )

    trades:   list[dict] = []
    equity   = start_capital
    eq_curve = [equity]
    train_rows = rows[:test_start_idx]

    for test_idx in range(test_start_idx, n - 1):
        curr_row = rows[test_idx]
        bar_time = curr_row.get('bar_time', '')

        # ── Feature-Vektor + State ─────────────────────────────────────────────
        curr_feat = np.array([curr_row.get(c, np.nan) for c in FEATURE_COLS], dtype=np.float64)
        if np.any(np.isnan(curr_feat)):
            continue

        train_centroids  = _recompute_centroids_from_rows(train_rows, centroids, len(state_defs))
        state_id         = assign_state_to_vector(curr_feat, train_centroids)
        state_rows_train = [r for r in train_rows if r.get('state_id') == state_id]
        if len(state_rows_train) < 5:
            continue

        knn_result = knn_within_state(curr_feat, state_rows_train, k=k)
        if knn_result is None:
            continue

        state_def  = next((sd for sd in state_defs if sd['state_id'] == state_id), None)
        # Prior aus Training-Daten berechnen (kein Look-Ahead via state_def['up_prob'])
        labeled_in_state = [r for r in state_rows_train if r.get('next_close_pct') is not None]
        if labeled_in_state:
            p_prior = float(np.mean([1.0 if r['next_close_pct'] > 0 else 0.0
                                     for r in labeled_in_state]))
        else:
            p_prior = 0.5
        p_bayes    = fuse_prior_likelihood(p_prior, knn_result['p_up'])
        confidence = knn_result['confidence']
        stars      = quality_stars(p_bayes, confidence, knn_result['k_used'], len(state_rows_train))

        if confidence < min_confidence or stars < min_stars:
            continue

        if p_bayes >= threshold_long:
            side = 'long'
        elif p_bayes <= threshold_short:
            side = 'short'
        else:
            continue

        # ── OHLCV-Position für diesen Bar finden ──────────────────────────────
        bt_key   = bar_time[:16].replace('T', ' ')
        ohlcv_at = ohlcv_pos.get(bt_key)
        if ohlcv_at is None or ohlcv_at + 1 >= len(ohlcv_rows):
            continue

        signal_candle = ohlcv_rows[ohlcv_at]   # Kerze mit Signal (Close)

        # ── ATR aus den letzten 14 Kerzen berechnen ────────────────────────────
        atr_start = max(0, ohlcv_at - 14)
        atr_candles = ohlcv_rows[atr_start: ohlcv_at + 1]
        if len(atr_candles) >= 2:
            ranges = [c['high'] - c['low'] for c in atr_candles]
            atr_val = float(np.mean(ranges))
        else:
            atr_val = float(signal_candle['close']) * 0.015

        # ── Entry bei Open der nächsten Kerze + Slippage ───────────────────────
        next_candle  = ohlcv_rows[ohlcv_at + 1]
        raw_entry    = float(next_candle['open'])
        entry_price  = (raw_entry * (1 + SLIPPAGE_PCT)
                        if side == 'long'
                        else raw_entry * (1 - SLIPPAGE_PCT))

        # ── SL berechnen: ATR-basiert ──────────────────────────────────────────
        sl_dist_atr  = atr_val * atr_mult_sl
        sl_dist_pct  = entry_price * (sl_pct / 100.0)
        sl_dist      = max(sl_dist_atr, sl_dist_pct)

        if side == 'long':
            sl_price = entry_price - sl_dist
        else:
            sl_price = entry_price + sl_dist

        # ── Structure Protection: SL hinter vorherige Kerze ───────────────────
        structure_sl_applied = False
        if use_structure_protection:
            prev_low  = float(signal_candle['low'])
            prev_high = float(signal_candle['high'])
            orig_sl   = sl_price
            if side == 'long' and prev_low < sl_price:
                sl_price = prev_low * (1 - 0.001)
            elif side == 'short' and prev_high > sl_price:
                sl_price = prev_high * (1 + 0.001)
            if sl_price != orig_sl:
                structure_sl_applied = True
                sl_dist = abs(entry_price - sl_price)

        tp_price          = (entry_price + sl_dist * rr_ratio
                             if side == 'long'
                             else entry_price - sl_dist * rr_ratio)
        activation_price  = (entry_price + sl_dist * trailing_act_rr
                             if side == 'long'
                             else entry_price - sl_dist * trailing_act_rr)
        callback_rate     = trailing_callback_pct / 100.0

        # ── Trade-Simulation über die nächsten Kerzen ─────────────────────────
        trailing_active = False
        peak_price      = entry_price
        exit_price      = None
        outcome         = 'TIMEOUT'

        for j in range(ohlcv_at + 1, min(ohlcv_at + 1 + max_hold_bars, len(ohlcv_rows))):
            c     = ohlcv_rows[j]
            c_hi  = float(c['high'])
            c_lo  = float(c['low'])

            if side == 'long':
                # Trailing Stop aktualisieren
                if not trailing_active and c_hi >= activation_price:
                    trailing_active = True
                if trailing_active:
                    peak_price = max(peak_price, c_hi)
                    new_sl     = peak_price * (1 - callback_rate)
                    sl_price   = max(sl_price, new_sl)

                # SL Hit (conservative: SL zuerst prüfen)
                if c_lo <= sl_price:
                    exit_price = sl_price * (1 - SLIPPAGE_PCT)
                    outcome    = 'LOSS'
                    break
                # TP Hit (nur wenn Trailing noch nicht aktiv)
                if not trailing_active and c_hi >= tp_price:
                    exit_price = tp_price * (1 - SLIPPAGE_PCT)
                    outcome    = 'WIN'
                    break

            else:  # short
                if not trailing_active and c_lo <= activation_price:
                    trailing_active = True
                if trailing_active:
                    peak_price = min(peak_price, c_lo)
                    new_sl     = peak_price * (1 + callback_rate)
                    sl_price   = min(sl_price, new_sl)

                if c_hi >= sl_price:
                    exit_price = sl_price * (1 + SLIPPAGE_PCT)
                    outcome    = 'LOSS'
                    break
                if not trailing_active and c_lo <= tp_price:
                    exit_price = tp_price * (1 + SLIPPAGE_PCT)
                    outcome    = 'WIN'
                    break

        # TIMEOUT: zum Close der letzten Haltekerze schließen
        if exit_price is None:
            last_j = min(ohlcv_at + max_hold_bars, len(ohlcv_rows) - 1)
            exit_price = float(ohlcv_rows[last_j]['close'])
            exit_price *= (1 - SLIPPAGE_PCT) if side == 'long' else (1 + SLIPPAGE_PCT)

        # ── PnL berechnen ──────────────────────────────────────────────────────
        if side == 'long':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (1 - exit_price / entry_price) * 100

        fees_pct   = (FEE_PCT * 2) * 100
        net_pnl_pct = pnl_pct - fees_pct

        sl_dist_pct_actual = abs(entry_price - sl_price) / entry_price
        risk_usd    = equity * (risk_per_trade_pct / 100.0)
        notional    = min(risk_usd / (sl_dist_pct_actual + 1e-10),
                          equity * leverage)
        actual_pnl  = notional * (net_pnl_pct / 100.0)
        equity     += actual_pnl

        # ── Membership + Regime für Attribution ───────────────────────────────
        membership   = compute_membership_score(curr_feat, state_id, train_centroids, state_rows_train)
        hurst_idx    = FEATURE_COLS.index('hurst')
        hurst_val    = float(curr_feat[hurst_idx])
        regime       = "TREND" if hurst_val > 0.55 else "RANGE" if hurst_val < 0.45 else "NEUTRAL"
        quality_score = float(state_def.get('quality_score', 0.5)) if state_def else 0.5
        state_name   = state_def['name'] if state_def else f"STATE_{state_id}"

        prediction_snapshot = {
            'state_id':          state_id,
            'state_name':        state_name,
            'regime':            regime,
            'p_prior':           round(p_prior, 4),
            'p_knn':             round(knn_result['p_up'], 4),
            'p_bayes':           round(p_bayes, 4),
            'membership':        round(membership, 3),
            'quality_score':     round(quality_score, 3),
            'confidence':        round(confidence, 3),
            'stars':             stars,
            'k_used':            knn_result['k_used'],
            'k_requested':       knn_result.get('k_requested', k),
            'gap_detected':      knn_result.get('gap_detected', False),
            'htf_trend':         None,   # HTF-Daten nicht im Backtester verfügbar
            'structure_sl_applied': structure_sl_applied,
            'trailing_activated':   trailing_active,
        }

        trades.append({
            'bar_time':           bar_time,
            'side':               side,
            'state_id':           state_id,
            'state_name':         state_name,
            'regime':             regime,
            'outcome':            outcome,
            'entry_price':        round(entry_price, 6),
            'exit_price':         round(exit_price, 6),
            'sl_price':           round(sl_price, 6),
            'tp_price':           round(tp_price, 6),
            'trailing_activated': trailing_active,
            'structure_sl_applied': structure_sl_applied,
            'pnl_pct':            round(net_pnl_pct, 4),
            'pnl_usdt':           round(actual_pnl, 4),
            'equity_after':       round(equity, 4),
            'p_prior':            round(p_prior, 4),
            'p_knn':              round(knn_result['p_up'], 4),
            'p_bayes':            round(p_bayes, 4),
            'confidence':         round(confidence, 4),
            'stars':              stars,
            'membership':         round(membership, 3),
            'quality_score':      round(quality_score, 3),
            'prediction_snapshot': prediction_snapshot,
        })
        eq_curve.append(equity)

        # Train-Set inkrementell erweitern
        train_rows = rows[:test_idx + 1]

    stats = _compute_stats(trades, eq_curve, start_capital)
    logger.info(
        f"[PnL-Backtest] {stats.get('total_trades',0)} Trades | "
        f"WR: {stats.get('win_rate',0):.1%} | "
        f"PnL: {stats.get('total_pnl_usdt',0):+.2f} USDT | "
        f"DD: {stats.get('max_drawdown_pct',0):.1f}%"
    )
    return {"trades": trades, "stats": stats, "equity_curve": eq_curve}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _recompute_centroids_from_rows(train_rows, fallback_centroids, n_states):
    by_state: dict[int, list] = {}
    for r in train_rows:
        s = r.get('state_id')
        if s is not None:
            feat = [r.get(c, np.nan) for c in FEATURE_COLS]
            by_state.setdefault(s, []).append(feat)
    result = fallback_centroids.copy()
    for s_id, feats in by_state.items():
        if s_id < n_states:
            result[s_id] = np.nanmean(feats, axis=0)
    return result


def _compute_stats(trades, eq_curve, start_capital):
    if not trades:
        return {"total_trades": 0}
    wins    = [t for t in trades if t['outcome'] == 'WIN']
    losses  = [t for t in trades if t['outcome'] == 'LOSS']
    total   = len(trades)
    wr      = len(wins) / total
    total_pnl = sum(t['pnl_usdt'] for t in trades)
    avg_win   = sum(t['pnl_usdt'] for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(t['pnl_usdt'] for t in losses) / len(losses) if losses else 0.0
    sum_wins  = abs(sum(t['pnl_usdt'] for t in wins))
    sum_loss  = abs(sum(t['pnl_usdt'] for t in losses))
    pf        = sum_wins / sum_loss if sum_loss > 0 else float('inf')
    eq        = eq_curve or [start_capital]
    peak      = eq[0]
    max_dd    = 0.0
    for e in eq:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return {
        "total_trades":    total,
        "wins":            len(wins),
        "losses":          len(losses),
        "timeouts":        total - len(wins) - len(losses),
        "win_rate":        wr,
        "profit_factor":   pf,
        "total_pnl_usdt":  total_pnl,
        "total_pnl_pct":   total_pnl / start_capital * 100,
        "avg_win_usdt":    avg_win,
        "avg_loss_usdt":   avg_loss,
        "max_drawdown_pct": max_dd,
        "final_equity":    eq[-1] if eq else start_capital,
        "avg_stars":       sum(t.get('stars', 0) for t in trades) / total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Ausgabe
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results, market, tf, mode='wf'):
    stats = results.get("stats", {})
    label = "BACKTEST WF" if mode == 'wf' else "BACKTEST PnL"
    print(f"\n{'='*62}\n  {label}: {market} ({tf})\n{'='*62}")
    print(f"  Trades:         {stats.get('total_trades', 0)}")
    print(f"  Wins / Losses:  {stats.get('wins', 0)} / {stats.get('losses', 0)}  "
          f"(Timeout: {stats.get('timeouts', 0)})")
    print(f"  Win-Rate:       {stats.get('win_rate', 0):.1%}")
    print(f"  Profit Factor:  {stats.get('profit_factor', 0):.2f}")
    print(f"  Total PnL:      {stats.get('total_pnl_usdt', 0):+.2f} USDT  "
          f"({stats.get('total_pnl_pct', 0):+.1f}%)")
    print(f"  Max Drawdown:   {stats.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Final Equity:   {stats.get('final_equity', 0):.2f} USDT")
    print(f"  Avg Sterne:     {stats.get('avg_stars', 0):.1f}")
    print(f"{'='*62}\n")


def save_results(results, market, tf, mode='wf'):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = f"{market.replace('/','').replace(':','')}_{tf}"
    path = os.path.join(RESULTS_DIR, f"backtest_{mode}_{safe}.json")
    with open(path, 'w') as f:
        json.dump({
            "market": market, "timeframe": tf, "mode": mode,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "stats":  results.get("stats", {}),
            "trades": results.get("trades", []),
        }, f, indent=2, default=str)
    logger.info(f"Ergebnisse gespeichert: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description="statebot Backtester")
    parser.add_argument('--symbol',     default='BTC/USDT:USDT')
    parser.add_argument('--timeframe',  default='1d')
    parser.add_argument('--mode',       default='wf', choices=['wf', 'pnl'],
                        help='wf = Walk-Forward (close-to-close)  |  pnl = Candle-Simulation')
    parser.add_argument('--capital',    type=float, default=1000.0)
    parser.add_argument('--risk',       type=float, default=1.0)
    parser.add_argument('--k',          type=int,   default=20)
    parser.add_argument('--sl-pct',     type=float, default=1.5, dest='sl_pct')
    parser.add_argument('--rr',         type=float, default=2.0)
    parser.add_argument('--atr-mult',   type=float, default=2.0, dest='atr_mult')
    parser.add_argument('--no-struct',  action='store_true', dest='no_struct',
                        help='Structure Protection deaktivieren')
    parser.add_argument('--start-date', default='2022-01-01', dest='start_date')
    parser.add_argument('--end-date',   default='2025-01-01', dest='end_date')
    args = parser.parse_args()

    store = StateStore(DB_PATH)

    if args.mode == 'wf':
        results = run_walkforward_backtest(
            store, args.symbol, args.timeframe,
            k=args.k, sl_pct=args.sl_pct, rr_ratio=args.rr,
            start_capital=args.capital, risk_per_trade_pct=args.risk,
        )
        print_summary(results, args.symbol, args.timeframe, mode='wf')
        if results.get('stats', {}).get('total_trades', 0) > 0:
            save_results(results, args.symbol, args.timeframe, mode='wf')

    else:  # pnl
        ohlcv = load_ohlcv_data(args.symbol, args.timeframe, args.start_date, args.end_date)
        if ohlcv.empty:
            print("FEHLER: Keine OHLCV-Daten verfügbar. Prüfe secret.json und API.")
        else:
            results = run_pnl_backtest(
                store, ohlcv, args.symbol, args.timeframe,
                k=args.k, sl_pct=args.sl_pct, rr_ratio=args.rr,
                atr_mult_sl=args.atr_mult,
                use_structure_protection=not args.no_struct,
                start_capital=args.capital, risk_per_trade_pct=args.risk,
            )
            print_summary(results, args.symbol, args.timeframe, mode='pnl')
            if results.get('stats', {}).get('total_trades', 0) > 0:
                save_results(results, args.symbol, args.timeframe, mode='pnl')

    store.close()
