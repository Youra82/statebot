# utils/trade_manager.py — Trade-Management für statebot
# Adaptiert aus knnbot/utils/trade_manager.py
#
# Anpassungen:
#   - Signal kommt von get_state_signal() (nicht knn_signal)
#   - Telegram-Nachricht enthält State-Info + Eigenmodes + Multi-Target
#   - Chart zeigt P(bayes), State-Übergänge, Eigenmodes

import logging
import time
import json
import os
import sys
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TRACKER_DIR  = os.path.join(PROJECT_ROOT, 'artifacts', 'tracker')
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.utils.telegram    import send_message, send_photo
from statebot.utils.exchange    import Exchange
from statebot.engine.store      import StateStore
from statebot.engine.htf_filter import auto_htf
from statebot.strategy.signal   import get_state_signal, update_prediction_result

MIN_NOTIONAL_USDT = 5.0
MAX_NOTIONAL_USDT = 200_000.0
FETCH_LIMIT       = 600


# ─── Tracker ─────────────────────────────────────────────────────────────────

def get_tracker_file_path(symbol: str, timeframe: str) -> str:
    os.makedirs(TRACKER_DIR, exist_ok=True)
    safe = f"{symbol.replace('/', '-').replace(':', '-')}_{timeframe}.json"
    return os.path.join(TRACKER_DIR, safe)


def read_tracker(path: str) -> dict:
    default = {
        "status": "ok_to_trade", "last_side": None,
        "stop_loss_ids": [], "take_profit_ids": [],
        "active_prediction": None,
        "performance": {"total_trades": 0, "wins": 0, "losses": 0,
                        "consecutive_losses": 0, "consecutive_wins": 0},
    }
    if not os.path.exists(path):
        _write_tracker(path, default)
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        _write_tracker(path, default)
        return default


def _write_tracker(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def record_trade_result(path: str, outcome: str):
    tracker = read_tracker(path)
    perf    = tracker.setdefault('performance', {
        "total_trades": 0, "wins": 0, "losses": 0,
        "consecutive_losses": 0, "consecutive_wins": 0,
    })
    perf['total_trades'] = perf.get('total_trades', 0) + 1
    if outcome == 'win':
        perf['wins']               = perf.get('wins', 0) + 1
        perf['consecutive_wins']   = perf.get('consecutive_wins', 0) + 1
        perf['consecutive_losses'] = 0
    else:
        perf['losses']             = perf.get('losses', 0) + 1
        perf['consecutive_losses'] = perf.get('consecutive_losses', 0) + 1
        perf['consecutive_wins']   = 0
    total = perf['total_trades']
    if total > 0:
        perf['win_rate'] = perf['wins'] / total
    _write_tracker(path, tracker)


def should_skip_trading(path: str) -> tuple[bool, str]:
    tracker = read_tracker(path)
    perf    = tracker.get('performance', {})
    if perf.get('consecutive_losses', 0) >= 5:
        return True, f"{perf['consecutive_losses']} aufeinanderfolgende Verluste"
    total = perf.get('total_trades', 0)
    if total >= 30 and perf.get('win_rate', 1.0) < 0.25:
        return True, f"Win-Rate {perf.get('win_rate', 0):.1%} nach {total} Trades"
    return False, "OK"


# ─── Order Management ─────────────────────────────────────────────────────────

def cancel_entry_orders(exchange: Exchange, symbol: str, logger: logging.Logger,
                         tracker_path: str = None):
    protected_ids = set()
    if tracker_path:
        t = read_tracker(tracker_path)
        protected_ids.update(t.get('take_profit_ids', []))
        protected_ids.update(t.get('stop_loss_ids', []))
    count = 0
    for order in exchange.fetch_open_orders(symbol):
        if order['id'] in protected_ids:
            continue
        try:
            exchange.cancel_order(order['id'], symbol)
            count += 1
            time.sleep(0.1)
        except ccxt.OrderNotFound:
            pass
        except Exception as e:
            logger.warning(f"Konnte Order {order['id']} nicht stornieren: {e}")
    return count


def housekeeper_routine(exchange: Exchange, symbol: str, logger: logging.Logger) -> bool:
    try:
        exchange.cancel_all_orders_for_symbol(symbol)
        time.sleep(1)
        position = exchange.fetch_open_positions(symbol)
        if position:
            pos_info   = position[0]
            close_side = 'sell' if pos_info['side'] == 'long' else 'buy'
            exchange.place_market_order(symbol, close_side, float(pos_info['contracts']), reduce=True)
            time.sleep(3)
        return True
    except Exception as e:
        logger.error(f"Housekeeper-Fehler: {e}", exc_info=True)
        return False


def ensure_tp_sl(exchange: Exchange, position: dict, signal: dict | None,
                  params: dict, tracker_path: str, telegram_config: dict,
                  logger: logging.Logger):
    """Self-Repair: SL/TP neu setzen wenn verschwunden."""
    symbol    = params['market']['symbol']
    pos_side  = position['side']
    entry_price = float(position.get('entryPrice', 0))
    contracts   = float(position.get('contracts', 0))
    if contracts == 0:
        return

    triggers    = exchange.fetch_open_trigger_orders(symbol)
    trigger_ids = {o['id'] for o in triggers}
    tracker     = read_tracker(tracker_path)
    tp_ids      = set(tracker.get('take_profit_ids', []))
    sl_ids      = set(tracker.get('stop_loss_ids', []))

    if not tracker.get('active_prediction') and not tp_ids and not sl_ids:
        return

    tp_exists = bool(tp_ids & trigger_ids)
    sl_exists = bool(sl_ids & trigger_ids)
    if tp_exists and sl_exists:
        return

    logger.warning(f"Self-Repair: SL={sl_exists} TP={tp_exists} fuer {symbol}")
    active     = tracker.get('active_prediction') or {}
    tp_price   = active.get('tp_price') or (signal.get('tp_price') if signal else None)
    sl_price   = active.get('sl_price') or (signal.get('sl_price') if signal else None)
    trailing_callback = params['risk'].get('trailing_callback_rate_pct', 1.0) / 100.0
    tp_sl_side = 'sell' if pos_side == 'long' else 'buy'

    new_tp_ids = list(tp_ids)
    new_sl_ids = list(sl_ids)

    if not tp_exists and tp_price:
        try:
            o = exchange.place_trailing_stop_order(symbol, tp_sl_side, contracts, float(tp_price), trailing_callback)
            if o and 'id' in o:
                new_tp_ids = [o['id']]
        except Exception as e:
            logger.error(f"TP-Reparatur: {e}")

    if not sl_exists and sl_price:
        try:
            o = exchange.place_trigger_market_order(symbol, tp_sl_side, contracts, float(sl_price), reduce=True)
            if o and 'id' in o:
                new_sl_ids = [o['id']]
        except Exception as e:
            logger.error(f"SL-Reparatur: {e}")

    tracker['take_profit_ids'] = new_tp_ids
    tracker['stop_loss_ids']   = new_sl_ids
    _write_tracker(tracker_path, tracker)


# ─── State-Chart ─────────────────────────────────────────────────────────────

def _generate_state_chart(df: pd.DataFrame, signal: dict,
                            symbol: str, timeframe: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
    except ImportError:
        return None

    if df is None or df.empty or signal is None:
        return None

    n_candles = 40
    display_df = df[['open', 'high', 'low', 'close']].iloc[-n_candles:].reset_index(drop=True)
    n = len(display_df)
    if n == 0:
        return None

    opens  = display_df['open'].values
    highs  = display_df['high'].values
    lows   = display_df['low'].values
    closes = display_df['close'].values

    side        = signal.get('side', 'long')
    entry_price = signal.get('entry_price', closes[-1])
    sl_price    = signal.get('sl_price', 0)
    tp_price    = signal.get('tp_price', 0)
    p_bayes     = signal.get('p_bayes', 0.5)
    p_prior     = signal.get('p_prior', 0.5)
    p_knn       = signal.get('p_knn', 0.5)
    confidence  = signal.get('confidence', 0)
    stars       = signal.get('stars', 1)
    state_id    = signal.get('state_id', '?')
    state_name  = signal.get('state_name', '?')
    regime      = signal.get('regime', '?')
    k_used      = signal.get('k_used', 0)
    transitions = signal.get('top_transitions', [])
    eigenmodes  = signal.get('eigenmodes', [])
    exp_close   = signal.get('expected_close', closes[-1])
    exp_high    = signal.get('expected_high', closes[-1])
    exp_low     = signal.get('expected_low', closes[-1])

    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')
    gs  = gridspec.GridSpec(3, 3, width_ratios=[3, 1, 1], height_ratios=[3, 1, 1],
                             hspace=0.1, wspace=0.3)
    ax_c  = fig.add_subplot(gs[0, 0])
    ax_p  = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[0, 2])
    ax_f  = fig.add_subplot(gs[1, 0])
    ax_em = fig.add_subplot(gs[1:, 1:])

    for ax in [ax_c, ax_p, ax_tr, ax_f, ax_em]:
        ax.set_facecolor('#0d1117')

    # ── Kerzen ────────────────────────────────────────────────────────────────
    y_min = float(lows.min()) * 0.998
    y_max = float(highs.max()) * 1.002
    for p in [entry_price, sl_price, tp_price, exp_high, exp_low]:
        if p:
            y_min = min(y_min, float(p) * 0.999)
            y_max = max(y_max, float(p) * 1.001)

    ax_c.set_xlim(-1, n + 1)
    ax_c.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.05)
    bar_w = 0.6
    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color = '#26a69a' if c >= o else '#ef5350'
        ax_c.plot([i, i], [l, h], color=color, linewidth=0.8)
        ax_c.add_patch(mpatches.FancyBboxPatch(
            (i - bar_w/2, min(o, c)), bar_w, max(abs(c-o), (h-l)*0.005),
            boxstyle="square,pad=0", linewidth=0, facecolor=color,
        ))

    def _hline(price, label, color, lw=1.2):
        if not price or not (y_min < float(price) < y_max):
            return
        ax_c.axhline(price, color=color, linewidth=lw, linestyle='--')
        ax_c.text(n - 0.5, price, f'  {label}: {float(price):.5g}',
                  color='#0d1117', fontsize=7.5, va='center', ha='right',
                  bbox=dict(facecolor=color, edgecolor='none', alpha=0.9, boxstyle='square,pad=0.2'))

    _hline(tp_price,    'TP',    '#00c853')
    _hline(exp_high,    'eHigh', '#4caf50', lw=0.7)
    _hline(entry_price, 'Entry', '#ffd700')
    _hline(exp_close,   'eClose','#90caf9', lw=0.7)
    _hline(exp_low,     'eLow',  '#ff7043', lw=0.7)
    _hline(sl_price,    'SL',    '#ff1744')

    if sl_price and tp_price:
        ax_c.axhspan(min(sl_price, entry_price), max(sl_price, entry_price), color='#ff1744', alpha=0.05)
        ax_c.axhspan(min(tp_price, entry_price), max(tp_price, entry_price), color='#00c853', alpha=0.05)

    star_str = '★' * stars + '☆' * (5 - stars)
    ax_c.set_title(
        f"STATEBOT  {symbol} {timeframe}  |  "
        f"{'LONG' if side == 'long' else 'SHORT'}  |  "
        f"State {state_id} ({state_name})  |  {star_str}",
        color='#e0e0e0', fontsize=9, pad=6,
    )
    ax_c.tick_params(colors='#888888', labelsize=7)
    for sp in ax_c.spines.values():
        sp.set_edgecolor('#2a3a4a')
    ax_c.set_xticks([])
    ax_c.yaxis.tick_right()
    ax_c.grid(axis='y', color='#1e2a3a', linewidth=0.4)

    # ── P(up) Balken ──────────────────────────────────────────────────────────
    labels_p = ['Prior', 'KNN', 'Bayes']
    vals_p   = [p_prior, p_knn, p_bayes]
    colors_p = ['#64b5f6', '#81c784', '#ffd54f']
    y_pos    = [2, 1, 0]
    ax_p.barh(y_pos, vals_p, color=colors_p, height=0.45)
    ax_p.axvline(0.5, color='#666666', linewidth=0.8, linestyle='--')
    ax_p.set_xlim(0, 1)
    ax_p.set_yticks([0, 1, 2])
    ax_p.set_yticklabels(labels_p, color='#cccccc', fontsize=8)
    for i, v in enumerate(vals_p):
        ax_p.text(v + 0.02, y_pos[i], f'{v:.1%}', color='#e0e0e0', fontsize=7.5, va='center')
    ax_p.set_title(f'P(up)  |  Conf={confidence:.2f}', color='#cccccc', fontsize=8, pad=4)
    ax_p.tick_params(colors='#888888', labelsize=7)
    for sp in ax_p.spines.values():
        sp.set_edgecolor('#2a3a4a')

    # ── State-Übergänge ──────────────────────────────────────────────────────
    if transitions:
        tr_names = [t['name'][:12] for t in transitions]
        tr_probs = [t['probability'] for t in transitions]
        tr_y     = list(range(len(tr_names)))
        ax_tr.barh(tr_y, tr_probs, color='#ce93d8', height=0.45)
        ax_tr.set_yticks(tr_y)
        ax_tr.set_yticklabels(tr_names, color='#cccccc', fontsize=7.5)
        ax_tr.set_xlim(0, 1)
        for i, v in enumerate(tr_probs):
            ax_tr.text(v + 0.02, tr_y[i], f'{v:.0%}', color='#e0e0e0', fontsize=7)
    ax_tr.set_title(f'Regime: {regime}\nk={k_used} Nachbarn', color='#cccccc', fontsize=8, pad=4)
    ax_tr.tick_params(colors='#888888', labelsize=7)
    for sp in ax_tr.spines.values():
        sp.set_edgecolor('#2a3a4a')

    # ── Neighbor-Returns Histogram ────────────────────────────────────────────
    neighbor_ret = signal.get('neighbor_returns') or []
    if neighbor_ret:
        n_bins = min(10, len(neighbor_ret))
        colors_hist = ['#26a69a' if r > 0 else '#ef5350' for r in neighbor_ret]
        sorted_ret = sorted(neighbor_ret)
        xs = list(range(len(sorted_ret)))
        ax_f.bar(xs, sorted_ret,
                 color=['#26a69a' if r > 0 else '#ef5350' for r in sorted_ret],
                 width=0.8)
        ax_f.axhline(0, color='#666666', linewidth=0.6)
        ax_f.set_title('Neighbor-Returns (%)', color='#cccccc', fontsize=7.5)
    else:
        ax_f.text(0.5, 0.5, 'Keine Daten', color='#888888', ha='center', va='center')
    ax_f.tick_params(colors='#888888', labelsize=6)
    for sp in ax_f.spines.values():
        sp.set_edgecolor('#2a3a4a')

    # ── Eigenmodes ────────────────────────────────────────────────────────────
    if eigenmodes:
        em_labels = [f"M{m['mode']} {m['dominant_feature'][:8]}" for m in eigenmodes]
        em_vals   = [m['explained_pct'] for m in eigenmodes]
        em_colors = ['#4fc3f7', '#81c784', '#ffb74d', '#f06292'][:len(em_labels)]
        ax_em.barh(em_labels, em_vals, color=em_colors[:len(em_labels)], height=0.45)
        ax_em.set_xlabel('Erklärt %', color='#888888', fontsize=7)
        ax_em.set_title('Eigenmodes (PCA)', color='#cccccc', fontsize=8, pad=4)
        for i, v in enumerate(em_vals):
            ax_em.text(v + 0.5, i, f'{v:.1f}%', color='#e0e0e0', fontsize=7, va='center')
    else:
        ax_em.text(0.5, 0.5, 'Eigenmodes\nN/A', color='#888888', ha='center', va='center')
    ax_em.tick_params(colors='#888888', labelsize=7)
    for sp in ax_em.spines.values():
        sp.set_edgecolor('#2a3a4a')

    tmp_dir  = os.path.join(PROJECT_ROOT, 'artifacts', 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    sym_safe = symbol.replace('/', '-').replace(':', '-')
    path     = os.path.join(tmp_dir, f'state_entry_{sym_safe}_{timeframe}_{ts}.png')
    fig.savefig(path, dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _send_state_chart(df, signal, symbol, timeframe, telegram_config, logger):
    if not telegram_config or not telegram_config.get('bot_token'):
        return
    try:
        path = _generate_state_chart(df, signal, symbol, timeframe)
        if path and os.path.exists(path):
            stars_str = '★' * signal.get('stars', 1) + '☆' * (5 - signal.get('stars', 1))
            caption = (
                f"statebot | {symbol} ({timeframe})\n"
                f"{'LONG' if signal.get('side')=='long' else 'SHORT'}  |  "
                f"State {signal.get('state_id')} ({signal.get('state_name')})\n"
                f"P(bayes): {signal.get('p_bayes', 0):.1%}  |  "
                f"Konfidenz: {signal.get('confidence', 0):.2f}  |  {stars_str}"
            )
            send_photo(telegram_config.get('bot_token'), telegram_config.get('chat_id'), path, caption)
            os.remove(path)
    except Exception as e:
        logger.warning(f"State-Chart senden fehlgeschlagen: {e}")


# ─── Entry Orders ─────────────────────────────────────────────────────────────

def place_entry_orders(exchange, signal, params, balance, tracker_path,
                        telegram_config, logger, df=None):
    symbol = params['market']['symbol']
    side   = signal.get('side')
    if not side:
        return

    if side == 'long'  and not params.get('behavior', {}).get('use_longs', True):
        return
    if side == 'short' and not params.get('behavior', {}).get('use_shorts', True):
        return

    skip, reason = should_skip_trading(tracker_path)
    if skip:
        logger.warning(f"Trading pausiert: {reason}")
        return

    risk     = params['risk']
    leverage = risk['leverage']
    risk_pct = risk.get('risk_per_entry_pct', 1.0)
    trailing_callback = risk.get('trailing_callback_rate_pct', 1.0) / 100.0

    entry_price = signal['entry_price']
    sl_price    = signal['sl_price']
    tp_price    = signal['tp_price']
    sl_pct      = signal['sl_pct']

    if sl_pct <= 0:
        return

    sl_dist       = abs(entry_price - sl_price)
    risk_amount   = balance * (risk_pct / 100.0)
    amount_coins  = risk_amount / sl_dist
    notional_unc  = amount_coins * entry_price
    if notional_unc > MAX_NOTIONAL_USDT:
        amount_coins = MAX_NOTIONAL_USDT / entry_price

    min_amount = exchange.fetch_min_amount_tradable(symbol)
    if amount_coins < min_amount:
        logger.warning(f"Menge {amount_coins:.6f} unter Minimum. Ueberspringe.")
        return
    if amount_coins * entry_price < MIN_NOTIONAL_USDT:
        logger.warning("Notional unter Minimum. Ueberspringe.")
        return

    try:
        exchange.set_margin_mode(symbol, risk.get('margin_mode', 'isolated'))
        time.sleep(0.3)
        exchange.set_leverage(symbol, leverage, risk.get('margin_mode', 'isolated'))
        time.sleep(0.3)
    except Exception as e:
        logger.warning(f"Margin/Leverage: {e}")

    order_side = 'buy' if side == 'long' else 'sell'
    tp_sl_side = 'sell' if side == 'long' else 'buy'

    logger.info(
        f"[Entry] {side.upper()} {amount_coins:.6f} {symbol} | "
        f"~{entry_price:.4f} | SL={sl_price:.4f} | TP={tp_price:.4f} | "
        f"P(bayes)={signal['p_bayes']:.1%} | {signal['stars']}★"
    )

    new_tp_ids = []
    new_sl_ids = []

    try:
        exchange.place_market_order(symbol, order_side, amount_coins, reduce=False,
                                     margin_mode=risk.get('margin_mode', 'isolated'))
    except ccxt.InsufficientFunds as e:
        logger.error(f"Nicht genug Guthaben: {e}")
        return
    except Exception as e:
        logger.error(f"Entry-Fehler: {e}", exc_info=True)
        return

    time.sleep(2)
    open_positions = exchange.fetch_open_positions(symbol)
    if not open_positions:
        logger.error("Entry gesendet aber keine offene Position.")
        return

    pos_info         = open_positions[0]
    actual_contracts = float(pos_info['contracts'])
    actual_entry     = float(pos_info.get('entryPrice') or entry_price)

    try:
        sl_order = exchange.place_trigger_market_order(symbol, tp_sl_side, actual_contracts, sl_price, reduce=True)
        if sl_order and 'id' in sl_order:
            new_sl_ids.append(sl_order['id'])
        time.sleep(0.2)
        tp_order = exchange.place_trailing_stop_order(symbol, tp_sl_side, actual_contracts, tp_price, trailing_callback)
        if tp_order and 'id' in tp_order:
            new_tp_ids.append(tp_order['id'])
    except Exception as e:
        logger.error(f"SL/TP-Placement: {e}", exc_info=True)
        for oid in new_tp_ids + new_sl_ids:
            try:
                exchange.cancel_trigger_order(oid, symbol)
            except Exception:
                pass
        housekeeper_routine(exchange, symbol, logger)
        return

    tracker = read_tracker(tracker_path)
    tracker.update({
        'stop_loss_ids':  new_sl_ids,
        'take_profit_ids': new_tp_ids,
        'last_side':       side,
        'status':          'ok_to_trade',
        'last_notified_entry_price': actual_entry,
        'last_notified_side':        side,
        'active_prediction': {
            'prediction_id':    signal['prediction_id'],
            'direction':        side.upper(),
            'entry_price':      actual_entry,
            'sl_price':         sl_price,
            'tp_price':         tp_price,
            'state_id':         signal['state_id'],
            'state_name':       signal['state_name'],
            'p_bayes':          signal['p_bayes'],
            'confidence':       signal['confidence'],
            'stars':            signal['stars'],
            'expected_close':   signal.get('expected_close'),
            'feature_vector':   signal.get('feature_vector', {}),
        }
    })
    _write_tracker(tracker_path, tracker)

    # Telegram
    try:
        timeframe = params['market']['timeframe']
        star_str  = '★' * signal['stars'] + '☆' * (5 - signal['stars'])
        sl_d_pct  = abs(actual_entry - sl_price) / actual_entry * 100
        tp_d_pct  = abs(tp_price - actual_entry) / actual_entry * 100
        rr        = tp_d_pct / sl_d_pct if sl_d_pct > 0 else 0
        msg = (
            f"statebot SIGNAL: {symbol} ({timeframe})\n"
            f"{'─' * 32}\n"
            f"Richtung:      {'LONG' if side == 'long' else 'SHORT'}\n"
            f"Entry:         ${actual_entry:.6g}\n"
            f"SL:            ${sl_price:.6g} (-{sl_d_pct:.2f}%)\n"
            f"Trailing (ab): ${tp_price:.6g} (+{tp_d_pct:.2f}%)\n"
            f"Min R:R:       1:{rr:.1f}\n"
            f"Hebel:         {leverage}x\n"
            f"{'─' * 32}\n"
            f"State:         {signal['state_id']} ({signal['state_name']})\n"
            f"Regime:        {signal['regime']}\n"
            f"P(prior):      {signal['p_prior']:.1%}\n"
            f"P(knn):        {signal['p_knn']:.1%}\n"
            f"P(bayes):      {signal['p_bayes']:.1%}\n"
            f"Konfidenz:     {signal['confidence']:.2f}\n"
            f"Qualität:      {star_str}\n"
            f"{'─' * 32}\n"
            f"E[Close]:      {signal.get('expected_close', 0):.6g} ({signal.get('expected_close_pct', 0):+.2f}%)\n"
            f"E[High]:       {signal.get('expected_high', 0):.6g}\n"
            f"E[Low]:        {signal.get('expected_low', 0):.6g}"
        )
        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), msg)
    except Exception as e:
        logger.warning(f"Telegram: {e}")

    _send_state_chart(df, signal, symbol, params['market']['timeframe'], telegram_config, logger)


# ─── Self-Learning ────────────────────────────────────────────────────────────

def self_learn(tracker_path, store, market, timeframe, exit_price, logger):
    tracker = read_tracker(tracker_path)
    active  = tracker.get('active_prediction')
    if not active:
        return
    entry_price    = active.get('entry_price', 0)
    direction      = active.get('direction', 'LONG')
    prediction_id  = active.get('prediction_id', '')
    if entry_price > 0 and exit_price > 0:
        if direction == 'LONG':
            actual_return = (exit_price - entry_price) / entry_price * 100
        else:
            actual_return = (entry_price - exit_price) / entry_price * 100
    else:
        actual_return = 0.0
    if prediction_id:
        update_prediction_result(store, market, timeframe, prediction_id, actual_return)
        # Kalibrierungs-Outcome für alle Signale schreiben
        went_up = actual_return > 0
        calibrator = getattr(store, 'calibrator', None)
        if calibrator:
            calibrator.record_outcome(market, timeframe, prediction_id, went_up)
    tracker['active_prediction'] = None
    _write_tracker(tracker_path, tracker)


# ─── Haupt-Zyklus ─────────────────────────────────────────────────────────────

def full_trade_cycle(exchange: Exchange, params: dict,
                     telegram_config: dict, db_path: str,
                     logger: logging.Logger):
    symbol    = params['market']['symbol']
    timeframe = params['market']['timeframe']
    tracker_path = get_tracker_file_path(symbol, timeframe)

    tracker = read_tracker(tracker_path)
    tracker.update({'market': symbol, 'timeframe': timeframe})
    _write_tracker(tracker_path, tracker)

    logger.info(f"Lade {FETCH_LIMIT} Kerzen fuer {symbol} ({timeframe})...")
    df = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=FETCH_LIMIT)
    if df is None or len(df) < 220:
        logger.error(f"Zu wenig Daten. Abbruch.")
        return

    # ── HTF-Daten für Supertrend-Veto laden ───────────────────────────────────
    htf_df  = None
    htf_cfg = params.get('htf_filter', {})
    if htf_cfg.get('enabled', False):
        htf_tf = htf_cfg.get('timeframe') or auto_htf(timeframe)
        logger.info(f"Lade HTF-Daten {symbol} ({htf_tf}) für Supertrend-Filter...")
        try:
            htf_df = exchange.fetch_recent_ohlcv(symbol, htf_tf, limit=200)
            if htf_df is None or len(htf_df) < htf_cfg.get('period', 10) + 2:
                logger.warning("HTF-Daten unzureichend — Veto deaktiviert für diesen Lauf.")
                htf_df = None
        except Exception as e:
            logger.warning(f"HTF-Daten konnten nicht geladen werden: {e}")
            htf_df = None

    store  = StateStore(db_path)
    signal = get_state_signal(df, params, store, htf_df=htf_df)

    if signal:
        logger.info(f"Signal: {signal['side'].upper()} | P(bayes)={signal['p_bayes']:.1%} | {signal['stars']}★")
    else:
        logger.info("Kein Signal.")

    current_price = float(df['close'].iloc[-1])
    cancel_entry_orders(exchange, symbol, logger, tracker_path)
    open_positions = exchange.fetch_open_positions(symbol)

    if open_positions:
        position = open_positions[0]
        ensure_tp_sl(exchange, position, signal, params, tracker_path, telegram_config, logger)

        # Overshoot-Check
        try:
            ov_tracker = read_tracker(tracker_path)
            ov_active  = ov_tracker.get('active_prediction') or {}
            sl_price_ov = ov_active.get('sl_price')
            tp_price_ov = ov_active.get('tp_price')
            pos_side_ov  = position.get('side', 'long')
            contracts_ov = float(position.get('contracts', 0))
            close_side_ov = 'sell' if pos_side_ov == 'long' else 'buy'

            if sl_price_ov and tp_price_ov and contracts_ov > 0 and current_price > 0:
                sl_val = float(sl_price_ov)
                tp_val = float(tp_price_ov)
                if pos_side_ov == 'long':
                    breached = current_price <= sl_val or current_price >= tp_val
                    reason   = "SL" if current_price <= sl_val else "TP"
                else:
                    breached = current_price >= sl_val or current_price <= tp_val
                    reason   = "SL" if current_price >= sl_val else "TP"
                if breached:
                    exchange.cancel_all_orders_for_symbol(symbol)
                    exchange.place_market_order(symbol, close_side_ov, contracts_ov, reduce=True)
                    time.sleep(2)
                    if not exchange.fetch_open_positions(symbol):
                        _write_tracker(tracker_path, {})
                    send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'),
                                 f"statebot NOTSCHLIESSUNG ({symbol}): Preis {current_price:.6f} {reason}")
        except Exception as e:
            logger.error(f"Overshoot-Check: {e}")

    else:
        housekeeper_routine(exchange, symbol, logger)
        tracker    = read_tracker(tracker_path)
        had_orders = bool(tracker.get('take_profit_ids') or tracker.get('stop_loss_ids'))

        if had_orders:
            active     = tracker.get('active_prediction') or {}
            entry_p    = active.get('entry_price', 0)
            last_side  = tracker.get('last_side', 'long')
            sl_p       = active.get('sl_price', 0)
            outcome    = None

            if entry_p > 0 and sl_p > 0:
                outcome = 'loss' if (
                    (last_side == 'long'  and current_price <= sl_p * 1.005) or
                    (last_side == 'short' and current_price >= sl_p * 0.995)
                ) else 'win'

            if outcome:
                record_trade_result(tracker_path, outcome)
                try:
                    self_learn(tracker_path, store, symbol, timeframe, current_price, logger)
                except Exception as e:
                    logger.error(f"Self-Learning: {e}")
                send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'),
                             f"statebot {'WIN' if outcome=='win' else 'LOSS'}: {symbol} ({timeframe})")

            tracker.update({"stop_loss_ids": [], "take_profit_ids": [], "status": "ok_to_trade"})
            _write_tracker(tracker_path, tracker)

        balance = exchange.fetch_balance_usdt()
        logger.info(f"Guthaben: {balance:.2f} USDT")

        if balance < MIN_NOTIONAL_USDT or signal is None:
            store.close()
            return

        place_entry_orders(exchange, signal, params, balance, tracker_path,
                           telegram_config, logger, df=df)

    store.close()
    logger.info(f"Zyklus abgeschlossen: {symbol} ({timeframe}).")
