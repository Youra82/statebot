# strategy/run.py — Einzelstrategie Entry Point
import os, sys, json, logging, argparse, ccxt
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.utils.exchange      import Exchange
from statebot.utils.trade_manager import full_trade_cycle
from statebot.utils.guardian      import guardian_decorator

DB_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'states.db')


def setup_logging(symbol: str, tf: str) -> logging.Logger:
    safe = f"{symbol.replace('/', '').replace(':', '')}_{tf}"
    log_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    lg = logging.getLogger(f'statebot_{safe}')
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fh = RotatingFileHandler(os.path.join(log_dir, f'statebot_{safe}.log'),
                                  maxBytes=5*1024*1024, backupCount=3)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(f'%(asctime)s [{safe}] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
        lg.addHandler(fh)
        lg.addHandler(ch)
        lg.propagate = False
    return lg


def load_config(symbol: str, tf: str, settings: dict) -> dict:
    gr   = settings.get('risk_settings', {})
    gk   = settings.get('knn_settings', {})
    ovr  = {}
    for s in settings.get('live_trading_settings', {}).get('active_strategies', []):
        if s.get('symbol') == symbol and s.get('timeframe') == tf:
            ovr = s
            break
    ro = ovr.get('risk_overrides', {})
    ko = ovr.get('knn_overrides', {})
    gh   = settings.get('htf_filter', {})
    return {
        "market": {"symbol": symbol, "timeframe": tf},
        "risk": {
            "risk_per_entry_pct":         ro.get('risk_per_entry_pct',         gr.get('risk_per_entry_pct', 1.0)),
            "leverage":                   ro.get('leverage',                   gr.get('leverage', 5)),
            "margin_mode":                ro.get('margin_mode',                gr.get('margin_mode', 'isolated')),
            "rr_ratio":                   ro.get('rr_ratio',                   gr.get('rr_ratio', 2.0)),
            "sl_pct":                     ro.get('sl_pct',                     gr.get('sl_pct', 1.5)),
            "trailing_callback_rate_pct": ro.get('trailing_callback_rate_pct', gr.get('trailing_callback_rate_pct', 1.0)),
            "use_expected_targets":       ro.get('use_expected_targets',       gr.get('use_expected_targets', False)),
            "use_structure_protection":   ro.get('use_structure_protection',   gr.get('use_structure_protection', True)),
        },
        "htf_filter": {
            "enabled":    gh.get('enabled',    False),
            "timeframe":  gh.get('timeframe',  None),
            "factor":     gh.get('factor',     3.0),
            "period":     gh.get('period',     10),
        },
        "knn": {
            "k":               ko.get('k',               gk.get('k', 20)),
            "threshold_long":  ko.get('threshold_long',  gk.get('threshold_long',  0.62)),
            "threshold_short": ko.get('threshold_short', gk.get('threshold_short', 0.38)),
            "min_confidence":  ko.get('min_confidence',  gk.get('min_confidence',  0.45)),
            "min_stars":       ko.get('min_stars',        gk.get('min_stars', 2)),
            "min_vectors":     ko.get('min_vectors',      gk.get('min_vectors', 40)),
            "feature_weights": ko.get('feature_weights',  gk.get('feature_weights', None)),
        },
        "behavior": {
            "use_longs":  ro.get('use_longs',  True),
            "use_shorts": ro.get('use_shorts', True),
        },
    }


@guardian_decorator
def run_for_account(account, telegram_config, params, db_path, logger):
    logger.info(f"--- statebot: {params['market']['symbol']} ({params['market']['timeframe']}) ---")
    exchange = Exchange(account)
    full_trade_cycle(exchange, params, telegram_config, db_path, logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',    required=True)
    parser.add_argument('--timeframe', required=True)
    args = parser.parse_args()
    logger = setup_logging(args.symbol, args.timeframe)
    try:
        with open(os.path.join(PROJECT_ROOT, 'settings.json')) as f:
            settings = json.load(f)
        with open(os.path.join(PROJECT_ROOT, 'secret.json')) as f:
            secrets = json.load(f)
        params          = load_config(args.symbol, args.timeframe, settings)
        accounts        = secrets.get('statebot', [])
        telegram_config = secrets.get('telegram', {})
        if not accounts:
            logger.critical("Kein 'statebot'-Account in secret.json")
            sys.exit(1)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.critical(f"Konfigurationsfehler: {e}")
        sys.exit(1)
    for account in accounts:
        run_for_account(account, telegram_config, params, DB_PATH, logger)
    logger.info("Lauf abgeschlossen.")

if __name__ == "__main__":
    main()
