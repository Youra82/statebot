#!/usr/bin/env python3
# master_runner.py — Alle aktiven Strategien aus settings.json starten
import os, sys, json, time, logging, subprocess
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s [master_runner] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT  = os.path.abspath(os.path.dirname(__file__))
RUNNER_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'statebot', 'strategy', 'run.py')
SETTINGS_PATH = os.path.join(PROJECT_ROOT, 'settings.json')


def main():
    logger.info(f"statebot master_runner  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    try:
        with open(SETTINGS_PATH) as f:
            settings = json.load(f)
    except Exception as e:
        logger.critical(f"settings.json: {e}")
        sys.exit(1)

    strategies = [s for s in settings.get('live_trading_settings', {}).get('active_strategies', [])
                  if s.get('enabled', True)]
    if not strategies:
        logger.warning("Keine aktiven Strategien.")
        return

    logger.info(f"{len(strategies)} Strategie(n):")
    for s in strategies:
        logger.info(f"  {s.get('symbol')} ({s.get('timeframe')})")

    for i, s in enumerate(strategies):
        sym = s.get('symbol')
        tf  = s.get('timeframe')
        if not sym or not tf:
            continue
        logger.info(f"Starte: {sym} ({tf})")
        try:
            subprocess.run([sys.executable, RUNNER_SCRIPT, '--symbol', sym, '--timeframe', tf],
                           timeout=300)
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout: {sym}")
        except Exception as e:
            logger.error(f"Fehler {sym}: {e}")
        if i < len(strategies) - 1:
            time.sleep(2)

    logger.info("master_runner: Fertig.")

if __name__ == "__main__":
    main()
