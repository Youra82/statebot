#!/usr/bin/env python3
import os, sys, logging, argparse
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.engine.store       import StateStore
from statebot.analysis.backtester import run_walkforward_backtest, print_summary, save_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
DB_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'states.db')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',    type=str,   default=None)
    parser.add_argument('--timeframe', type=str,   default=None)
    parser.add_argument('--capital',   type=float, default=1000.0)
    parser.add_argument('--risk',      type=float, default=1.0)
    parser.add_argument('--k',         type=int,   default=20)
    parser.add_argument('--sl-pct',    type=float, default=1.5, dest='sl_pct')
    parser.add_argument('--rr',        type=float, default=2.0)
    args = parser.parse_args()

    store = StateStore(DB_PATH)
    pairs = store.get_all_market_pairs()
    if not pairs:
        print("Keine Daten. Zuerst build_states.py ausführen.")
        store.close()
        return

    if args.symbol and args.timeframe:
        pairs = [(m, tf) for m, tf in pairs if m == args.symbol and tf == args.timeframe]
    elif args.symbol:
        pairs = [(m, tf) for m, tf in pairs if m == args.symbol]

    for market, tf in pairs:
        results = run_walkforward_backtest(
            store, market, tf,
            k=args.k, sl_pct=args.sl_pct, rr_ratio=args.rr,
            start_capital=args.capital, risk_per_trade_pct=args.risk,
        )
        print_summary(results, market, tf)
        if results.get('stats', {}).get('total_trades', 0) > 0:
            save_results(results, market, tf)
    store.close()

if __name__ == "__main__":
    main()
