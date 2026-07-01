#!/usr/bin/env python3
import os, sys, logging, argparse
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.engine.store        import StateStore
from statebot.analysis.backtester import run_walkforward_backtest, print_summary, save_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
DB_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'states.db')

SL_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
RR_SWEEP = [1.5, 2.0, 2.5, 3.0]


def _run_sweep(store, market, tf, args, allowed_states=None, save_mode='wf'):
    combos = [(sl, rr) for sl in SL_SWEEP for rr in RR_SWEEP]
    print(f"\n  Sweep: {len(combos)} Kombinationen für {market} ({tf}) ...")

    rows = []
    for sl, rr in combos:
        r = run_walkforward_backtest(
            store, market, tf,
            k=args.k, sl_pct=sl, rr_ratio=rr,
            start_capital=args.capital, risk_per_trade_pct=args.risk,
            allowed_states=allowed_states,
        )
        stats = r.get('stats', {})
        n = stats.get('total_trades', 0)
        if n < args.min_trades:
            continue
        rows.append({
            'sl':   sl,
            'rr':   rr,
            'n':    n,
            'wr':   round(stats.get('win_rate', 0) * 100, 1),
            'pf':   round(stats.get('profit_factor', 0), 2),
            'pnl':  round(stats.get('total_pnl_pct', 0), 1),
            'dd':   round(stats.get('max_drawdown_pct', 0), 1),
            '_r':   r,
        })

    rows.sort(key=lambda x: x['pf'], reverse=True)
    top = rows[:args.top_n]

    if not top:
        print(f"  Keine Kombinationen mit >= {args.min_trades} Trades gefunden.")
        return

    sep = '  ' + '─' * 58
    print(f"\n  Beste SL/RR-Kombinationen — {market} ({tf}):")
    print(f"  {'#':<3} {'SL%':<6} {'RR':<5} {'Trades':<8} {'WR%':<7} {'PF':<6} {'PnL%':<8} {'DD%'}")
    print(sep)
    for i, row in enumerate(top, 1):
        pnl_s  = f"{row['pnl']:+.1f}%"
        marker = '  ← Empfehlung' if i == 1 else ''
        print(f"  {i:<3} {row['sl']:<6.1f} {row['rr']:<5.1f} {row['n']:<8} "
              f"{row['wr']:<7.1f} {row['pf']:<6.2f} {pnl_s:<8} {row['dd']:.1f}%{marker}")
    print(sep)

    best = top[0]
    print(f"\n  → Empfehlung: SL={best['sl']}%, RR={best['rr']} "
          f"(PF={best['pf']}, WR={best['wr']}%, PnL={best['pnl']:+.1f}%)")

    if best['_r'].get('stats', {}).get('total_trades', 0) > 0:
        save_results(best['_r'], market, tf, mode=save_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',     type=str,   default=None)
    parser.add_argument('--timeframe',  type=str,   default=None)
    parser.add_argument('--capital',    type=float, default=1000.0)
    parser.add_argument('--risk',       type=float, default=1.0)
    parser.add_argument('--k',          type=int,   default=20)
    parser.add_argument('--sl-pct',     type=float, default=1.5,  dest='sl_pct')
    parser.add_argument('--rr',         type=float, default=2.0)
    parser.add_argument('--sweep',      action='store_true', default=False,
                        help='SL/RR-Sweep: alle Kombinationen testen, beste anzeigen')
    parser.add_argument('--min-trades', type=int,   default=10,   dest='min_trades',
                        help='Mindest-Trades im Sweep (default: 10)')
    parser.add_argument('--top-n',      type=int,   default=5,    dest='top_n',
                        help='Anzahl beste Kombinationen im Sweep (default: 5)')
    parser.add_argument('--states',     type=str,   default=None,
                        help='Nur diese State-IDs handeln, kommasepariert (z.B. 8,15)')
    args = parser.parse_args()

    allowed_states = None
    if args.states:
        try:
            allowed_states = [int(x.strip()) for x in args.states.split(',') if x.strip()]
            print(f"  Filter: nur States {allowed_states}")
        except ValueError:
            print(f"  WARNUNG: --states '{args.states}' konnte nicht geparst werden — ignoriert")

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

    save_mode = 'wf_filtered' if allowed_states else 'wf'

    for market, tf in pairs:
        if args.sweep:
            _run_sweep(store, market, tf, args, allowed_states=allowed_states, save_mode=save_mode)
        else:
            results = run_walkforward_backtest(
                store, market, tf,
                k=args.k, sl_pct=args.sl_pct, rr_ratio=args.rr,
                start_capital=args.capital, risk_per_trade_pct=args.risk,
                allowed_states=allowed_states,
            )
            print_summary(results, market, tf)
            if results.get('stats', {}).get('total_trades', 0) > 0:
                save_results(results, market, tf, mode=save_mode)

    store.close()


if __name__ == "__main__":
    main()
