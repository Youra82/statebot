# analysis/show_results.py — Ergebnisse und State-Übersicht anzeigen

import os, sys, json, argparse
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.engine.store import StateStore

DB_PATH     = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'states.db')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'results')


def show_store_summary(store):
    summary = store.get_summary()
    if not summary:
        print("  Keine Daten im StateStore.")
        return
    print(f"\n{'='*70}\n  STATE STORE ÜBERSICHT\n{'='*70}")
    print(f"  {'Symbol':<25} {'TF':<6} {'Vektoren':>9} {'Clustered':>9} {'States':>7} {'Zuletzt':<12}")
    print(f"  {'-'*65}")
    for r in summary:
        print(f"  {r.get('market','?'):<25} {r.get('timeframe','?'):<6} "
              f"{r.get('total',0):>9} {r.get('clustered',0):>9} "
              f"{r.get('n_clusters','-'):>7} {(r.get('latest_bar') or '')[:10]:<12}")
    print(f"{'='*70}\n")


def show_state_definitions(store, market, tf):
    defs = store.get_state_definitions(market, tf)
    if not defs:
        print(f"  Keine States für {market}/{tf}")
        return
    print(f"\n  STATES: {market} ({tf}) — {len(defs)} Cluster")
    print(f"  {'ID':>4} {'Name':<18} {'N':>6} {'Avg Ret':>9} {'Up%':>7} {'Std':>7} {'Qual':>6}")
    print(f"  {'-'*63}")
    for d in sorted(defs, key=lambda x: x['state_id']):
        qs   = d.get('quality_score', None)
        grade = ("HIGH" if qs and qs >= 0.60 else
                 "MED"  if qs and qs >= 0.35 else
                 "LOW"  if qs and qs >= 0.20 else
                 "POOR" if qs is not None else "n/a")
        qs_str = f"{qs:.2f}" if qs is not None else " n/a"
        flag   = " !" if grade == "POOR" else ""
        print(f"  {d['state_id']:>4} {d['name']:<18} {d.get('n_samples',0):>6} "
              f"{d.get('avg_return',0):>+8.3f}% {d.get('up_prob',0)*100:>6.1f}% "
              f"{d.get('std_return',0):>7.3f} {qs_str:>6}{flag}")


def show_backtest_results():
    if not os.path.exists(RESULTS_DIR):
        print("  Kein Ergebnisordner.")
        return
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('backtest_') and f.endswith('.json')]
    if not files:
        print("  Keine Backtest-Ergebnisse.")
        return
    print(f"\n  BACKTEST-ERGEBNISSE")
    for fname in sorted(files):
        try:
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                data = json.load(f)
        except Exception:
            continue
        stats  = data.get('stats', {})
        total  = stats.get('total_trades', 0)
        if total == 0:
            continue
        wr     = stats.get('win_rate', 0) * 100
        pnl    = stats.get('total_pnl_pct', 0)
        dd     = stats.get('max_drawdown_pct', 0)
        pf     = stats.get('profit_factor', 0)
        status = "PASS" if wr >= 50 and pnl > 0 and dd < 30 else "WARN"
        stars  = '★' * round(stats.get('avg_stars', 0)) if stats.get('avg_stars') else ''
        print(f"\n  [{status}] {data.get('market','?')} ({data.get('timeframe','?')})  "
              f"|  Trades: {total}  |  WR: {wr:.1f}%  |  "
              f"PF: {pf:.2f}  |  PnL: {pnl:+.1f}%  |  DD: {dd:.1f}%  |  {stars}")


def show_calibration_curve(store, market, tf, signal_name='knn'):
    """
    ASCII Kalibrierungskurve.
    Ideal: Bot sagt X% → trat ein X%.
    Überoptimistisch: Bot sagt 90% → trat ein 58%.
    """
    buckets = store.calibrator.get_calibration_curve(signal_name, market, tf, n_bins=10)
    if not buckets:
        print(f"\n  Calibration Curve: Noch keine Daten für {signal_name} / {market}/{tf}")
        return

    print(f"\n  CALIBRATION CURVE — {signal_name.upper()}  |  {market} ({tf})")
    print(f"  {'Predicted':>10} {'Actual':>8} {'N':>5}  Abweichung")
    print(f"  {'-'*50}")
    for b in buckets:
        pred = b['predicted_avg']
        act  = b['actual_rate']
        n    = b['n_samples']
        diff = act - pred
        sign = "▲" if diff > 0.05 else "▼" if diff < -0.05 else "≈"
        bar_len = round(act * 20)
        bar_empty = 20 - bar_len
        bar = "█" * bar_len + "░" * bar_empty
        print(f"  {pred*100:>8.0f}%  {act*100:>6.1f}%  {n:>5}  |{bar}| {sign}{abs(diff)*100:.0f}%")

    # Diagnose
    preds   = [b['predicted_avg'] for b in buckets]
    actuals = [b['actual_rate']   for b in buckets]
    bias    = sum(a - p for a, p in zip(actuals, preds)) / len(buckets)
    print(f"\n  Mittlere Abweichung: {bias*100:+.1f}%  "
          f"({'überoptimistisch' if bias < -0.03 else 'zu konservativ' if bias > 0.03 else 'gut kalibriert'})")


def show_signal_calibration(store, market, tf):
    stats = store.calibrator.get_signal_stats(market, tf)
    if not stats:
        print(f"\n  Signal-Kalibrierung: Noch keine Daten für {market}/{tf}")
        return
    print(f"\n  SIGNAL-KALIBRIERUNG: {market} ({tf})")
    print(f"  {'Signal':<14} {'Vorhersagen':>12} {'Bewertet':>9} {'Brier':>8} {'Genauigkeit':>12}  {'Reliability':>12}")
    print(f"  {'-'*72}")
    for s in stats:
        bs   = s.get('brier_score')
        acc  = s.get('accuracy')
        rel  = max(0.0, 1.0 - 4.0 * bs) if bs is not None else None
        rel_dyn = store.calibrator.get_reliability(s['signal_name'], market, tf)
        bs_str  = f"{bs:.4f}" if bs  is not None else "n/a"
        acc_str = f"{acc*100:.1f}%"  if acc is not None else "n/a"
        rel_str = f"{rel_dyn:.2f}"
        star    = " ★" if rel is not None and rel >= 0.80 else ""
        print(f"  {s['signal_name']:<14} {s['total']:>12} {s['evaluated']:>9} "
              f"{bs_str:>8} {acc_str:>12}  {rel_str:>12}{star}")


def show_maintenance_status():
    """Zeigt Maintenance-Log: wann wurde zuletzt geupdated / reclustert."""
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', '..', 'artifacts', 'maintenance_log.json')
    log_path = os.path.abspath(log_path)
    if not os.path.exists(log_path):
        print("\n  Maintenance-Log: Noch kein Eintrag (maintenance.py noch nicht gelaufen)")
        return
    try:
        with open(log_path) as f:
            log = json.load(f)
    except Exception:
        return

    print(f"\n  MAINTENANCE STATUS")
    now = datetime.now()
    for key, ts in sorted(log.items()):
        try:
            dt     = datetime.fromisoformat(ts)
            delta  = (now - dt.replace(tzinfo=None)).days
            age    = f"{delta}d" if delta < 365 else "alt"
            status = "OK" if delta < 7 else ("⚠" if delta < 30 else "!")
        except Exception:
            age, status = ts[:10], "?"
        print(f"  [{status}]  {key:<45} {age} alt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',    type=str, default=None)
    parser.add_argument('--timeframe', type=str, default=None)
    parser.add_argument('--states',    action='store_true')
    parser.add_argument('--status',    action='store_true', help='Maintenance-Status')
    parser.add_argument('--signals',   action='store_true', help='Zeige Signal-Kalibrierung')
    parser.add_argument('--curve',     type=str,  default=None, metavar='SIGNAL',
                        help='Kalibrierungskurve für Signal (z.B. knn, markov)')
    args = parser.parse_args()

    print(f"\nstatebot Viewer  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    store = StateStore(DB_PATH)
    show_store_summary(store)

    if args.states and args.symbol and args.timeframe:
        show_state_definitions(store, args.symbol, args.timeframe)

    if args.signals and args.symbol and args.timeframe:
        show_signal_calibration(store, args.symbol, args.timeframe)

    if args.curve and args.symbol and args.timeframe:
        show_calibration_curve(store, args.symbol, args.timeframe, signal_name=args.curve)

    store.close()
    show_backtest_results()
    if args.status:
        show_maintenance_status()
    print()


if __name__ == "__main__":
    main()
