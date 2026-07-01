# analysis/ab_test.py — A/B-Test fuer State Filtering
#
# Testet die Hypothese:
#   H0: Alle States tragen gleich zur Edge bei
#   H1: Negative States entfernen verbessert Risk-adjusted Returns
#
# Methodik:
#   1. TRAIN → pareto_breakdown → bad_keys (Buckets mit PnL < 0)
#   2. TEST  → Vollstaendige Metriken vor/nach Filter
#   3. Bootstrapped CI fuer Delta E[V] (Paired Bootstrap — filter innerhalb Sample)
#   4. Formales Verdict: GAIN / NEUTRAL / LOSS
#
# Wichtig: Direkte String-Bucket-Matching fuer kategorische Felder (state_id,
# regime, stars). Fuer numerische Felder (membership, p_bayes) bad_values
# manuell als Set[str] uebergeben (Bucket-Strings aus pareto_breakdown()).
#
# CLI:
#   python -m statebot.analysis.ab_test \
#       --train artifacts/results/backtest_train.json \
#       --test  artifacts/results/backtest_test.json \
#       --field state_id

import os, sys, json, argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from statebot.analysis.attribution import pareto_breakdown, _get_field

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'results')


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _bucket_key(trade: dict, field: str) -> str | None:
    """Gibt den String-Bucket-Key zurueck — identisch zu attribute() in attribution.py."""
    v = _get_field(trade, field)
    return str(v) if v is not None else None


def _metrics(trades: list[dict]) -> dict:
    """Erweiterte Metriken fuer Before/After-Vergleich."""
    if not trades:
        return {
            'n_trades': 0, 'ev': 0.0, 'profit_factor': 0.0, 'win_rate': 0.0,
            'std_pnl': 0.0, 'max_drawdown_pct': 0.0, 'total_pnl': 0.0,
        }
    pnls   = [t.get('pnl_usdt', 0.0) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    pf     = sum(wins) / sum(losses) if losses else float('inf')

    sorted_t = sorted(trades, key=lambda t: t.get('bar_time', ''))
    eq, peak, max_dd = 0.0, 0.0, 0.0
    for t in sorted_t:
        eq   += t.get('pnl_usdt', 0.0)
        peak  = max(peak, eq)
        dd    = (peak - eq) / abs(peak) * 100 if peak != 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        'n_trades':         len(trades),
        'ev':               float(np.mean(pnls)),
        'profit_factor':    round(min(pf, 999.0), 3),
        'win_rate':         len(wins) / len(pnls),
        'std_pnl':          float(np.std(pnls)),
        'max_drawdown_pct': round(max_dd, 2),
        'total_pnl':        round(float(sum(pnls)), 2),
    }


def _bootstrap_ci(trades_test: list[dict], bad_keys: set[str], field: str,
                   n_bootstrap: int = 2000,
                   ci_level:    float = 0.95) -> tuple[float, float]:
    """
    Paired Bootstrap fuer Delta E[V] CI.

    Korrekte Methode: Jedes Bootstrap-Sample wird gefiltert (statt zwei
    unabhaengige Samples) — weil 'after' ein Subset von 'before' ist.

    Returns: (ci_lo, ci_hi) fuer delta = E[V]_after - E[V]_before
    """
    if not trades_test:
        return 0.0, 0.0

    n   = len(trades_test)
    rng = np.random.default_rng(42)
    deltas = []

    for _ in range(n_bootstrap):
        idx    = rng.integers(0, n, size=n)
        sample = [trades_test[i] for i in idx]
        after  = [t for t in sample if _bucket_key(t, field) not in bad_keys]

        ev_b = float(np.mean([t.get('pnl_usdt', 0.0) for t in sample]))
        if after:
            ev_a = float(np.mean([t.get('pnl_usdt', 0.0) for t in after]))
        else:
            ev_a = ev_b

        deltas.append(ev_a - ev_b)

    alpha = (1 - ci_level) / 2
    return float(np.quantile(deltas, alpha)), float(np.quantile(deltas, 1 - alpha))


# ─── Haupt-API ────────────────────────────────────────────────────────────────

def run_ab_test(trades_train: list[dict],
                trades_test:  list[dict],
                field:        str        = 'state_id',
                min_trades:   int        = 3,
                bad_values:   set | None = None,
                n_bootstrap:  int        = 2000,
                ci_level:     float      = 0.95) -> dict:
    """
    Sauberer A/B-Test fuer State Filtering.

    H0: Alle States tragen gleich zur Edge bei
    H1: Negative States entfernen verbessert Risk-adjusted Returns

    Args:
        trades_train: Trades aus dem TRAINING-Zeitraum (Pareto-Ableitung)
        trades_test:  Trades aus dem TEST-Zeitraum (OOS-Evaluation)
        field:        Feld fuer Bucket-Selektion (state_id, regime, stars, ...)
        min_trades:   Mindest-Trades pro Bucket im Training
        bad_values:   Optionale manuelle Bad-Bucket-Keys (ueberschreibt Pareto-Ableitung)
                      Muss Set[str] sein (Bucket-Strings wie '4', 'TREND', '0.5-0.65')
        n_bootstrap:  Bootstrap-Iterationen fuer CI
        ci_level:     Konfidenzintervall-Niveau (0.95 = 95%)

    Returns:
        Dict mit: field, bad_keys, n_*, metrics_before, metrics_after, delta,
                  bootstrap (CI, significant), verdict (GAIN/NEUTRAL/LOSS)

    Verdict-Logik:
        GAIN    — CI vollstaendig ueber 0: Filtering bringt strukturellen Vorteil
        LOSS    — CI vollstaendig unter 0: Filtering schadet (States zu gekoppelt)
        NEUTRAL — CI schneidet 0: kein signifikanter Effekt
    """
    # 1. Bad Buckets aus Training ableiten (oder manuell)
    if bad_values is not None:
        bad_keys = {str(v) for v in bad_values}
    else:
        train_pareto = pareto_breakdown(trades_train, field=field, min_trades=min_trades)
        bad_keys = {str(r['bucket']) for r in train_pareto if r['total_pnl'] < 0}

    # 2. Test-Set filtern
    trades_after = [t for t in trades_test if _bucket_key(t, field) not in bad_keys]
    n_removed    = len(trades_test) - len(trades_after)

    # 3. Metriken
    m_before = _metrics(trades_test)
    m_after  = _metrics(trades_after)

    # 4. Beobachteter Delta + Bootstrap CI
    obs_delta_ev = m_after['ev'] - m_before['ev']
    ci_lo, ci_hi = _bootstrap_ci(
        trades_test, bad_keys, field,
        n_bootstrap=n_bootstrap, ci_level=ci_level,
    )

    # 5. Verdict
    if ci_lo > 0:
        verdict = 'GAIN'
    elif ci_hi < 0:
        verdict = 'LOSS'
    else:
        verdict = 'NEUTRAL'

    def _d(key): return round(m_after[key] - m_before[key], 4)

    return {
        'field':           field,
        'bad_keys':        sorted(bad_keys),
        'n_train':         len(trades_train),
        'n_test_before':   len(trades_test),
        'n_test_after':    len(trades_after),
        'n_removed':       n_removed,
        'pct_removed':     round(n_removed / len(trades_test) * 100 if trades_test else 0.0, 1),
        'metrics_before':  m_before,
        'metrics_after':   m_after,
        'delta': {
            'ev':               round(obs_delta_ev, 4),
            'profit_factor':    round(m_after['profit_factor'] - m_before['profit_factor'], 3),
            'win_rate':         _d('win_rate'),
            'std_pnl':          _d('std_pnl'),
            'max_drawdown_pct': round(m_after['max_drawdown_pct'] - m_before['max_drawdown_pct'], 2),
            'n_trades':         m_after['n_trades'] - m_before['n_trades'],
            'total_pnl':        round(m_after['total_pnl'] - m_before['total_pnl'], 2),
        },
        'bootstrap': {
            'n':           n_bootstrap,
            'ci_level':    ci_level,
            'delta_ev':    round(obs_delta_ev, 4),
            'ci_lo':       round(ci_lo, 4),
            'ci_hi':       round(ci_hi, 4),
            'significant': ci_lo > 0 or ci_hi < 0,
        },
        'verdict': verdict,
    }


def print_ab_result(result: dict) -> None:
    """Formatierte Terminal-Ausgabe des A/B-Test-Ergebnisses."""
    field    = result['field']
    bad_keys = result['bad_keys']
    mb       = result['metrics_before']
    ma       = result['metrics_after']
    d        = result['delta']
    bs       = result['bootstrap']
    verdict  = result['verdict']

    print(f"\n{'='*64}")
    print(f"  A/B TEST: State Filtering  (field={field})")
    print(f"  Train: {result['n_train']} Trades  |  "
          f"Test: {result['n_test_before']} Trades")
    bk_str = ', '.join(bad_keys) if bad_keys else 'keine'
    print(f"  Negative {field}s aus Train (PnL < 0): {bk_str}")
    print(f"  Entfernt: {result['n_removed']} Trades ({result['pct_removed']:.1f}%)  |  "
          f"Verbleibend: {result['n_test_after']}")
    print(f"{'='*64}")

    def _pct(v): return f"{v*100:+.1f}%"

    print(f"\n  {'Metrik':<24} {'Vorher':>10} {'Nachher':>10} {'Delta':>10}")
    print(f"  {'-'*58}")
    print(f"  {'N Trades':<24} {mb['n_trades']:>10d} "
          f"{ma['n_trades']:>10d} {d['n_trades']:>+10d}")
    print(f"  {'E[V] (USDT/Trade)':<24} {mb['ev']:>+10.4f} "
          f"{ma['ev']:>+10.4f} {d['ev']:>+10.4f}")
    print(f"  {'Profit Factor':<24} {mb['profit_factor']:>10.3f} "
          f"{ma['profit_factor']:>10.3f} {d['profit_factor']:>+10.3f}")
    print(f"  {'Win Rate':<24} {mb['win_rate']:>9.1%} "
          f"{ma['win_rate']:>9.1%} {_pct(d['win_rate']):>10}")
    print(f"  {'StdDev PnL':<24} {mb['std_pnl']:>10.3f} "
          f"{ma['std_pnl']:>10.3f} {d['std_pnl']:>+10.3f}")
    print(f"  {'Max Drawdown %':<24} {mb['max_drawdown_pct']:>9.1f}% "
          f"{ma['max_drawdown_pct']:>9.1f}% {d['max_drawdown_pct']:>+9.1f}%")
    print(f"  {'Total PnL (USDT)':<24} {mb['total_pnl']:>+10.2f} "
          f"{ma['total_pnl']:>+10.2f} {d['total_pnl']:>+10.2f}")

    ci_lo = bs['ci_lo']
    ci_hi = bs['ci_hi']

    print(f"\n  Signifikanz ({int(bs['ci_level'] * 100)}%-CI, "
          f"n={bs['n']:,} Bootstrap)")
    print(f"  D E[V] = {bs['delta_ev']:+.4f}  "
          f"CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    if bs['significant'] and ci_lo > 0:
        print(f"  -> CI vollstaendig ueber 0: statistisch signifikant (positiv)")
    elif bs['significant'] and ci_hi < 0:
        print(f"  -> CI vollstaendig unter 0: statistisch signifikant (negativ)")
    else:
        print(f"  -> CI schneidet 0: NICHT signifikant (zufaellig / zu wenig Daten)")

    print(f"\n  Verdict: ", end='')
    if verdict == 'GAIN':
        print("GAIN")
        print(f"  State Filtering bringt strukturellen Vorteil.")
        print(f"  E[V] steigt, Effekt ist durch Bootstrap bestaetigt.")
    elif verdict == 'LOSS':
        print("LOSS")
        print(f"  State Filtering verschlechtert die Performance.")
        print(f"  States zu stark gekoppelt — Filter entfernt guten Edge.")
    else:
        print("NEUTRAL")
        print(f"  Kein signifikanter Effekt nachweisbar.")
        print(f"  States moeglicherweise bereits implizit durch KNN/Markov gefiltert.")
    print(f"{'='*64}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A/B-Test: State Filtering — struktureller Edge-Test'
    )
    parser.add_argument('--train', required=True,
                        help='JSON-Backtest-Ergebnis (Trainperiode, Schluessel: trades)')
    parser.add_argument('--test', required=True,
                        help='JSON-Backtest-Ergebnis (Testperiode, Schluessel: trades)')
    parser.add_argument('--field', default='state_id',
                        help='Feld fuer Bucket-Selektion (default: state_id)')
    parser.add_argument('--min-trades', type=int, default=3,
                        help='Mindest-Trades pro Bucket im Training (default: 3)')
    parser.add_argument('--bootstrap', type=int, default=2000,
                        help='Bootstrap-Iterationen (default: 2000)')
    parser.add_argument('--ci', type=float, default=0.95,
                        help='CI-Niveau (default: 0.95)')
    parser.add_argument('--bad', nargs='+',
                        help='Manuelle bad_keys (ueberschreibt Pareto-Ableitung)')
    parser.add_argument('--save', help='Ergebnis als JSON speichern')
    args = parser.parse_args()

    def _load(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    train_data = _load(args.train)
    test_data  = _load(args.test)

    trades_train = train_data.get('trades', [])
    trades_test  = test_data.get('trades', [])

    if not trades_train:
        print(f"FEHLER: Keine Trades in {args.train} (erwartet Schluessel 'trades')")
        sys.exit(1)
    if not trades_test:
        print(f"FEHLER: Keine Trades in {args.test} (erwartet Schluessel 'trades')")
        sys.exit(1)

    bad_values = set(args.bad) if args.bad else None

    result = run_ab_test(
        trades_train, trades_test,
        field=args.field,
        min_trades=args.min_trades,
        bad_values=bad_values,
        n_bootstrap=args.bootstrap,
        ci_level=args.ci,
    )
    print_ab_result(result)

    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"  Ergebnis gespeichert: {args.save}")
