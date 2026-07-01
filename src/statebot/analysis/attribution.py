# analysis/attribution.py — Trade Attribution Analyzer
#
# Beantwortet: Welche Modell-Komponenten tragen tatsächlich zum Gewinn bei?
#
# Generisches API:
#   attribute(trades, field)    → Slicet nach jedem Feld, gibt Metriken pro Bucket
#   run_full_attribution(trades)→ Alle Standard-Felder auf einmal
#   print_report(results)       → Terminal-Ausgabe
#   save_report(results, path)  → JSON-Export
#
# Felder werden aus dem Trade-Dict direkt oder aus 'prediction_snapshot' gelesen.
# Kategorische Felder → Group by Wert.
# Numerische Felder   → Automatisches Quantil-Bucketing (n=4 Standard).
#
# CLI:
#   python -m statebot.analysis.attribution --file artifacts/results/backtest_pnl_*.json
#   python -m statebot.analysis.attribution --file ... --field membership

import os, sys, json, argparse, math
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'results')

# Felder die immer als Kategorien behandelt werden (auch wenn sie Zahlen sind)
CATEGORICAL_OVERRIDES = {'state_id', 'stars', 'htf_trend', 'markov_order', 'side', 'outcome'}

# Standard-Felder für run_full_attribution()
DEFAULT_FIELDS = [
    'state_id',
    'state_name',
    'regime',
    'stars',
    'membership',
    'quality_score',
    'confidence',
    'p_prior',
    'p_knn',
    'p_bayes',
    'k_used',
    'htf_trend',
    'trailing_activated',
    'structure_sl_applied',
]


# ═══════════════════════════════════════════════════════════════════════════════
# Kern-Logik
# ═══════════════════════════════════════════════════════════════════════════════

def _get_field(trade: dict, field: str):
    """Liest Feld aus Trade-Dict oder aus trade['prediction_snapshot']."""
    if field in trade and trade[field] is not None:
        return trade[field]
    snap = trade.get('prediction_snapshot') or {}
    return snap.get(field)


def _trade_metrics(group: list[dict]) -> dict:
    """Berechnet Attribution-Metriken für eine Gruppe von Trades."""
    n = len(group)
    if n == 0:
        return {}
    wins    = [t for t in group if t.get('outcome') == 'WIN']
    losses  = [t for t in group if t.get('outcome') == 'LOSS']
    pnls    = [t['pnl_usdt'] for t in group if 'pnl_usdt' in t]

    wr      = len(wins) / n
    avg_pnl = float(np.mean(pnls)) if pnls else 0.0
    med_pnl = float(np.median(pnls)) if pnls else 0.0
    sum_pos = sum(p for p in pnls if p > 0)
    sum_neg = abs(sum(p for p in pnls if p < 0))
    pf      = sum_pos / sum_neg if sum_neg > 0 else (float('inf') if sum_pos > 0 else 0.0)
    total   = float(sum(pnls)) if pnls else 0.0

    # Max Drawdown innerhalb der Gruppe (Reihenfolge temporal)
    sorted_trades = sorted(group, key=lambda t: t.get('bar_time', ''))
    eq     = 0.0
    peak   = 0.0
    max_dd = 0.0
    for t in sorted_trades:
        eq   += t.get('pnl_usdt', 0.0)
        peak  = max(peak, eq)
        dd    = (peak - eq) / abs(peak) * 100 if peak != 0 else 0.0
        max_dd = max(max_dd, dd)

    # Erwartungswert pro Trade
    avg_win  = float(np.mean([t['pnl_usdt'] for t in wins]))  if wins  else 0.0
    avg_loss = float(np.mean([t['pnl_usdt'] for t in losses])) if losses else 0.0
    expectancy = wr * avg_win + (1 - wr) * avg_loss

    return {
        'n_trades':       n,
        'wins':           len(wins),
        'losses':         len(losses),
        'win_rate':       round(wr, 3),
        'avg_pnl':        round(avg_pnl, 4),
        'median_pnl':     round(med_pnl, 4),
        'profit_factor':  round(pf, 3) if not math.isinf(pf) else 999.0,
        'expectancy':     round(expectancy, 4),
        'total_pnl':      round(total, 2),
        'max_drawdown':   round(max_dd, 2),
    }


def attribute(trades: list[dict], field: str, n_buckets: int = 4) -> list[dict]:
    """
    Slicet eine Liste von Trades nach einem Feld und gibt Metriken pro Bucket zurück.

    Kategorische Felder → ein Bucket pro einzigartigem Wert.
    Numerische Felder   → n_buckets Quantile.

    Returns:
        Liste von Dicts mit Schlüsseln: bucket, n_trades, win_rate, avg_pnl,
        median_pnl, profit_factor, expectancy, total_pnl, max_drawdown
    """
    pairs = [(_get_field(t, field), t) for t in trades]
    pairs = [(v, t) for v, t in pairs if v is not None]

    if not pairs:
        return []

    first_val = pairs[0][0]
    is_numeric = isinstance(first_val, (int, float)) and field not in CATEGORICAL_OVERRIDES

    if is_numeric:
        vals = [v for v, _ in pairs]
        percentiles = np.linspace(0, 100, n_buckets + 1)
        edges = sorted(set(float(np.percentile(vals, p)) for p in percentiles))

        if len(edges) < 2:
            groups = {'all': [t for _, t in pairs]}
            is_numeric = False
        else:
            groups: dict[str, list] = {}
            for v, t in pairs:
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i + 1]
                    at_last = (i == len(edges) - 2)
                    if lo <= v < hi or (at_last and v == hi):
                        key = f"{lo:.3g}-{hi:.3g}"
                        groups.setdefault(key, []).append(t)
                        break

    if not is_numeric:
        groups = {}
        for v, t in pairs:
            key = str(v)
            groups.setdefault(key, []).append(t)

    results = []
    for bucket_key, bucket_trades in sorted(
            groups.items(),
            key=lambda x: -len(x[1]) if not is_numeric else 0):
        row = {'bucket': bucket_key, 'field': field}
        row.update(_trade_metrics(bucket_trades))
        results.append(row)

    # Für numerische Felder Buckets in Reihenfolge sortieren
    if is_numeric:
        results.sort(key=lambda r: float(r['bucket'].split('-')[0]))

    return results


def run_full_attribution(trades: list[dict],
                          fields: list[str] | None = None,
                          n_buckets: int = 4) -> dict[str, list[dict]]:
    """Führt Attribution für alle Standard-Felder (oder übergebene) durch."""
    if not trades:
        return {}
    target_fields = fields or DEFAULT_FIELDS
    return {f: attribute(trades, f, n_buckets=n_buckets) for f in target_fields}


# ═══════════════════════════════════════════════════════════════════════════════
# Ausgabe
# ═══════════════════════════════════════════════════════════════════════════════

def _pf_str(pf: float) -> str:
    return '∞' if pf >= 999 else f"{pf:.2f}"


def print_attribution_table(field: str, rows: list[dict], min_trades: int = 3):
    """Gibt eine formatierte Attribution-Tabelle für ein Feld aus."""
    rows = [r for r in rows if r.get('n_trades', 0) >= min_trades]
    if not rows:
        return

    print(f"\n  ── {field} ──")
    print(f"  {'Bucket':<22} {'N':>5} {'WR':>7} {'AvgPnL':>9} {'PF':>6} {'E[V]':>9} {'Total':>9}")
    print("  " + "─" * 72)
    for r in rows:
        wr_s = f"{r['win_rate']:.0%}"
        ev_s = f"{r['expectancy']:+.3f}"
        tot  = f"{r['total_pnl']:+.2f}"
        avg  = f"{r['avg_pnl']:+.4f}"
        print(f"  {str(r['bucket']):<22} {r['n_trades']:>5} {wr_s:>7} {avg:>9} "
              f"{_pf_str(r['profit_factor']):>6} {ev_s:>9} {tot:>9}")


def print_report(results: dict[str, list[dict]], title: str = ""):
    """Gibt den vollständigen Attribution-Report aus."""
    print("\n" + "=" * 76)
    if title:
        print(f"  TRADE ATTRIBUTION  |  {title}")
    else:
        print("  TRADE ATTRIBUTION")
    print("=" * 76)
    for field, rows in results.items():
        print_attribution_table(field, rows)
    print("\n" + "=" * 76)


def save_report(results: dict[str, list[dict]],
                metadata: dict,
                path: str | None = None) -> str:
    """Speichert den Report als JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if path is None:
        market = metadata.get('market', 'unknown').replace('/', '').replace(':', '')
        tf     = metadata.get('timeframe', 'x')
        path   = os.path.join(RESULTS_DIR, f"attribution_{market}_{tf}.json")
    output = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'metadata':     metadata,
        'attribution':  results,
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Kurzanalysen
# ═══════════════════════════════════════════════════════════════════════════════

def best_states(trades: list[dict], top_n: int = 5) -> list[dict]:
    """Gibt die Top-N profitabelsten States zurück (min. 5 Trades)."""
    rows = attribute(trades, 'state_id')
    qualified = [r for r in rows if r['n_trades'] >= 5]
    return sorted(qualified, key=lambda r: r['expectancy'], reverse=True)[:top_n]


def component_summary(trades: list[dict]) -> dict:
    """
    Schnelle Übersicht: Trägt jede Komponente positiv zum Erwartungswert bei?

    Returns:
        Dict: Komponente → {'with': E[V], 'without': E[V], 'delta': E[V]}
    """
    def ev_split(field: str, val):
        grp_with    = [t for t in trades if _get_field(t, field) == val]
        grp_without = [t for t in trades if _get_field(t, field) != val]
        m_with    = _trade_metrics(grp_with)
        m_without = _trade_metrics(grp_without)
        return {
            f'with_{field}_{val}':    m_with.get('expectancy', 0),
            f'without_{field}_{val}': m_without.get('expectancy', 0),
            'delta': m_with.get('expectancy', 0) - m_without.get('expectancy', 0),
            'n_with': len(grp_with),
            'n_without': len(grp_without),
        }

    return {
        'trailing_stop':       ev_split('trailing_activated', True),
        'structure_protection': ev_split('structure_sl_applied', True),
        'htf_aligned':         ev_split('htf_trend', 1),   # Long + HTF bullish
    }


def print_component_summary(summary: dict):
    """Gibt den Component-Summary als Text aus."""
    print("\n  KOMPONENTEN-BEITRAG (D Erwartungswert)")
    print("  " + "─" * 52)
    for name, data in summary.items():
        delta = data.get('delta', 0)
        n_w   = data.get('n_with', 0)
        sign  = '+' if delta >= 0 else ''
        print(f"  {name:<28}  D E[V] = {sign}{delta:+.4f}  (n={n_w})")


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Decomposition — wo entsteht der Erwartungswert?
# ═══════════════════════════════════════════════════════════════════════════════

def pareto_breakdown(trades: list[dict],
                     field: str = 'state_id',
                     min_trades: int = 3) -> list[dict]:
    """
    Pareto-Analyse: Welche 20% der Buckets erzeugen 80% des Profits?

    Sortiert Buckets nach absolutem PnL-Beitrag (absteigend) und berechnet
    kumulativen Anteil am Gesamt-Edge. Hilft zu erkennen:

      - Welche States sind unverzichtbar?
      - Auf welche States koennte man verzichten ohne viel Edge zu verlieren?
      - Gibt es States die negativ beitragen (Edge-Zaehler)?

    Returns:
      Liste von Dicts mit: bucket, total_pnl, pnl_share_pct, cumulative_pct,
                           n_trades, win_rate, expectancy
    """
    rows  = attribute(trades, field)
    rows  = [r for r in rows if r['n_trades'] >= min_trades]
    total = sum(r['total_pnl'] for r in rows)

    if total == 0:
        return []

    # Positive Edge-Quellen zuerst (nach absolutem Beitrag)
    rows_sorted = sorted(rows, key=lambda r: r['total_pnl'], reverse=True)

    cumulative = 0.0
    result     = []
    for r in rows_sorted:
        share      = r['total_pnl'] / total * 100
        cumulative += share
        result.append({
            'bucket':         r['bucket'],
            'total_pnl':      r['total_pnl'],
            'pnl_share_pct':  round(share, 1),
            'cumulative_pct': round(cumulative, 1),
            'n_trades':       r['n_trades'],
            'win_rate':       r['win_rate'],
            'expectancy':     r['expectancy'],
        })
    return result


def print_pareto_breakdown(rows: list[dict], field: str = 'state_id'):
    """Gibt die Pareto-Tabelle aus."""
    if not rows:
        return
    total_pnl = sum(r['total_pnl'] for r in rows)
    print(f"\n  EDGE DECOMPOSITION: {field}  (Gesamt: {total_pnl:+.2f} USDT)")
    print(f"  {'Bucket':<22} {'PnL':>9} {'Anteil':>8} {'Kumulat.':>9} {'N':>5} {'WR':>6}")
    print("  " + "-" * 64)
    for r in rows:
        marker = '  <<< 80%' if r['cumulative_pct'] <= 80.0 and r['pnl_share_pct'] > 0 else ''
        print(f"  {str(r['bucket']):<22} {r['total_pnl']:>+9.2f} "
              f"{r['pnl_share_pct']:>7.1f}% {r['cumulative_pct']:>8.1f}%"
              f" {r['n_trades']:>5} {r['win_rate']:>6.0%}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# Research Report — kompakter Gesamtüberblick (ein Befehl, alles auf einmal)
# ═══════════════════════════════════════════════════════════════════════════════

def _calibration_quality(trades: list[dict]) -> tuple[str, float]:
    """
    Bewertet Kalibrierung: p_bayes der Gewinner sollte > p_bayes der Verlierer sein.
    Returns (label, delta_p_bayes)
    """
    wins   = [t for t in trades if t.get('outcome') == 'WIN'  and 'p_bayes' in t]
    losses = [t for t in trades if t.get('outcome') == 'LOSS' and 'p_bayes' in t]
    if not wins or not losses:
        return 'n/a', 0.0
    delta = float(np.mean([t['p_bayes'] for t in wins])) \
          - float(np.mean([t['p_bayes'] for t in losses]))
    label = 'Gut'   if delta >= 0.10 else \
            'Mittel' if delta >= 0.05 else 'Schwach'
    return label, round(delta, 3)


def expected_calibration_error(trades: list[dict],
                               n_bins: int = 10) -> tuple[float, list[dict]]:
    """
    Expected Calibration Error (ECE) — Kernfrage:
    "War eine 72%-Prognose langfristig tatsaechlich ca. 72% Treffer?"

    Methode:
      Teilt p_bayes in n_bins gleich breite Intervalle.
      Pro Bin: confidence = mean(p_bayes), accuracy = Trefferquote.
      ECE = sum( |accuracy - confidence| * n_bin / n_total )

      ECE = 0.00  → perfekt kalibriert
      ECE < 0.05  → gut
      ECE < 0.10  → akzeptabel
      ECE > 0.10  → schlecht kalibriert

    Returns:
      ece    — Gesamtfehler (0.0 bis 1.0)
      curve  — Liste von Dicts pro Bin fuer Reliability-Kurven
    """
    pairs = [(t.get('p_bayes', 0.5), t.get('outcome') == 'WIN')
             for t in trades if 'p_bayes' in t and 'outcome' in t]
    if len(pairs) < 10:
        return 0.0, []

    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = len(pairs)
    ece     = 0.0
    curve   = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        include_last = (i == n_bins - 1)
        group = [(p, w) for p, w in pairs
                 if lo <= p < hi or (include_last and p == hi)]
        if not group:
            continue
        confidence = float(np.mean([p for p, _ in group]))
        accuracy   = float(np.mean([float(w) for _, w in group]))
        n_bin      = len(group)
        gap        = accuracy - confidence
        ece       += abs(gap) * n_bin / n_total
        curve.append({
            'bin':        f"{lo:.1f}-{hi:.1f}",
            'confidence': round(confidence, 3),
            'accuracy':   round(accuracy, 3),
            'gap':        round(gap, 3),
            'n':          n_bin,
        })

    return round(ece, 4), curve


def brier_score(trades: list[dict]) -> float:
    """
    Brier Score = mean( (p_bayes - actual)^2 )
    0.0 = perfekt, 0.25 = uninformatives Modell (immer 0.5), 1.0 = maximal falsch.
    Misst die Schaerfe der Wahrscheinlichkeitsvorhersagen.
    """
    pairs = [(t.get('p_bayes', 0.5), 1.0 if t.get('outcome') == 'WIN' else 0.0)
             for t in trades if 'p_bayes' in t and 'outcome' in t]
    if not pairs:
        return 0.25
    return round(float(np.mean([(p - a) ** 2 for p, a in pairs])), 4)


def _drift_signal(trades: list[dict]) -> tuple[str, float, float]:
    """
    Vergleicht Profit-Factor der ersten vs. zweiten Tradehalfte (temporal).
    Returns (label, pf_early, pf_late)
    """
    sorted_t = sorted(trades, key=lambda t: t.get('bar_time', ''))
    mid      = len(sorted_t) // 2
    early    = sorted_t[:mid]
    late     = sorted_t[mid:]

    def _pf(grp):
        pos = sum(t['pnl_usdt'] for t in grp if t.get('pnl_usdt', 0) > 0)
        neg = abs(sum(t['pnl_usdt'] for t in grp if t.get('pnl_usdt', 0) < 0))
        return round(pos / neg, 3) if neg > 0 else 999.0

    pf_e = _pf(early)
    pf_l = _pf(late)
    diff  = pf_l - pf_e

    if abs(diff) < 0.30:
        label = 'Keiner'
    elif diff > 0:
        label = 'Verbessernd'
    else:
        label = 'Verschlechternd'

    return label, pf_e, pf_l


def generate_research_report(trades: list[dict], title: str = "") -> dict:
    """
    Kompakter Gesamtüberblick — alle wichtigen Attribution-Ergebnisse in einem Dict.

    Enthält:
        overview:         Basis-Kennzahlen (Trades, WR, PF, E[V], Total PnL)
        best_states:      Top-3 States nach Expectancy (min. 5 Trades)
        worst_states:     Bottom-3 States nach Expectancy (min. 5 Trades)
        membership_buckets: 4 Quantile
        regime_buckets:   TREND / RANGE / NEUTRAL
        stars_buckets:    1–5 Sterne
        component:        HTF / Structure SL / Trailing — absoluter + relativer Beitrag
        calibration:      Kalibrierungsqualität und D p_bayes
        drift:            PF früh vs. spät
    """
    if not trades:
        return {}

    total  = len(trades)
    pnls   = [t.get('pnl_usdt', 0) for t in trades]
    wins   = [t for t in trades if t.get('outcome') == 'WIN']
    losses = [t for t in trades if t.get('outcome') == 'LOSS']
    wr     = len(wins) / total
    sum_pos = sum(p for p in pnls if p > 0)
    sum_neg = abs(sum(p for p in pnls if p < 0))
    pf      = round(sum_pos / sum_neg, 3) if sum_neg > 0 else 999.0
    ev      = round(sum(pnls) / total, 4)

    overview = {
        'total_trades': total,
        'wins': len(wins),
        'losses': len(losses),
        'timeouts': total - len(wins) - len(losses),
        'win_rate': round(wr, 3),
        'profit_factor': pf,
        'expectancy': ev,
        'total_pnl': round(sum(pnls), 2),
    }

    # States
    state_rows  = attribute(trades, 'state_id')
    qualified   = [r for r in state_rows if r['n_trades'] >= 5]
    best_st     = sorted(qualified, key=lambda r: r['expectancy'], reverse=True)[:3]
    worst_st    = sorted(qualified, key=lambda r: r['expectancy'])[:3]

    # Numerische Buckets
    mem_rows    = attribute(trades, 'membership',   n_buckets=4)
    regime_rows = attribute(trades, 'regime')
    stars_rows  = attribute(trades, 'stars')

    # Komponenten-Beitrag (absolut + relativ zur Gesamt-Expectancy)
    comp = component_summary(trades)
    comp_enriched = {}
    for name, data in comp.items():
        delta    = data.get('delta', 0.0)
        rel_pct  = round(delta / abs(ev) * 100, 1) if ev != 0 else None
        comp_enriched[name] = {**data, 'delta_pct_of_ev': rel_pct}

    calib_label, calib_delta = _calibration_quality(trades)
    drift_label, pf_e, pf_l  = _drift_signal(trades)
    ece, calib_curve          = expected_calibration_error(trades)
    bs                        = brier_score(trades)

    return {
        'title':             title,
        'overview':          overview,
        'best_states':       best_st,
        'worst_states':      worst_st,
        'membership_buckets': mem_rows,
        'regime_buckets':    regime_rows,
        'stars_buckets':     stars_rows,
        'component':         comp_enriched,
        'calibration': {
            'label':          calib_label,
            'delta_p_bayes':  calib_delta,
            'ece':            ece,
            'brier_score':    bs,
            'curve':          calib_curve,
        },
        'drift': {'label': drift_label, 'pf_early': pf_e, 'pf_late': pf_l},
    }


def print_research_report(report: dict):
    """Gibt den Research Report formatiert auf dem Terminal aus."""
    W = 54
    print("\n" + "=" * W)
    print(f"  STATEBOT RESEARCH REPORT")
    if report.get('title'):
        print(f"  {report['title']}")
    print("=" * W)

    ov = report.get('overview', {})
    print(f"\n  Trades:         {ov.get('total_trades', 0)}"
          f"   WR: {ov.get('win_rate', 0):.0%}")
    print(f"  Profit Factor:  {_pf_str(ov.get('profit_factor', 0))}"
          f"   E[V]: {ov.get('expectancy', 0):+.4f} USDT")
    print(f"  Total PnL:      {ov.get('total_pnl', 0):+.2f} USDT")

    # States
    def _state_line(r):
        bid = r['bucket']
        return (f"  #{bid:<6} PF={_pf_str(r['profit_factor'])}  "
                f"E[V]={r['expectancy']:+.3f}  n={r['n_trades']}")

    if report.get('best_states'):
        print(f"\n  STATES -- Beste")
        for r in report['best_states']:
            print(_state_line(r))
    if report.get('worst_states'):
        print(f"\n  STATES -- Schlechteste")
        for r in report['worst_states']:
            print(_state_line(r))

    # Membership
    if report.get('membership_buckets'):
        print(f"\n  MEMBERSHIP")
        for r in report['membership_buckets']:
            if r['n_trades'] >= 3:
                print(f"  {r['bucket']:<18}  PF={_pf_str(r['profit_factor'])}"
                      f"  WR={r['win_rate']:.0%}  E[V]={r['expectancy']:+.3f}")

    # Regime
    if report.get('regime_buckets'):
        print(f"\n  REGIME")
        for r in report['regime_buckets']:
            if r['n_trades'] >= 3:
                print(f"  {r['bucket']:<12}  PF={_pf_str(r['profit_factor'])}"
                      f"  E[V]={r['expectancy']:+.3f}  n={r['n_trades']}")

    # Komponenten
    comp = report.get('component', {})
    if comp:
        print(f"\n  KOMPONENTEN-BEITRAG  (Basis E[V]={report['overview'].get('expectancy', 0):+.4f})")
        comp_labels = {
            'trailing_stop':        'Trailing Stop',
            'structure_protection': 'Structure SL',
            'htf_aligned':          'HTF Filter',
        }
        for key, label in comp_labels.items():
            c = comp.get(key, {})
            delta    = c.get('delta', 0.0)
            rel      = c.get('delta_pct_of_ev')
            n        = c.get('n_with', 0)
            rel_str  = f"  ({rel:+.0f}% von E[V])" if rel is not None else ""
            print(f"  {label:<22}  D={delta:+.4f} USDT{rel_str}  n={n}")

    # Stars
    if report.get('stars_buckets'):
        print(f"\n  STERNE-QUALITAET")
        for r in sorted(report['stars_buckets'], key=lambda x: -int(x['bucket'])):
            if r['n_trades'] >= 3:
                label = f"{'*' * int(r['bucket'])} ({r['bucket']})"
                print(f"  {label:<12}  PF={_pf_str(r['profit_factor'])}"
                      f"  E[V]={r['expectancy']:+.3f}  n={r['n_trades']}")

    # Kalibrierung & Drift
    cal = report.get('calibration', {})
    drf = report.get('drift', {})

    ece = cal.get('ece', None)
    bs  = cal.get('brier_score', None)
    ece_label = ('gut' if ece is not None and ece < 0.05
                 else 'akzeptabel' if ece is not None and ece < 0.10
                 else 'schlecht' if ece is not None else 'n/a')
    ece_str = f"{ece:.4f} [{ece_label}]" if ece is not None else 'n/a'
    bs_str  = f"{bs:.4f}" if bs is not None else 'n/a'

    print(f"\n  KALIBRIERUNG")
    print(f"  ECE:          {ece_str}   (0=perfekt, <0.05=gut, <0.10=ok)")
    print(f"  Brier Score:  {bs_str}   (0=perfekt, 0.25=uninformativ)")
    print(f"  D p_bayes:    {cal.get('delta_p_bayes', 0):+.3f}"
          f"   (Gewinner minus Verlierer — Trennschaerfe)")

    curve = cal.get('curve', [])
    if curve:
        print(f"\n  RELIABILITY CURVE  (p_bayes-Bin vs. tatsaechl. Trefferquote)")
        print(f"  {'Bin':<12} {'Conf':>7} {'Acc':>7} {'Gap':>7} {'N':>5}")
        print("  " + "-" * 40)
        for b in curve:
            gap_s = f"{b['gap']:+.3f}"
            flag  = "  <<" if abs(b['gap']) > 0.10 else ""
            print(f"  {b['bin']:<12} {b['confidence']:>7.3f} {b['accuracy']:>7.3f}"
                  f" {gap_s:>7} {b['n']:>5}{flag}")

    print(f"\n  DRIFT:        {drf.get('label', 'n/a')}"
          f"  (frueh PF={_pf_str(drf.get('pf_early', 0))}"
          f"  spaet PF={_pf_str(drf.get('pf_late', 0))})")

    print("\n" + "=" * W)


# ═══════════════════════════════════════════════════════════════════════════════
# Stability Report — Cross-Coin Robustheit
# ═══════════════════════════════════════════════════════════════════════════════

def generate_stability_report(coin_results: dict[str, list[dict]]) -> dict:
    """
    Fasst Research-Ergebnisse mehrerer Coins zu einem Stabilitaets-Bericht zusammen.

    coin_results: {coin_label: trades_list}
        z.B. {'BTC/USDT 1d': trades_btc, 'ETH/USDT 1d': trades_eth, ...}

    Returns:
        per_coin:   Metriken pro Coin
        aggregate:  Median/StdDev/Min/Max ueber alle Coins
        stability:  Anteil Coins mit PF > 1.0, E[V] > 0
    """
    per_coin: dict[str, dict] = {}
    for label, trades in coin_results.items():
        if not trades:
            continue
        pnls    = [t.get('pnl_usdt', 0) for t in trades]
        wins    = [t for t in trades if t.get('outcome') == 'WIN']
        total   = len(trades)
        sum_pos = sum(p for p in pnls if p > 0)
        sum_neg = abs(sum(p for p in pnls if p < 0))
        pf      = round(sum_pos / sum_neg, 3) if sum_neg > 0 else 999.0
        ev      = round(sum(pnls) / total, 4) if total > 0 else 0.0
        wr      = round(len(wins) / total, 3) if total > 0 else 0.0
        ece, _  = expected_calibration_error(trades)
        bs      = brier_score(trades)
        per_coin[label] = {
            'n_trades':      total,
            'win_rate':      wr,
            'profit_factor': pf,
            'expectancy':    ev,
            'total_pnl':     round(sum(pnls), 2),
            'ece':           ece,
            'brier_score':   bs,
        }

    if not per_coin:
        return {}

    pf_vals  = [v['profit_factor'] for v in per_coin.values() if v['profit_factor'] < 999]
    ev_vals  = [v['expectancy']    for v in per_coin.values()]
    ece_vals = [v['ece']           for v in per_coin.values() if v.get('ece') is not None]
    bs_vals  = [v['brier_score']   for v in per_coin.values() if v.get('brier_score') is not None]

    aggregate = {
        'median_pf':    round(float(np.median(pf_vals)), 3)    if pf_vals  else 0.0,
        'std_pf':       round(float(np.std(pf_vals)), 3)       if pf_vals  else 0.0,
        'min_pf':       round(float(np.min(pf_vals)), 3)       if pf_vals  else 0.0,
        'max_pf':       round(float(np.max(pf_vals)), 3)       if pf_vals  else 0.0,
        'median_ev':    round(float(np.median(ev_vals)), 4)    if ev_vals  else 0.0,
        'std_ev':       round(float(np.std(ev_vals)), 4)       if ev_vals  else 0.0,
        'median_ece':   round(float(np.median(ece_vals)), 4)   if ece_vals else None,
        'median_brier': round(float(np.median(bs_vals)), 4)    if bs_vals  else None,
        'n_coins':      len(per_coin),
        'total_trades': sum(v['n_trades'] for v in per_coin.values()),
    }

    n = len(per_coin)
    stability = {
        'pf_above_1':     round(sum(1 for v in per_coin.values() if v['profit_factor'] > 1.0) / n, 2),
        'ev_positive':    round(sum(1 for v in per_coin.values() if v['expectancy'] > 0) / n, 2),
        'pf_above_1_5':   round(sum(1 for v in per_coin.values() if v['profit_factor'] > 1.5) / n, 2),
        'consistent':     aggregate['std_pf'] < 0.30,
    }

    return {'per_coin': per_coin, 'aggregate': aggregate, 'stability': stability}


def print_stability_report(report: dict):
    """Terminal-Ausgabe des Stabilitaets-Reports."""
    if not report:
        print("Kein Stability Report verfuegbar.")
        return

    W   = 62
    agg = report.get('aggregate', {})
    stb = report.get('stability', {})

    print("\n" + "=" * W)
    print("  STATEBOT STABILITY REPORT")
    print(f"  {agg.get('n_coins', 0)} Coins  |  {agg.get('total_trades', 0)} Trades gesamt")
    print("=" * W)
    print(f"\n  {'Coin':<26} {'N':>5} {'WR':>6} {'PF':>7} {'E[V]':>9} {'ECE':>7} {'Brier':>7}")
    print("  " + "-" * 68)

    per_coin = report.get('per_coin', {})
    for label, m in sorted(per_coin.items(), key=lambda x: -x[1]['profit_factor']):
        pf_s   = _pf_str(m['profit_factor'])
        wr_s   = f"{m['win_rate']:.0%}"
        ev_s   = f"{m['expectancy']:+.4f}"
        ece_s  = f"{m['ece']:.4f}"  if m.get('ece')  is not None else 'n/a'
        bs_s   = f"{m['brier_score']:.4f}" if m.get('brier_score') is not None else 'n/a'
        print(f"  {label:<26} {m['n_trades']:>5} {wr_s:>6} {pf_s:>7} {ev_s:>9} {ece_s:>7} {bs_s:>7}")

    print("\n  " + "-" * 68)
    print(f"  {'Median PF':<26} {agg.get('median_pf', 0):>7.3f}")
    print(f"  {'StdDev PF':<26} {agg.get('std_pf', 0):>7.3f}"
          f"   {'<< konsistent' if stb.get('consistent') else '>> inkonsistent'}")
    print(f"  {'Min / Max PF':<26} {agg.get('min_pf', 0):.3f} / {agg.get('max_pf', 0):.3f}")
    print(f"  {'Median E[V]':<26} {agg.get('median_ev', 0):>+9.4f} USDT")
    if agg.get('median_ece') is not None:
        ece_lbl = 'gut' if agg['median_ece'] < 0.05 else 'akzeptabel' if agg['median_ece'] < 0.10 else 'schlecht'
        print(f"  {'Median ECE':<26} {agg['median_ece']:.4f}   [{ece_lbl}]")
    if agg.get('median_brier') is not None:
        print(f"  {'Median Brier Score':<26} {agg['median_brier']:.4f}   (0.25=uninformativ)")

    print(f"\n  STABILITAET")
    print(f"  PF > 1.0   auf {stb.get('pf_above_1', 0):.0%} der Coins")
    print(f"  PF > 1.5   auf {stb.get('pf_above_1_5', 0):.0%} der Coins")
    print(f"  E[V] > 0   auf {stb.get('ev_positive', 0):.0%} der Coins")

    print("\n" + "=" * W)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Coin Invarianz — welche States sind ueber Environments stabil?
# ═══════════════════════════════════════════════════════════════════════════════

def cross_coin_pareto_consensus(coin_results: dict[str, list[dict]],
                                 field: str = 'state_id',
                                 min_trades: int = 3,
                                 pareto_threshold: float = 80.0) -> list[dict]:
    """
    Wie oft erscheint jeder State in der Pareto-80%-Zone verschiedener Coins?

    Hohe Coin-Frequenz → strukturell invarianter Edge (IRM-Proxy).
    Niedrige Frequenz  → environment-spezifisch (BTC-only, Bullenmarkt-Artefakt).

    coin_results: {coin_label: trades_list}
        z.B. {'BTC/USDT 1d': trades_btc, 'ETH/USDT 1d': trades_eth}

    Returns sorted list (invarianteste zuerst):
        bucket, n_coins, pct_coins, coins
    """
    counter: dict[str, dict] = {}
    total_coins = len(coin_results)

    for coin, trades in coin_results.items():
        if not trades:
            continue
        rows = pareto_breakdown(trades, field=field, min_trades=min_trades)
        in_zone = {
            str(r['bucket'])
            for r in rows
            if r['cumulative_pct'] <= pareto_threshold and r['pnl_share_pct'] > 0
        }
        for bucket in in_zone:
            if bucket not in counter:
                counter[bucket] = {'n_coins': 0, 'coins': []}
            counter[bucket]['n_coins'] += 1
            counter[bucket]['coins'].append(coin)

    results = [
        {
            'bucket':    k,
            'n_coins':   v['n_coins'],
            'pct_coins': round(v['n_coins'] / total_coins * 100, 1) if total_coins else 0.0,
            'coins':     sorted(v['coins']),
        }
        for k, v in counter.items()
    ]
    return sorted(results, key=lambda r: (-r['n_coins'], r['bucket']))


def print_cross_coin_consensus(rows: list[dict], field: str = 'state_id',
                                total_coins: int = 0):
    """Gibt die Cross-Coin Invarianz-Tabelle aus."""
    if not rows:
        print(f"  Keine Daten fuer {field}.")
        return
    n = total_coins or max((r['n_coins'] for r in rows), default=1)
    print(f"\n  CROSS-COIN INVARIANZ: {field}  ({n} Coins)")
    print(f"  {'Bucket':<22} {'Coins':>8} {'Anteil':>8}  Invarianz")
    print("  " + "-" * 52)
    for r in rows:
        pct = r['pct_coins']
        label = 'invariant' if pct >= 60 else 'partiell' if pct >= 40 else 'lokal'
        bar = '*' * int(pct / 10)
        print(f"  {str(r['bucket']):<22} {r['n_coins']:>4}/{n:<3} {pct:>7.0f}%  "
              f"{bar:<10}  {label}")
    print()


# ─── State Scorecard ─────────────────────────────────────────────────────────

def _state_brier(group: list[dict]) -> float | None:
    """Brier Score fuer eine State-Gruppe: mean((p_bayes - actual)^2)."""
    pairs = [
        (t.get('p_bayes'), 1.0 if t.get('outcome') == 'WIN' else 0.0)
        for t in group
        if t.get('p_bayes') is not None and t.get('outcome') in ('WIN', 'LOSS')
    ]
    if len(pairs) < 3:
        return None
    p_arr = np.array([p for p, _ in pairs])
    y_arr = np.array([y for _, y in pairs])
    return float(np.mean((p_arr - y_arr) ** 2))


def state_scorecard(coin_results: dict[str, list[dict]],
                    field: str = 'state_id',
                    min_trades: int = 3,
                    pareto_threshold: float = 80.0) -> list[dict]:
    """
    Drei-Achsen-Bewertung pro State: Profitability x Calibration x Invarianz.

    Profitability: E[V], PF, WR  aus kombinierten Trades (pareto_breakdown)
    Calibration:   Brier Score   pro State aus kombinierten Trades
    Invarianz:     pct_coins     in Pareto-80%-Zone (cross_coin_pareto_consensus)

    Scorecard-Sterne (0-3):
        +1  E[V] > 0
        +1  pct_coins >= 50%
        +1  Brier < 0.22

    Core Edge State = alle 3 Punkte (invariant, profitabel, kalibriert)

    Returns sorted list (Core zuerst, dann nach Invarianz, dann nach E[V]).
    """
    all_trades = [t for trades in coin_results.values() for t in trades]
    total_coins = len(coin_results)

    # Profitability: pareto gibt E[V], PF, WR pro Bucket
    pareto = pareto_breakdown(all_trades, field=field, min_trades=min_trades)
    pareto_map = {str(r['bucket']): r for r in pareto}

    # Calibration: Trades pro Bucket gruppieren
    groups: dict[str, list[dict]] = {}
    for t in all_trades:
        v = _get_field(t, field)
        if v is not None:
            groups.setdefault(str(v), []).append(t)

    # Invarianz: Cross-Coin Konsens
    consensus = cross_coin_pareto_consensus(
        coin_results, field=field,
        min_trades=min_trades, pareto_threshold=pareto_threshold,
    )
    consensus_map = {str(r['bucket']): r for r in consensus}

    rows = []
    for bucket, group in groups.items():
        if len(group) < min_trades:
            continue

        prof  = pareto_map.get(bucket, {})
        brier = _state_brier(group)
        inv   = consensus_map.get(bucket, {'n_coins': 0, 'pct_coins': 0.0})

        ev        = prof.get('expectancy', 0.0)
        pf        = prof.get('profit_factor', 0.0)
        wr        = prof.get('win_rate', 0.0)
        n_coins   = inv['n_coins']
        pct_coins = inv['pct_coins']

        score = 0
        if ev > 0:                              score += 1
        if pct_coins >= 50.0:                   score += 1
        if brier is not None and brier < 0.22:  score += 1

        rows.append({
            'bucket':          bucket,
            'n_trades':        len(group),
            'ev':              round(ev, 4),
            'profit_factor':   round(pf, 3),
            'win_rate':        round(wr, 3),
            'brier':           round(brier, 4) if brier is not None else None,
            'n_coins':         n_coins,
            'pct_coins':       pct_coins,
            'scorecard_stars': score,
            'is_core_edge':    score == 3,
            'is_negative':     ev < 0,
        })

    rows.sort(key=lambda r: (-r['scorecard_stars'], -r['pct_coins'], -r['ev']))
    return rows


def print_state_scorecard(rows: list[dict], field: str = 'state_id',
                           total_coins: int = 0):
    """Formatierte Terminal-Ausgabe der State Scorecard."""
    W = 80
    n = total_coins or max((r['n_coins'] for r in rows), default=1)

    print(f"\n{'='*W}")
    print(f"  STATE SCORECARD  (field={field}  |  {n} Coins  |  "
          f"{sum(r['n_trades'] for r in rows)} Trades gesamt)")
    print(f"{'='*W}")
    print(f"  {'State':<20} {'E[V]':>8} {'PF':>6} {'WR':>6} "
          f"{'Brier':>7} {'Invarianz':>12} {'Score'}")
    print(f"  {'-'*72}")

    for r in rows:
        brier_s = f"{r['brier']:.3f}" if r['brier'] is not None else '  n/a'
        stars_s = '*' * r['scorecard_stars'] if r['scorecard_stars'] > 0 else '-'
        inv_s   = f"{r['n_coins']}/{n} ({r['pct_coins']:.0f}%)"
        flag    = '  [CORE]' if r['is_core_edge'] else ('  [neg]' if r['is_negative'] else '')
        print(f"  {str(r['bucket']):<20} {r['ev']:>+8.4f} {r['profit_factor']:>6.2f} "
              f"{r['win_rate']:>5.0%} {brier_s:>7} {inv_s:>12}  {stars_s}{flag}")

    core_states = [r['bucket'] for r in rows if r['is_core_edge']]
    neg_states  = [r['bucket'] for r in rows if r['is_negative']]

    print(f"  {'-'*72}")
    print(f"\n  Core Edge States (***): "
          f"{core_states if core_states else 'keine'}")
    print(f"  Negative E[V] States:   "
          f"{neg_states if neg_states else 'keine'}")
    print(f"\n  Legende: E[V]>0 (+1pt)  |  Invarianz>=50% (+1pt)  "
          f"|  Brier<0.22 (+1pt)")
    print(f"{'='*W}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging, glob as _glob
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="statebot Trade Attribution + Stability")
    parser.add_argument('--file',     default=None,
                        help='Eine backtest_pnl_*.json Datei')
    parser.add_argument('--files',    nargs='+', default=None,
                        help='Mehrere Dateien (oder Glob-Pattern) fuer Stability Report')
    parser.add_argument('--field',    default=None,
                        help='Einzelnes Feld analysieren (Standard: alle)')
    parser.add_argument('--buckets',  type=int, default=4,
                        help='Anzahl numerische Buckets (Standard: 4)')
    parser.add_argument('--min-trades', type=int, default=3, dest='min_trades',
                        help='Minimum Trades pro Bucket fuer Ausgabe')
    parser.add_argument('--summary',  action='store_true',
                        help='Komponenten-Beitrag ausgeben')
    parser.add_argument('--report',   action='store_true',
                        help='Research Report (alle Dimensionen auf einmal)')
    parser.add_argument('--stability', action='store_true',
                        help='Stability Report ueber mehrere Coins (benoetigt --files)')
    parser.add_argument('--consensus', action='store_true',
                        help='Cross-Coin Invarianz: welche States tauchen in 80%%-Zone mehrerer Coins auf? (benoetigt --files)')
    parser.add_argument('--scorecard', action='store_true',
                        help='State Scorecard: Profitability x Calibration x Invarianz (benoetigt --files)')
    parser.add_argument('--save',     action='store_true',
                        help='Report als JSON speichern')
    args = parser.parse_args()

    # ── Stability Report (multi-file) ──────────────────────────────────────────
    if args.stability:
        paths = []
        for pattern in (args.files or []):
            paths.extend(_glob.glob(pattern))
        if args.file:
            paths.append(args.file)
        if not paths:
            print("--stability benoetigt --files <glob> oder --file.")
            sys.exit(1)

        coin_results: dict[str, list[dict]] = {}
        for p in sorted(set(paths)):
            try:
                with open(p) as f:
                    d = json.load(f)
                m   = d.get('market', p)
                tf  = d.get('timeframe', '?')
                lbl = f"{m} {tf}"
                coin_results[lbl] = d.get('trades', [])
            except Exception as e:
                print(f"  Uebersprungen {p}: {e}")

        stab = generate_stability_report(coin_results)
        print_stability_report(stab)
        if args.save:
            p_out = os.path.join(RESULTS_DIR, 'stability_report.json')
            with open(p_out, 'w') as f:
                json.dump({'generated_at': datetime.now(timezone.utc).isoformat(),
                           **stab}, f, indent=2, default=str)
            print(f"\n  Gespeichert: {p_out}")
        sys.exit(0)

    # ── Cross-Coin Invarianz ───────────────────────────────────────────────────
    if args.consensus or args.scorecard:
        paths = []
        for pattern in (args.files or []):
            paths.extend(_glob.glob(pattern))
        if args.file:
            paths.append(args.file)
        if not paths:
            print("--consensus / --scorecard benoetigt --files <glob> oder --file.")
            sys.exit(1)

        coin_results: dict[str, list[dict]] = {}
        for p in sorted(set(paths)):
            try:
                with open(p) as f:
                    d = json.load(f)
                m   = d.get('market', p)
                tf  = d.get('timeframe', '?')
                lbl = f"{m} {tf}"
                coin_results[lbl] = d.get('trades', [])
            except Exception as e:
                print(f"  Uebersprungen {p}: {e}")

        fld = args.field or 'state_id'

        if args.consensus:
            rows = cross_coin_pareto_consensus(coin_results, field=fld)
            print_cross_coin_consensus(rows, field=fld,
                                        total_coins=len(coin_results))

        if args.scorecard:
            rows = state_scorecard(coin_results, field=fld)
            print_state_scorecard(rows, field=fld,
                                   total_coins=len(coin_results))

        if args.save and args.scorecard:
            p_out = os.path.join(RESULTS_DIR, f'scorecard_{fld}.json')
            with open(p_out, 'w') as f:
                json.dump({'generated_at': datetime.now(timezone.utc).isoformat(),
                           'field': fld, 'rows': rows}, f, indent=2, default=str)
            print(f"\n  Gespeichert: {p_out}")

        sys.exit(0)

    # ── Einzel-Datei Analyse ───────────────────────────────────────────────────
    if not args.file:
        print("Bitte --file oder --stability angeben.")
        sys.exit(1)

    try:
        with open(args.file) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FEHLER: {e}")
        sys.exit(1)

    trades = data.get('trades', [])
    if not trades:
        print("Keine Trades in Datei.")
        sys.exit(1)

    market = data.get('market', '?')
    tf     = data.get('timeframe', '?')
    title  = f"{market} ({tf})  |  {len(trades)} Trades"

    if args.report:
        report = generate_research_report(trades, title=title)
        print_research_report(report)
        if args.save:
            safe = market.replace('/','').replace(':','')
            p_out = os.path.join(RESULTS_DIR, f"research_{safe}_{tf}.json")
            with open(p_out, 'w') as f:
                json.dump({'generated_at': datetime.now(timezone.utc).isoformat(),
                           'market': market, 'timeframe': tf,
                           'research_report': report}, f, indent=2, default=str)
            print(f"\n  Gespeichert: {p_out}")
    else:
        fields = [args.field] if args.field else None
        results = run_full_attribution(trades, fields, n_buckets=args.buckets)
        print_report(results, title=title)

        if args.summary:
            summary = component_summary(trades)
            print_component_summary(summary)

        if args.save:
            path = save_report(results, {'market': market, 'timeframe': tf,
                                          'source_file': args.file, 'n_trades': len(trades)})
            print(f"\n  Gespeichert: {path}")
