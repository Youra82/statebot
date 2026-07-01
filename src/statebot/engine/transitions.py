# engine/transitions.py — Markov-Übergangsmatrix
#
# Berechnet für jeden State:
#   P(state_t+1 | state_t) — Übergangswahrscheinlichkeiten
#
# Wichtig für die Vorhersage: nicht NUR "welche States folgten",
# sondern auch "wie oft war der nächste Close höher/niedriger?"

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def build_transition_matrix(store, market: str, tf: str) -> dict:
    """
    Baut die vollständige Markov-Übergangsmatrix aus dem Store.
    Gibt ein Dict zurück: {from_state: {to_state: probability, ...}}
    """
    rows = store.get_all_vectors(market, tf)

    # Nur Zeilen mit State-ID
    labeled = [(r['bar_time'], r['state_id'], r.get('next_close_pct'))
               for r in rows if r.get('state_id') is not None]

    if len(labeled) < 2:
        logger.warning(f"Zu wenige gelabelte States für Transitions: {market}/{tf}")
        return {}

    # Übergangs-Zähler
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # Up-Wahrscheinlichkeit per State
    up_counts: dict[int, list[float]] = defaultdict(list)

    for i in range(len(labeled) - 1):
        _, from_state, from_ret = labeled[i]
        _, to_state,   _        = labeled[i + 1]
        if from_state is not None and to_state is not None:
            counts[from_state][to_state] += 1
        if from_state is not None and from_ret is not None:
            up_counts[from_state].append(1.0 if from_ret > 0 else 0.0)

    # Wahrscheinlichkeiten normieren
    store.delete_transitions(market, tf)
    transition_dict: dict[int, dict] = {}

    for from_state, to_counts in counts.items():
        total    = sum(to_counts.values())
        probs    = {to: cnt / total for to, cnt in to_counts.items()}
        up_p     = float(np.mean(up_counts[from_state])) if up_counts[from_state] else 0.5
        n_samp   = len(up_counts[from_state])

        transition_dict[from_state] = {
            'up_prob':     up_p,
            'n_samples':   n_samp,
            'transitions': probs,
        }

        for to_state, prob in probs.items():
            store.upsert_transition(market, tf, from_state, to_state,
                                    to_counts[to_state], prob)

    store.commit()
    logger.info(f"Übergangsmatrix: {len(transition_dict)} States, "
                f"{sum(sum(v.values()) for v in counts.values())} Übergänge")
    return transition_dict


def get_top_transitions(store, market: str, tf: str,
                         from_state: int, top_n: int = 3) -> list[dict]:
    """Gibt die wahrscheinlichsten Nachfolge-States zurück."""
    transitions = store.get_transitions_from(market, tf, from_state)
    result      = []

    for t in transitions[:top_n]:
        to_def = store.get_state_definition(market, tf, t['to_state'])
        name   = to_def['name'] if to_def else f"State {t['to_state']}"
        result.append({
            'to_state':    t['to_state'],
            'name':        name,
            'probability': t['probability'],
        })

    return result


def state_up_probability(store, market: str, tf: str, state_id: int) -> float:
    """P(next_close > current_close | current_state) aus state_definitions."""
    state_def = store.get_state_definition(market, tf, state_id)
    if state_def and state_def.get('up_prob') is not None:
        return float(state_def['up_prob'])
    return 0.5


def build_transition_matrix_order2(store, market: str, tf: str) -> int:
    """
    Markov-Kette 2. Ordnung: P(S_t+1 | S_t, S_t-1)

    In Märkten besitzen Regime eine gewisse Trägheit —
    der vorherige Zustand enthält oft zusätzliche Information
    darüber, ob der aktuelle Zustand stabil ist oder kurz vor
    einem Regime-Wechsel steht.

    Speichert Triplets (S_t-1, S_t, S_t+1) in transitions_order2.
    Gibt Anzahl gespeicherter Triplets zurück.
    """
    rows = store.get_all_vectors(market, tf)
    labeled = [(r['bar_time'], r['state_id'], r.get('next_close_pct'))
               for r in rows if r.get('state_id') is not None]

    if len(labeled) < 3:
        logger.warning(f"Zu wenige States für Order-2 Transitions: {market}/{tf}")
        return 0

    # Triplet-Zähler: (s1, s2) → {s3: count}
    counts: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    up_outcomes: dict[tuple, list]      = defaultdict(list)

    for i in range(len(labeled) - 2):
        _, s1, ret1 = labeled[i]
        _, s2, _    = labeled[i + 1]
        _, s3, _    = labeled[i + 2]
        if s1 is None or s2 is None or s3 is None:
            continue
        counts[(s1, s2)][s3] += 1
        if ret1 is not None:
            up_outcomes[(s1, s2)].append(1.0 if ret1 > 0 else 0.0)

    n_triplets = 0
    for (s1, s2), to_counts in counts.items():
        total   = sum(to_counts.values())
        up_p    = float(np.mean(up_outcomes[(s1, s2)])) if up_outcomes[(s1, s2)] else 0.5
        for s3, cnt in to_counts.items():
            store.upsert_transition_order2(market, tf, s1, s2, s3,
                                            cnt, cnt / total, up_p)
            n_triplets += 1

    store.commit()
    logger.info(f"Order-2 Matrix: {len(counts)} (s1,s2)-Paare → {n_triplets} Triplets | {market}/{tf}")
    return n_triplets


def state_up_probability_order2(store, market: str, tf: str,
                                  prev_state: int, curr_state: int,
                                  min_samples: int = 10) -> float | None:
    """
    P(up | prev_state, curr_state) aus der Order-2 Übergangsmatrix.

    Gibt None zurück wenn zu wenig Daten (dann Fallback auf Order-1).
    min_samples: Mindestanzahl beobachteter Triplets für dieses Paar.
    """
    transitions = store.get_transition_order2(market, tf, prev_state, curr_state)
    if not transitions:
        return None
    total_count = sum(t['count'] for t in transitions)
    if total_count < min_samples:
        return None   # Zu wenige Daten → kein Vertrauen
    # Gewichtetes Mittel der up_prob über alle möglichen Nachfolge-States
    total_weight = sum(t['count'] for t in transitions)
    p_up = sum(t['up_prob'] * t['count'] for t in transitions) / total_weight
    return float(p_up)
