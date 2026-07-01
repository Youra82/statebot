# engine/signal_pipeline.py — Modulare probabilistische Signal-Pipeline
#
# Klare Rollentrennung:
#
#   Markov  → PRIOR      P(up | State)            fuse_prior_likelihood()
#   KNN     → LIKELIHOOD P(obs | up) via k-NN      fuse_prior_likelihood()
#   Extern  → EVIDENCE   Funding, OI, Macro, ...   SignalPipeline.fuse()
#
#   Struktur:
#       p_prior  (Markov)   ─┐
#                             ├─ Bayes → p_posterior → SignalPipeline → p_final
#       p_knn    (Likelihood) ─┘         (Evidenz-Pipeline, nur externe Signale)
#
# Markov und KNN sind KEINE gleichartigen Stimmen — sie haben unterschiedliche
# probabilistische Rollen. Externe Signale dagegen sind additionale Evidenz
# auf dem bereits fusionierten Posterior.
#
# Korrelationsproblem (funding, OI, liquidations oft korreliert):
#   → Wird durch Reliability-Gewichte gedämpft (Soft-Mixing statt hartes Bayes)

import logging
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ─── Reliability-gewichtetes Bayes-Update ─────────────────────────────────────

def reliability_weighted_update(prior: float,
                                 evidence: float,
                                 reliability: float) -> float:
    """
    Kombiniert Prior und Evidence in Abhängigkeit der Signal-Zuverlässigkeit.

    reliability=1.0 → vollständiges Bayes-Update (log-odds)
    reliability=0.0 → Prior bleibt unverändert
    reliability=0.5 → halbierter Einfluss

    Soft-Mixing statt hartem Bayes:
        posterior = (1 - r) * prior + r * bayesian_update(prior, evidence)

    Dadurch können korrelierte Signale (OI + Liquidations) nicht
    dieselbe Information doppelt einrechnen — bei r=0.65 statt r=1.0
    beträgt ihr kombinierter Einfluss maximal ~1.3× statt 2×.
    """
    eps = 1e-7
    prior    = float(np.clip(prior,    eps, 1 - eps))
    evidence = float(np.clip(evidence, eps, 1 - eps))
    reliability = float(np.clip(reliability, 0.0, 1.0))

    # Vollständiges Bayes im Log-Odds-Raum
    lo_prior    = np.log(prior    / (1 - prior))
    lo_evidence = np.log(evidence / (1 - evidence))
    lo_full     = lo_prior + lo_evidence            # Bayes-Theorem in log-odds
    p_full      = 1.0 / (1.0 + np.exp(-lo_full))

    # Gewichtetes Mixing: reliability bestimmt, wie stark das Update wirkt
    return float((1.0 - reliability) * prior + reliability * p_full)


# ─── Markov × KNN Bayes-Fusion (Prior × Likelihood) ─────────────────────────

def fuse_prior_likelihood(p_prior: float, p_knn: float,
                           prior_reliability: float = 0.85,
                           knn_reliability: float = 0.90) -> float:
    """
    Proper Bayesian: Prior (Markov) × Likelihood (KNN)

    Markov ist der PRIOR: P(up | current_state)
    KNN ist die LIKELIHOOD: P(obs | up) ausgedrueckt als Log-Likelihood-Ratio

    logit(posterior) = prior_rel × logit(p_prior) + knn_rel × log_LR(p_knn)

    Reliability-Faktoren daempfen den jeweiligen Einfluss:
      prior_rel=0.85 → Markov-Modell leicht gedaempft (kann veraltet sein)
      knn_rel=0.90   → KNN hochvertrauen, aber nicht blind

    Unterschied zu naivem Bayes:
      Naiv: log_odds = logit(prior) + logit(knn)  → unkalibriert
      Hier: log_odds = r_p × logit(prior) + r_k × log_LR(knn)  → skaliert
    """
    eps = 1e-7
    p_prior = float(np.clip(p_prior, eps, 1 - eps))
    p_knn   = float(np.clip(p_knn,   eps, 1 - eps))
    prior_reliability = float(np.clip(prior_reliability, 0.0, 1.0))
    knn_reliability   = float(np.clip(knn_reliability,   0.0, 1.0))

    lo_prior  = np.log(p_prior / (1 - p_prior))       # logit(prior)
    log_LR    = np.log(p_knn   / (1 - p_knn))         # Log-Likelihood-Ratio aus KNN
    lo_post   = prior_reliability * lo_prior + knn_reliability * log_LR
    return float(1.0 / (1.0 + np.exp(-lo_post)))


# ─── Basis-Klasse ─────────────────────────────────────────────────────────────

class EvidenceSignal(ABC):
    """
    Basisklasse für alle Signal-Quellen.
    Jedes Signal gibt P(up) zurück oder None wenn nicht verfügbar.
    """
    name:               str   = "unnamed"
    static_reliability: float = 0.70   # Fallback wenn keine Kalibrierungsdaten

    @abstractmethod
    def compute(self, context: dict) -> float | None:
        """Gibt P(next_close > current_close) zurück, oder None."""
        ...

    def is_available(self, context: dict) -> bool:
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(reliability={self.static_reliability:.2f})"


# ─── Externe Evidenz-Signale ──────────────────────────────────────────────────
# Hinweis: Markov und KNN sind KEINE Pipeline-Signale mehr.
# Sie werden in fuse_prior_likelihood() als Prior × Likelihood fusioniert.
# Die Pipeline hier verarbeitet nur externe Evidenz-Quellen.
# (können später befüllt werden ohne Kernarchitektur zu ändern)

class FundingRateSignal(EvidenceSignal):
    """
    Funding Rate → Signal-Interpretation:
      stark negativ (Shorts zahlen Longs) → bullisch
      stark positiv (Longs zahlen Shorts) → bärisch
    Noch nicht implementiert — gibt None zurück.
    """
    name               = "funding"
    static_reliability = 0.65

    def compute(self, context: dict) -> float | None:
        funding = context.get('funding_rate')
        if funding is None:
            return None
        # Normierung: funding ∈ [-0.01, +0.01] typical range
        # negativ → P(up) > 0.5, positiv → P(up) < 0.5
        p_up = 0.5 - float(funding) * 20.0     # Skalierung anpassbar
        return float(np.clip(p_up, 0.05, 0.95))

    def is_available(self, context: dict) -> bool:
        return context.get('funding_rate') is not None


class OpenInterestSignal(EvidenceSignal):
    """
    OI-Trend als Bestätigungssignal.
    OI steigt + Preis steigt → Long-Bestätigung
    OI steigt + Preis fällt → Short-Bestätigung
    Noch nicht implementiert.
    """
    name               = "oi"
    static_reliability = 0.75

    def compute(self, context: dict) -> float | None:
        oi_change = context.get('oi_pct_change')
        if oi_change is None:
            return None
        p_knn = context.get('p_knn', 0.5)
        # OI steigend + bärisches Signal → stärkt Short
        if oi_change > 0.02 and p_knn < 0.45:
            return max(0.1, p_knn - 0.05)
        if oi_change > 0.02 and p_knn > 0.55:
            return min(0.9, p_knn + 0.05)
        return p_knn   # Neutral

    def is_available(self, context: dict) -> bool:
        return context.get('oi_pct_change') is not None


class FearGreedSignal(EvidenceSignal):
    """
    Fear & Greed Index (0 = extreme fear, 100 = extreme greed).
    Contrarian: extreme fear → bullisch, extreme greed → bärisch
    """
    name               = "fear_greed"
    static_reliability = 0.40   # Schwaches Signal

    def compute(self, context: dict) -> float | None:
        fg = context.get('fear_greed_index')
        if fg is None:
            return None
        # Contrarian: 0 → p_up=0.70, 50 → p_up=0.50, 100 → p_up=0.30
        p_up = 0.70 - float(fg) / 100.0 * 0.40
        return float(np.clip(p_up, 0.15, 0.85))

    def is_available(self, context: dict) -> bool:
        return context.get('fear_greed_index') is not None


# ─── Signal-Pipeline ──────────────────────────────────────────────────────────

class SignalPipeline:
    """
    Modulare Bayes-Pipeline:
      posterior = 0.5 (uniform)
      for signal in pipeline:
          posterior = reliability_weighted_update(posterior, signal.compute(), reliability)
    """

    def __init__(self, signals: list[EvidenceSignal],
                 calibrator=None,
                 market: str = "",
                 timeframe: str = ""):
        self.signals   = signals
        self.calibrator = calibrator
        self.market    = market
        self.timeframe = timeframe

    @classmethod
    def default(cls, calibrator=None, market="", timeframe="",
                signal_config: list[dict] | None = None) -> 'SignalPipeline':
        """
        Standard-Pipeline mit KNN + Markov.
        signal_config ermöglicht Konfiguration aus settings.json.
        """
        all_signal_classes = {
            # Markov und KNN sind NICHT hier — sie haben eigene Fusion via fuse_prior_likelihood()
            'funding':    FundingRateSignal,
            'oi':         OpenInterestSignal,
            'fear_greed': FearGreedSignal,
        }

        if signal_config:
            signals = []
            for cfg in signal_config:
                name    = cfg.get('name')
                enabled = cfg.get('enabled', True)
                if not enabled or name not in all_signal_classes:
                    continue
                sig = all_signal_classes[name]()
                sig.static_reliability = cfg.get('reliability', sig.static_reliability)
                signals.append(sig)
        else:
            # Minimum: Markov + KNN
            signals = [MarkovSignal(), KNNSignal()]

        return cls(signals, calibrator=calibrator, market=market, timeframe=timeframe)

    def fuse(self, context: dict, start_probability: float = 0.5) -> dict:
        """
        Führt alle Signale nacheinander aus und fusioniert sie.

        Returns:
            p_final      — finale Wahrscheinlichkeit
            signal_trace — Verlauf (jedes Signal mit vor/nach)
            n_active     — Anzahl tatsächlich aktiver Signale
        """
        posterior = float(np.clip(start_probability, 1e-6, 1 - 1e-6))
        trace     = []

        for signal in self.signals:
            if not signal.is_available(context):
                trace.append({'name': signal.name, 'status': 'not_available'})
                continue

            p_evidence = signal.compute(context)
            if p_evidence is None:
                trace.append({'name': signal.name, 'status': 'no_signal'})
                continue

            # Reliability aus Calibrator (falls vorhanden), sonst statisch
            if self.calibrator and self.market and self.timeframe:
                reliability = self.calibrator.get_reliability(
                    signal.name, self.market, self.timeframe,
                    static_fallback=signal.static_reliability,
                )
            else:
                reliability = signal.static_reliability

            prior       = posterior
            posterior   = reliability_weighted_update(prior, p_evidence, reliability)

            trace.append({
                'name':        signal.name,
                'p_signal':    p_evidence,
                'reliability': reliability,
                'prior':       prior,
                'posterior':   posterior,
                'delta':       posterior - prior,
                'status':      'applied',
            })

            # Kalibrierungs-Vorhersage aufzeichnen (wenn Calibrator vorhanden)
            if self.calibrator and context.get('prediction_id') and self.market:
                self.calibrator.record_prediction(
                    signal.name, self.market, self.timeframe,
                    context['prediction_id'], p_evidence,
                )

        n_active = sum(1 for t in trace if t.get('status') == 'applied')
        return {
            'p_final':     posterior,
            'signal_trace': trace,
            'n_active':    n_active,
        }

    def __repr__(self):
        return f"SignalPipeline([{', '.join(s.name for s in self.signals)}])"
