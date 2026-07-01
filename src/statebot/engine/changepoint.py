# engine/changepoint.py — Change Point Detection
#
# Unterschied zu Calibration Drift:
#   Drift:       Modell-Qualität sinkt langsam → Recluster empfohlen
#   Change Point: Markt hat Struktur gewechselt → Recluster dringend
#
# CUSUM (Cumulative Sum):
#   Akkumuliert Abweichungen vom Zielwert.
#   Alarm wenn S+ oder S- den Schwellenwert H überschreiten.
#   Mathematisch äquivalent zu Sequential Probability Ratio Test (SPRT).
#
# ADWIN (Adaptive Windowing) — vereinfacht:
#   Vergleicht statistisch, ob ältere und neuere Sub-Fenster
#   unterschiedliche Mittelwerte haben.
#   Wenn ja: Fenster wird verkleinert (ältere Daten verworfen).
#
# Persistenz: CUSUM-Zustand (S+, S-, N) wird in maintenance_log.json
# gespeichert, damit der Detektor über Cron-Runs hinweg kontinuierlich arbeitet.

import json
import logging
import math
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


# ─── CUSUM ────────────────────────────────────────────────────────────────────

class CUSUMDetector:
    """
    Page-Hinkley CUSUM Change Point Detector.

    Optimal für: langsame Mittelwertverschiebungen (Regime Drift).

    Parameter:
        target  — erwarteter Zielwert (z.B. 0.20 = guter Brier Score)
        k       — erlaubte Drift vor Akkumulation (Slack-Parameter)
        h       — Alarm-Schwelle (höher = weniger sensitiv, weniger Fehlalarme)

    Typische Werte für Brier Score Überwachung:
        target=0.22, k=0.01, h=5.0
    """

    def __init__(self, target: float = 0.22, k: float = 0.01, h: float = 5.0):
        self.target = target
        self.k      = k
        self.h      = h
        self._s_pos = 0.0   # Aufwärts-Akkumulator (Verschlechterung)
        self._s_neg = 0.0   # Abwärts-Akkumulator (unerwartete Verbesserung)
        self._n     = 0

    def update(self, observation: float) -> bool:
        """
        Verarbeitet eine neue Beobachtung.
        Gibt True zurück wenn Change Point detektiert.
        """
        self._n    += 1
        self._s_pos = max(0.0, self._s_pos + (observation - self.target) - self.k)
        self._s_neg = max(0.0, self._s_neg - (observation - self.target) - self.k)
        return (self._s_pos >= self.h) or (self._s_neg >= self.h)

    def reset(self):
        """Akkumulatoren zurücksetzen (nach erkanntem Change Point)."""
        self._s_pos = 0.0
        self._s_neg = 0.0
        logger.info(f"CUSUM reset nach n={self._n} Beobachtungen")

    @property
    def s_pos(self) -> float:
        return self._s_pos

    @property
    def s_neg(self) -> float:
        return self._s_neg

    @property
    def n_observations(self) -> int:
        return self._n

    def state_dict(self) -> dict:
        """Zustand für Persistenz in maintenance_log.json."""
        return {'s_pos': self._s_pos, 's_neg': self._s_neg, 'n': self._n,
                'target': self.target, 'k': self.k, 'h': self.h}

    def load_state(self, d: dict):
        """Zustand aus maintenance_log.json wiederherstellen."""
        self._s_pos = float(d.get('s_pos', 0.0))
        self._s_neg = float(d.get('s_neg', 0.0))
        self._n     = int(d.get('n', 0))

    def status_line(self) -> str:
        pct_pos = min(100, self._s_pos / self.h * 100)
        pct_neg = min(100, self._s_neg / self.h * 100)
        bar_p   = '█' * int(pct_pos / 10) + '░' * (10 - int(pct_pos / 10))
        bar_n   = '█' * int(pct_neg / 10) + '░' * (10 - int(pct_neg / 10))
        return (f"S+={self._s_pos:.2f}/{self.h:.1f}  [{bar_p}]  "
                f"S-={self._s_neg:.2f}/{self.h:.1f}  [{bar_n}]  n={self._n}")


# ─── ADWIN ────────────────────────────────────────────────────────────────────

class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) — vereinfachte Implementierung.

    Optimal für: abrupte Strukturbrüche (Regime Shifts).
    Auslöser: wenn Mittelwert-Differenz zwischen zwei Fensterhaelften
    statistisch signifikant ist.

    Parameter:
        delta     — Konfidenz-Schwelle (kleiner = sensitiver)
        max_size  — maximale Fenstergröße
        min_size  — Mindest-Fenstergröße für Test
    """

    def __init__(self, delta: float = 0.05, max_size: int = 500, min_size: int = 20):
        self.delta    = delta
        self.max_size = max_size
        self.min_size = min_size
        self._window  = deque(maxlen=max_size)

    def update(self, observation: float) -> bool:
        """Gibt True zurück wenn Strukturbruch erkannt."""
        self._window.append(float(observation))
        if len(self._window) < self.min_size:
            return False
        return self._test_for_change()

    def _test_for_change(self) -> bool:
        w   = list(self._window)
        n   = len(w)
        mu  = np.mean(w)
        # Teste alle Teilungspunkte
        for t in range(self.min_size // 2, n - self.min_size // 2):
            w0     = w[:t]
            w1     = w[t:]
            mu0    = np.mean(w0)
            mu1    = np.mean(w1)
            n0, n1 = len(w0), len(w1)
            # Hoeffding-Schranke für den Mittelwertunterschied
            eps_cut = math.sqrt((1.0 / (2 * n0) + 1.0 / (2 * n1)) *
                                math.log(4 * n * n / self.delta))
            if abs(mu0 - mu1) >= eps_cut:
                # Verwerfe alten Teil des Fensters
                for _ in range(t):
                    self._window.popleft()
                return True
        return False

    @property
    def window_size(self) -> int:
        return len(self._window)

    @property
    def mean(self) -> float:
        return float(np.mean(self._window)) if self._window else 0.5


# ─── Manager: beide Detektoren kombiniert ─────────────────────────────────────

class ChangepointManager:
    """
    Kombiniert CUSUM (langsamer Drift) + ADWIN (abrupte Brüche).
    Persistiert Zustand in maintenance_log.json für Kontinuität über Cron-Runs.
    """

    def __init__(self, name: str, cusum_params: dict | None = None):
        self.name    = name
        cfg          = cusum_params or {}
        self.cusum   = CUSUMDetector(
            target=cfg.get('target', 0.22),
            k=cfg.get('k', 0.01),
            h=cfg.get('h', 5.0),
        )
        self.adwin   = ADWINDetector(delta=cfg.get('adwin_delta', 0.05))
        self._alarms: list[dict] = []

    def update(self, brier_score: float, timestamp: str = "") -> dict:
        """
        Verarbeitet einen neuen Brier Score.
        Gibt Alarm-Dict zurück (leer wenn kein Alarm).
        """
        cusum_alarm = self.cusum.update(brier_score)
        adwin_alarm = self.adwin.update(brier_score)

        result = {
            'name':        self.name,
            'brier':       brier_score,
            'cusum_alarm': cusum_alarm,
            'adwin_alarm': adwin_alarm,
            'any_alarm':   cusum_alarm or adwin_alarm,
            'cusum_s_pos': self.cusum.s_pos,
            'cusum_s_neg': self.cusum.s_neg,
            'adwin_window':self.adwin.window_size,
            'adwin_mean':  self.adwin.mean,
            'timestamp':   timestamp,
        }

        if cusum_alarm:
            logger.warning(f"CUSUM Alarm [{self.name}]: Drift akkumuliert  "
                           f"{self.cusum.status_line()}")
            self.cusum.reset()   # nach Alarm zurücksetzen

        if adwin_alarm:
            logger.warning(f"ADWIN Alarm [{self.name}]: Strukturbruch erkannt  "
                           f"window={self.adwin.window_size}")

        if result['any_alarm']:
            self._alarms.append(result)

        return result

    def persist_state(self, log: dict):
        """Zustand in maintenance_log-Dict schreiben."""
        log[f'cusum_{self.name}'] = self.cusum.state_dict()

    def restore_state(self, log: dict):
        """Zustand aus maintenance_log-Dict laden."""
        key = f'cusum_{self.name}'
        if key in log:
            self.cusum.load_state(log[key])
            logger.debug(f"CUSUM [{self.name}] wiederhergestellt: {self.cusum.status_line()}")
