# engine/calibrator.py — Selbst-kalibrierendes Signal-Tracking
#
# Jeder Signal-Geber führt Statistik über sich selbst.
# Brier Score = mean((predicted_p - actual_outcome)²)
# Reliability = max(0, 1 - 4 * brier_score)
#   perfect (BS=0.0) → 1.0  |  random (BS=0.25) → 0.0
#
# Das Gewicht wird dadurch dynamisch — der Bot lernt,
# welchem Signal er im aktuellen Umfeld vertrauen kann.

import sqlite3
import logging
from datetime import datetime, timezone, timedelta

import numpy as np

logger = logging.getLogger(__name__)


class SignalCalibrator:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS signal_calibration (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name  TEXT    NOT NULL,
                market       TEXT    NOT NULL,
                timeframe    TEXT    NOT NULL,
                bar_time     TEXT    NOT NULL,
                predicted_p  REAL    NOT NULL,
                actual_outcome INTEGER,        -- 1 = up, 0 = down, NULL = noch offen
                recorded_at  TEXT    DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(signal_name, market, timeframe, bar_time)
            );
            CREATE INDEX IF NOT EXISTS idx_cal_lookup
                ON signal_calibration(signal_name, market, timeframe, recorded_at);
        """)
        self.conn.commit()

    # ── Schreiben ──────────────────────────────────────────────────────────────

    def record_prediction(self, signal_name: str, market: str, timeframe: str,
                          bar_time: str, predicted_p: float):
        """Speichert eine Vorhersage (Outcome = NULL bis Trade schliesst)."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO signal_calibration
                (signal_name, market, timeframe, bar_time, predicted_p, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_name, market, timeframe, bar_time) DO UPDATE SET
                predicted_p = excluded.predicted_p
        """, (signal_name, market, timeframe, bar_time, predicted_p, now))
        self.conn.commit()

    def record_outcome(self, market: str, timeframe: str,
                       bar_time: str, went_up: bool):
        """Setzt den tatsaechlichen Ausgang fuer alle Signale dieses bar_time."""
        outcome = 1 if went_up else 0
        self.conn.execute("""
            UPDATE signal_calibration SET actual_outcome = ?
            WHERE market = ? AND timeframe = ? AND bar_time = ?
              AND actual_outcome IS NULL
        """, (outcome, market, timeframe, bar_time))
        self.conn.commit()

    # ── Lesen ──────────────────────────────────────────────────────────────────

    def get_brier_score(self, signal_name: str, market: str, timeframe: str,
                         lookback_days: int = 30) -> float | None:
        """Brier Score der letzten N Tage (None wenn zu wenig Daten)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        rows = self.conn.execute("""
            SELECT predicted_p, actual_outcome FROM signal_calibration
            WHERE signal_name = ? AND market = ? AND timeframe = ?
              AND actual_outcome IS NOT NULL
              AND recorded_at >= ?
            ORDER BY recorded_at DESC
        """, (signal_name, market, timeframe, cutoff)).fetchall()

        if len(rows) < 5:
            return None

        errors = [(float(r['predicted_p']) - float(r['actual_outcome'])) ** 2 for r in rows]
        return float(np.mean(errors))

    def _get_calibration_score(self, signal_name: str, market: str, timeframe: str) -> float | None:
        """Kalibrierung: normierter Brier Score (0=perfekt, 1=schlecht)."""
        bs_30  = self.get_brier_score(signal_name, market, timeframe, lookback_days=30)
        bs_180 = self.get_brier_score(signal_name, market, timeframe, lookback_days=180)
        if bs_30 is None and bs_180 is None:
            return None
        if bs_30 is None:
            bs = bs_180
        elif bs_180 is None:
            bs = bs_30
        else:
            bs = 0.70 * bs_30 + 0.30 * bs_180   # Recent-Bias
        return max(0.0, 1.0 - 4.0 * bs)          # [0,1], 1=perfekt

    def _get_stability_score(self, signal_name: str, market: str, timeframe: str) -> float | None:
        """
        Stabilitaet: Wie konsistent ist der Brier Score ueber Zeit?
        4 aufeinanderfolgende 7-Tage-Fenster der letzten 28 Tage.
        Niedrige Varianz → hohes Stabilitaets-Score.
        """
        weekly_scores = []
        for week_offset in range(4):
            days_end   = week_offset * 7
            days_start = days_end + 7
            cutoff_end   = (datetime.now(timezone.utc) - timedelta(days=days_end)).isoformat()
            cutoff_start = (datetime.now(timezone.utc) - timedelta(days=days_start)).isoformat()
            rows = self.conn.execute("""
                SELECT predicted_p, actual_outcome FROM signal_calibration
                WHERE signal_name = ? AND market = ? AND timeframe = ?
                  AND actual_outcome IS NOT NULL
                  AND recorded_at >= ? AND recorded_at < ?
            """, (signal_name, market, timeframe, cutoff_start, cutoff_end)).fetchall()
            if len(rows) >= 3:
                bs = float(np.mean([(float(r['predicted_p']) - float(r['actual_outcome']))**2
                                     for r in rows]))
                weekly_scores.append(bs)
        if len(weekly_scores) < 2:
            return None
        variance = float(np.var(weekly_scores))
        # Normierung: Var=0 → 1.0 (perfekt stabil), Var=0.0625 (max) → 0.0
        return max(0.0, 1.0 - variance / 0.0625)

    def _get_sample_size_score(self, signal_name: str, market: str, timeframe: str) -> float:
        """
        Vertrauen waechst mit der Anzahl bewerteter Vorhersagen.
        n=0 → 0.0 | n=50 → 0.5 | n=100+ → 1.0
        """
        row = self.conn.execute("""
            SELECT COUNT(*) as n FROM signal_calibration
            WHERE signal_name = ? AND market = ? AND timeframe = ?
              AND actual_outcome IS NOT NULL
        """, (signal_name, market, timeframe)).fetchone()
        n = int(row['n']) if row else 0
        return float(min(1.0, n / 100.0))

    def get_reliability(self, signal_name: str, market: str, timeframe: str,
                         static_fallback: float | None = 0.7) -> float | None:
        """
        Reliability aus drei Kriterien:
            Calibration × 0.50   (Brier Score, 30d/180d gewichtet)
            Stability   × 0.30   (Konsistenz ueber Zeit)
            SampleSize  × 0.20   (Vertrauen steigt mit Datenbasis)

        15 Trades != 2000 Trades — SampleSize verhindert uebertriebene
        Zuverlaessigkeit bei kleinen Stichproben.
        """
        calibration = self._get_calibration_score(signal_name, market, timeframe)
        stability   = self._get_stability_score(signal_name, market, timeframe)
        sample_size = self._get_sample_size_score(signal_name, market, timeframe)

        if calibration is None and stability is None:
            if static_fallback is None:
                return None          # Explizit: Aufrufer weiß, dass keine Daten vorhanden
            return static_fallback

        cal = calibration if calibration is not None else static_fallback
        sta = stability   if stability   is not None else 0.5   # Neutral wenn unbekannt
        smp = sample_size

        return float(0.50 * cal + 0.30 * sta + 0.20 * smp)

    def get_calibration_curve(self, signal_name: str, market: str, timeframe: str,
                               n_bins: int = 10) -> list[dict]:
        """
        Kalibrierungskurve: Bucketed Predicted vs. Actual.

        Ideal: Bot sagt 60% → tatsächlich 60%.
        Überoptimistisch: Bot sagt 90% → tatsächlich 58% (zu extreme Vorhersagen)
        Zu konservativ: Bot sagt 60% → tatsächlich 75%.

        Returns Liste von Buckets:
            bin_low, bin_high, predicted_avg, actual_rate, n_samples
        """
        rows = self.conn.execute("""
            SELECT predicted_p, actual_outcome FROM signal_calibration
            WHERE signal_name = ? AND market = ? AND timeframe = ?
              AND actual_outcome IS NOT NULL
            ORDER BY predicted_p ASC
        """, (signal_name, market, timeframe)).fetchall()

        if len(rows) < 10:
            return []

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        preds     = np.array([float(r['predicted_p'])   for r in rows])
        outcomes  = np.array([float(r['actual_outcome']) for r in rows])

        buckets = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask   = (preds >= lo) & (preds < hi)
            if i == n_bins - 1:
                mask = (preds >= lo) & (preds <= hi)
            n = int(mask.sum())
            if n == 0:
                continue
            buckets.append({
                'bin_low':       float(lo),
                'bin_high':      float(hi),
                'bin_mid':       float((lo + hi) / 2),
                'predicted_avg': float(preds[mask].mean()),
                'actual_rate':   float(outcomes[mask].mean()),
                'n_samples':     n,
            })

        return buckets

    def get_signal_stats(self, market: str, timeframe: str) -> list[dict]:
        """Alle Signal-Statistiken fuer den Show-Results-Report."""
        rows = self.conn.execute("""
            SELECT
                signal_name,
                COUNT(*) as total,
                SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) as evaluated,
                AVG(CASE WHEN actual_outcome IS NOT NULL
                    THEN (predicted_p - actual_outcome) * (predicted_p - actual_outcome)
                    ELSE NULL END) as brier_score,
                AVG(CASE WHEN actual_outcome = 1 AND predicted_p >= 0.5 THEN 1.0
                         WHEN actual_outcome = 0 AND predicted_p <  0.5 THEN 1.0
                         WHEN actual_outcome IS NOT NULL THEN 0.0
                         ELSE NULL END) as accuracy
            FROM signal_calibration
            WHERE market = ? AND timeframe = ?
            GROUP BY signal_name
            ORDER BY brier_score ASC
        """, (market, timeframe)).fetchall()
        return [dict(r) for r in rows]
