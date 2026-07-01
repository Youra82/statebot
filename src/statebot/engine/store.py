# engine/store.py — SQLite Datenbank für statebot
#
# Tabellen:
#   feature_vectors   — 22 Features + State-Zuweisung + Multi-Target Labels
#   state_definitions — Cluster-Zentroide + Statistiken
#   transitions       — Markov-Übergangsmatrix
#   scan_log          — Letzter Scan pro Market/Timeframe

import sqlite3
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

from statebot.engine.features import FEATURE_COLS

logger = logging.getLogger(__name__)

# SQL-Spalten für die 22 Features
_FEAT_COLS_SQL = ', '.join(f'{c} REAL' for c in FEATURE_COLS)
_FEAT_SELECT   = ', '.join(FEATURE_COLS)
_FEAT_PLACEHOLDERS = ', '.join(['?'] * len(FEATURE_COLS))
_FEAT_UPDATE   = ', '.join(f'{c} = excluded.{c}' for c in FEATURE_COLS)


class StateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        # Calibration-Tabelle via separate Connection-Methode
        from statebot.engine.calibrator import SignalCalibrator
        self._calibrator = SignalCalibrator(self.conn)
        # Migration: quality_score-Spalte nachrüsten wenn DB bereits existiert
        try:
            self.conn.execute("ALTER TABLE state_definitions ADD COLUMN quality_score REAL DEFAULT 0.5")
            self.conn.commit()
        except Exception:
            pass  # Spalte existiert bereits
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS feature_vectors (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                market          TEXT    NOT NULL,
                timeframe       TEXT    NOT NULL,
                bar_time        TEXT    NOT NULL,
                {_FEAT_COLS_SQL},
                state_id        INTEGER,
                next_close_pct  REAL,
                next_high_pct   REAL,
                next_low_pct    REAL,
                created_at      TEXT    DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(market, timeframe, bar_time)
            );

            CREATE INDEX IF NOT EXISTS idx_fv_market_tf
                ON feature_vectors(market, timeframe);
            CREATE INDEX IF NOT EXISTS idx_fv_state
                ON feature_vectors(market, timeframe, state_id);

            CREATE TABLE IF NOT EXISTS state_definitions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                market          TEXT    NOT NULL,
                timeframe       TEXT    NOT NULL,
                state_id        INTEGER NOT NULL,
                name            TEXT    DEFAULT 'UNKNOWN',
                centroid_json   TEXT,
                n_samples       INTEGER DEFAULT 0,
                avg_return      REAL,
                std_return      REAL,
                up_prob         REAL,
                quality_score   REAL    DEFAULT 0.5,
                UNIQUE(market, timeframe, state_id)
            );

            CREATE TABLE IF NOT EXISTS transitions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market      TEXT    NOT NULL,
                timeframe   TEXT    NOT NULL,
                from_state  INTEGER NOT NULL,
                to_state    INTEGER NOT NULL,
                count       INTEGER DEFAULT 0,
                probability REAL    DEFAULT 0.0,
                UNIQUE(market, timeframe, from_state, to_state)
            );

            CREATE TABLE IF NOT EXISTS transitions_order2 (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                market       TEXT    NOT NULL,
                timeframe    TEXT    NOT NULL,
                state_1      INTEGER NOT NULL,   -- vorletzter State
                state_2      INTEGER NOT NULL,   -- aktueller State
                to_state     INTEGER NOT NULL,   -- nächster State
                count        INTEGER DEFAULT 0,
                probability  REAL    DEFAULT 0.0,
                up_prob      REAL    DEFAULT 0.5,
                UNIQUE(market, timeframe, state_1, state_2, to_state)
            );
            CREATE INDEX IF NOT EXISTS idx_tr2_lookup
                ON transitions_order2(market, timeframe, state_1, state_2);

            CREATE TABLE IF NOT EXISTS scan_log (
                market          TEXT    NOT NULL,
                timeframe       TEXT    NOT NULL,
                last_scan       TEXT,
                bars_processed  INTEGER DEFAULT 0,
                n_clusters      INTEGER DEFAULT 0,
                PRIMARY KEY(market, timeframe)
            );
        """)
        self.conn.commit()

    # ── Feature-Vektoren ───────────────────────────────────────────────────────

    def upsert_vector(self, market: str, tf: str, bar_time: str,
                      features: dict,
                      next_close_pct: float | None = None,
                      next_high_pct:  float | None = None,
                      next_low_pct:   float | None = None):
        feat_vals = [features.get(c) for c in FEATURE_COLS]
        self.conn.execute(f"""
            INSERT INTO feature_vectors
                (market, timeframe, bar_time, {_FEAT_SELECT},
                 next_close_pct, next_high_pct, next_low_pct)
            VALUES (?, ?, ?, {_FEAT_PLACEHOLDERS}, ?, ?, ?)
            ON CONFLICT(market, timeframe, bar_time) DO UPDATE SET
                {_FEAT_UPDATE},
                next_close_pct = COALESCE(excluded.next_close_pct, next_close_pct),
                next_high_pct  = COALESCE(excluded.next_high_pct,  next_high_pct),
                next_low_pct   = COALESCE(excluded.next_low_pct,   next_low_pct)
        """, [market, tf, bar_time] + feat_vals +
             [next_close_pct, next_high_pct, next_low_pct])
        self.conn.commit()

    def assign_state(self, market: str, tf: str, bar_time: str, state_id: int):
        self.conn.execute("""
            UPDATE feature_vectors SET state_id = ?
            WHERE market = ? AND timeframe = ? AND bar_time = ?
        """, (state_id, market, tf, bar_time))

    def commit(self):
        self.conn.commit()

    def get_all_vectors(self, market: str, tf: str) -> list[dict]:
        """Alle Vektoren, geordnet nach bar_time."""
        rows = self.conn.execute(f"""
            SELECT bar_time, {_FEAT_SELECT}, state_id,
                   next_close_pct, next_high_pct, next_low_pct
            FROM feature_vectors
            WHERE market = ? AND timeframe = ?
            ORDER BY bar_time ASC
        """, (market, tf)).fetchall()
        return [dict(r) for r in rows]

    def get_labeled_vectors(self, market: str, tf: str) -> list[dict]:
        """Nur Vektoren mit vollständigen Labels (für Training)."""
        rows = self.conn.execute(f"""
            SELECT bar_time, {_FEAT_SELECT}, state_id,
                   next_close_pct, next_high_pct, next_low_pct
            FROM feature_vectors
            WHERE market = ? AND timeframe = ?
              AND hurst IS NOT NULL
              AND next_close_pct IS NOT NULL
            ORDER BY bar_time ASC
        """, (market, tf)).fetchall()
        return [dict(r) for r in rows]

    def get_vectors_in_state(self, market: str, tf: str, state_id: int) -> list[dict]:
        rows = self.conn.execute(f"""
            SELECT bar_time, {_FEAT_SELECT}, state_id,
                   next_close_pct, next_high_pct, next_low_pct
            FROM feature_vectors
            WHERE market = ? AND timeframe = ? AND state_id = ?
              AND hurst IS NOT NULL
            ORDER BY bar_time ASC
        """, (market, tf, state_id)).fetchall()
        return [dict(r) for r in rows]

    def get_last_bar_time(self, market: str, tf: str) -> str | None:
        row = self.conn.execute("""
            SELECT bar_time FROM feature_vectors
            WHERE market = ? AND timeframe = ?
            ORDER BY bar_time DESC LIMIT 1
        """, (market, tf)).fetchone()
        return row['bar_time'] if row else None

    def get_count(self, market: str, tf: str) -> int:
        row = self.conn.execute("""
            SELECT COUNT(*) as n FROM feature_vectors
            WHERE market = ? AND timeframe = ? AND next_close_pct IS NOT NULL
        """, (market, tf)).fetchone()
        return int(row['n']) if row else 0

    def get_all_market_pairs(self) -> list[tuple[str, str]]:
        rows = self.conn.execute("""
            SELECT DISTINCT market, timeframe FROM feature_vectors
            ORDER BY market, timeframe
        """).fetchall()
        return [(r['market'], r['timeframe']) for r in rows]

    # ── State Definitions ───────────────────────────────────────────────────────

    def upsert_state_definition(self, market: str, tf: str, state_id: int,
                                 name: str, centroid: list[float],
                                 n_samples: int, avg_return: float,
                                 std_return: float, up_prob: float,
                                 quality_score: float = 0.5):
        self.conn.execute("""
            INSERT INTO state_definitions
                (market, timeframe, state_id, name, centroid_json,
                 n_samples, avg_return, std_return, up_prob, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, timeframe, state_id) DO UPDATE SET
                name          = excluded.name,
                centroid_json = excluded.centroid_json,
                n_samples     = excluded.n_samples,
                avg_return    = excluded.avg_return,
                std_return    = excluded.std_return,
                up_prob       = excluded.up_prob,
                quality_score = excluded.quality_score
        """, (market, tf, state_id, name, json.dumps(centroid),
              n_samples, avg_return, std_return, up_prob, quality_score))
        self.conn.commit()

    def get_state_definitions(self, market: str, tf: str) -> list[dict]:
        rows = self.conn.execute("""
            SELECT * FROM state_definitions
            WHERE market = ? AND timeframe = ?
            ORDER BY state_id
        """, (market, tf)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get('centroid_json'):
                d['centroid'] = json.loads(d['centroid_json'])
            result.append(d)
        return result

    def get_state_definition(self, market: str, tf: str, state_id: int) -> dict | None:
        row = self.conn.execute("""
            SELECT * FROM state_definitions
            WHERE market = ? AND timeframe = ? AND state_id = ?
        """, (market, tf, state_id)).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get('centroid_json'):
            d['centroid'] = json.loads(d['centroid_json'])
        return d

    def get_n_clusters(self, market: str, tf: str) -> int:
        row = self.conn.execute("""
            SELECT COUNT(*) as n FROM state_definitions
            WHERE market = ? AND timeframe = ?
        """, (market, tf)).fetchone()
        return int(row['n']) if row else 0

    # ── Transitions ────────────────────────────────────────────────────────────

    def upsert_transition(self, market: str, tf: str,
                           from_state: int, to_state: int,
                           count: int, probability: float):
        self.conn.execute("""
            INSERT INTO transitions
                (market, timeframe, from_state, to_state, count, probability)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, timeframe, from_state, to_state) DO UPDATE SET
                count = excluded.count, probability = excluded.probability
        """, (market, tf, from_state, to_state, count, probability))

    def delete_transitions(self, market: str, tf: str):
        self.conn.execute(
            "DELETE FROM transitions WHERE market = ? AND timeframe = ?", (market, tf))
        self.conn.execute(
            "DELETE FROM transitions_order2 WHERE market = ? AND timeframe = ?", (market, tf))

    def get_transitions_from(self, market: str, tf: str,
                              from_state: int) -> list[dict]:
        rows = self.conn.execute("""
            SELECT to_state, count, probability FROM transitions
            WHERE market = ? AND timeframe = ? AND from_state = ?
            ORDER BY probability DESC
        """, (market, tf, from_state)).fetchall()
        return [dict(r) for r in rows]

    def upsert_transition_order2(self, market: str, tf: str,
                                  state_1: int, state_2: int, to_state: int,
                                  count: int, probability: float, up_prob: float):
        self.conn.execute("""
            INSERT INTO transitions_order2
                (market, timeframe, state_1, state_2, to_state, count, probability, up_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, timeframe, state_1, state_2, to_state) DO UPDATE SET
                count = excluded.count, probability = excluded.probability,
                up_prob = excluded.up_prob
        """, (market, tf, state_1, state_2, to_state, count, probability, up_prob))

    def get_transition_order2(self, market: str, tf: str,
                               state_1: int, state_2: int) -> list[dict]:
        """Alle möglichen Nachfolge-States für (state_1 → state_2 → ?)."""
        rows = self.conn.execute("""
            SELECT to_state, count, probability, up_prob FROM transitions_order2
            WHERE market = ? AND timeframe = ? AND state_1 = ? AND state_2 = ?
            ORDER BY probability DESC
        """, (market, tf, state_1, state_2)).fetchall()
        return [dict(r) for r in rows]

    def get_previous_state(self, market: str, tf: str) -> int | None:
        """Gibt den State des vorletzten Bars zurück (für Order-2 Markov)."""
        rows = self.conn.execute("""
            SELECT state_id FROM feature_vectors
            WHERE market = ? AND timeframe = ? AND state_id IS NOT NULL
            ORDER BY bar_time DESC LIMIT 2
        """, (market, tf)).fetchall()
        return int(rows[1]['state_id']) if len(rows) >= 2 else None

    def get_all_transitions(self, market: str, tf: str) -> list[dict]:
        rows = self.conn.execute("""
            SELECT from_state, to_state, count, probability FROM transitions
            WHERE market = ? AND timeframe = ?
            ORDER BY from_state, probability DESC
        """, (market, tf)).fetchall()
        return [dict(r) for r in rows]

    # ── Scan Log ───────────────────────────────────────────────────────────────

    def update_scan_log(self, market: str, tf: str,
                         bars: int, n_clusters: int = 0):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO scan_log (market, timeframe, last_scan, bars_processed, n_clusters)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(market, timeframe) DO UPDATE SET
                last_scan = excluded.last_scan,
                bars_processed = excluded.bars_processed,
                n_clusters = excluded.n_clusters
        """, (market, tf, now, bars, n_clusters))
        self.conn.commit()

    def get_scan_log(self, market: str, tf: str) -> dict | None:
        row = self.conn.execute("""
            SELECT * FROM scan_log WHERE market = ? AND timeframe = ?
        """, (market, tf)).fetchone()
        return dict(row) if row else None

    def get_summary(self) -> list[dict]:
        rows = self.conn.execute("""
            SELECT fv.market, fv.timeframe,
                   COUNT(*) as total,
                   SUM(CASE WHEN fv.state_id IS NOT NULL THEN 1 ELSE 0 END) as clustered,
                   MAX(fv.bar_time) as latest_bar,
                   sl.last_scan, sl.n_clusters
            FROM feature_vectors fv
            LEFT JOIN scan_log sl ON sl.market = fv.market AND sl.timeframe = fv.timeframe
            GROUP BY fv.market, fv.timeframe
            ORDER BY fv.market, fv.timeframe
        """).fetchall()
        return [dict(r) for r in rows]

    @property
    def calibrator(self):
        return self._calibrator

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
