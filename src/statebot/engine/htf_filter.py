# engine/htf_filter.py — HTF Supertrend Veto
#
# Übernommen aus pbot/predictor_engine.py, angepasst für statebot.
#
# Funktion:
#   Ein übergeordneter Supertrend auf einem höheren Timeframe (HTF) blockiert
#   Gegentrend-Trades hart — unabhängig von p_bayes.
#
#   HTF-Supertrend grün (+1) → nur Longs  → Short wird geblockt
#   HTF-Supertrend rot  (-1) → nur Shorts → Long  wird geblockt
#
# Konfiguration in settings.json:
#   "htf_filter": {
#       "enabled": true,
#       "timeframe": null,   // null = Auto-Mapping vom Base-TF
#       "factor": 3.0,
#       "period": 10
#   }
#
# Auto-Mapping (falls timeframe=null):
#   1m/3m/5m → 15m | 15m → 1h | 30m → 4h | 1h → 4h | 4h → 1d | 1d → 3d | 3d → 1w

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

HTF_MAP: dict[str, str] = {
    '1m': '15m', '3m': '15m', '5m': '15m',
    '15m': '1h', '30m': '4h',
    '1h': '4h', '2h': '1d', '4h': '1d',
    '6h': '3d', '8h': '3d', '12h': '3d',
    '1d': '3d', '3d': '1w', '1w': '1M',
}


def auto_htf(base_tf: str) -> str:
    """Gibt den empfohlenen Higher Timeframe zurück."""
    return HTF_MAP.get(base_tf, '1d')


def compute_supertrend(df: pd.DataFrame,
                        period: int = 10,
                        factor: float = 3.0) -> np.ndarray:
    """
    Berechnet Supertrend-Richtung für jede Kerze.

    Returns:
        trend — np.ndarray derselben Länge wie df:
                +1 = bullish (Preis über Supertrend-Linie)
                -1 = bearish (Preis unter Supertrend-Linie)

    Algorithmus identisch zu pbot/predictor_engine.py (Wilder ATR, Trail).
    """
    if df.empty or len(df) < period:
        return np.ones(len(df), dtype=float)

    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    close = df['close'].values.astype(float)
    n     = len(df)

    # Wilder ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))

    atr = np.zeros(n)
    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2          = (high + low) / 2.0
    basic_upper  = hl2 + factor * atr
    basic_lower  = hl2 - factor * atr

    final_upper  = basic_upper.copy()
    final_lower  = basic_lower.copy()
    trend        = np.ones(n, dtype=float)

    for i in range(1, n):
        # Upper Band trailing
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower Band trailing
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Trend-Flip
        if trend[i - 1] == 1:
            trend[i] = -1.0 if close[i] <= final_lower[i] else 1.0
        else:
            trend[i] =  1.0 if close[i] >= final_upper[i] else -1.0

    return trend


def get_htf_trend(htf_df: pd.DataFrame,
                   period: int = 10,
                   factor: float = 3.0) -> int | None:
    """
    Berechnet HTF-Supertrend und gibt den aktuellen Trend zurück.

    Returns:
        +1  = HTF bullish
        -1  = HTF bearish
        None = zu wenig Daten
    """
    if htf_df is None or htf_df.empty or len(htf_df) < period + 2:
        logger.debug("HTF-Filter: nicht genug Daten für Supertrend")
        return None

    trend = compute_supertrend(htf_df, period=period, factor=factor)
    return int(trend[-1])


def apply_htf_veto(side: str, htf_trend: int | None) -> bool:
    """
    Gibt True zurück wenn der Trade geblockt werden soll.

    HTF +1 (grün) → nur Longs   → Short → True  (geblockt)
    HTF -1 (rot)  → nur Shorts  → Long  → True  (geblockt)
    None           → kein Veto   → False
    """
    if htf_trend is None:
        return False
    if htf_trend == 1 and side == 'short':
        logger.info("HTF-Supertrend grün — Short geblockt")
        return True
    if htf_trend == -1 and side == 'long':
        logger.info("HTF-Supertrend rot — Long geblockt")
        return True
    return False
