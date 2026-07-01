# engine/features.py — 22-dimensionaler Marktzustandsvektor
#
# Kategorien:
#   Trend        : ema20_dist, ema50_dist, ema200_dist, ema_cross
#   Momentum     : rsi, roc, macd_hist
#   Volatilität  : atr_ratio, atr_change, hv20
#   Fraktal      : hurst, fd
#   Marktenergie : entropy, shannon
#   Kerze        : body_ratio, upper_wick, lower_wick, gap
#   Orderflow    : cvd_slope
#   Volumen      : rel_volume, obv_slope, delta_ratio

import numpy as np
import pandas as pd
import math
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'hurst', 'fd', 'entropy', 'shannon',
    'rsi', 'roc', 'macd_hist',
    'ema20_dist', 'ema50_dist', 'ema200_dist', 'ema_cross',
    'atr_ratio', 'atr_change', 'hv20',
    'body_ratio', 'upper_wick', 'lower_wick', 'gap',
    'cvd_slope',
    'rel_volume', 'obv_slope', 'delta_ratio',
]

HURST_WINDOW  = 60
EMA200_PERIOD = 200
WARMUP_BARS   = EMA200_PERIOD + 15


# ─── Hurst-Exponent (R/S) ────────────────────────────────────────────────────

def _hurst_rs(series: np.ndarray) -> float:
    n = len(series)
    if n < 20:
        return 0.5
    log_ret = np.diff(np.log(np.clip(series, 1e-10, None)))
    n_ret = len(log_ret)
    max_lag = max(8, n_ret // 4)
    lags = np.unique(np.logspace(np.log10(4), np.log10(max_lag), 8).astype(int))
    log_lags, log_rs = [], []
    for lag in lags:
        if lag >= n_ret:
            continue
        chunks = [log_ret[i:i+lag] for i in range(0, n_ret - lag + 1, lag)]
        rs_vals = []
        for chunk in chunks:
            if len(chunk) < 4:
                continue
            dev = np.cumsum(chunk - chunk.mean())
            R   = dev.max() - dev.min()
            S   = chunk.std(ddof=1)
            if S > 1e-10:
                rs_vals.append(R / S)
        if rs_vals:
            log_lags.append(np.log(lag))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_lags) < 2:
        return 0.5
    return float(np.clip(np.polyfit(log_lags, log_rs, 1)[0], 0.1, 0.9))


# ─── Fraktale Dimension (Petrosian) ──────────────────────────────────────────

def _petrosian_fd(series: np.ndarray) -> float:
    n = len(series)
    if n < 5:
        return 1.0
    diff = np.diff(series)
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    if sign_changes == 0:
        return 1.0
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * sign_changes)))


# ─── Permutation Entropy ─────────────────────────────────────────────────────

def _permutation_entropy(series: np.ndarray, order: int = 3) -> float:
    n = len(series)
    if n < order + 2:
        return 0.5
    patterns: dict = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(series[i:i+order]))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    total = sum(patterns.values())
    probs = [v / total for v in patterns.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    max_ent = np.log(math.factorial(order))
    return float(entropy / max_ent) if max_ent > 0 else 0.5


# ─── Shannon Entropy ──────────────────────────────────────────────────────────

def _shannon_entropy(returns: np.ndarray, n_bins: int = 10) -> float:
    if len(returns) < 5:
        return 0.5
    hist, _ = np.histogram(returns, bins=n_bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.5
    prob = hist / total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log(prob + 1e-10)) / np.log(n_bins))


# ─── ATR ─────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df['close'].shift(1)
    tr   = pd.concat([df['high'] - df['low'],
                      (df['high'] - prev).abs(),
                      (df['low']  - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


# ─── CVD ─────────────────────────────────────────────────────────────────────

def _cvd(df: pd.DataFrame) -> pd.Series:
    hl = (df['high'] - df['low']).replace(0, np.nan)
    delta = df['volume'] * (2 * df['close'] - df['high'] - df['low']) / hl
    return delta.fillna(0).cumsum()


# ─── OBV ─────────────────────────────────────────────────────────────────────

def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff().fillna(0))
    return (df['volume'] * direction).cumsum()


# ─── Haupt-Feature-Berechnung ─────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    closes  = df['close'].values
    opens   = df['open'].values
    highs   = df['high'].values
    lows    = df['low'].values
    volumes = df['volume'].values
    n = len(df)

    # ── Trend ──────────────────────────────────────────────────────────────────
    ema20  = df['close'].ewm(span=20,  min_periods=20).mean()
    ema50  = df['close'].ewm(span=50,  min_periods=50).mean()
    ema200 = df['close'].ewm(span=200, min_periods=200).mean()
    out['ema20_dist']  = (df['close'] - ema20)  / ema20.replace(0, np.nan)
    out['ema50_dist']  = (df['close'] - ema50)  / ema50.replace(0, np.nan)
    out['ema200_dist'] = (df['close'] - ema200) / ema200.replace(0, np.nan)
    out['ema_cross']   = (ema20 - ema50) / ema50.replace(0, np.nan)

    # ── Momentum ───────────────────────────────────────────────────────────────
    # RSI(14)
    delta_c = df['close'].diff()
    gain    = delta_c.clip(lower=0).ewm(span=14, min_periods=14).mean()
    loss    = (-delta_c).clip(lower=0).ewm(span=14, min_periods=14).mean()
    rs      = gain / loss.replace(0, np.nan)
    out['rsi'] = (100 - 100 / (1 + rs)) / 100   # normiert 0–1

    # ROC(10)
    out['roc'] = df['close'].pct_change(10)

    # MACD histogram normiert durch ATR
    macd_line   = df['close'].ewm(span=12, min_periods=12).mean() - \
                  df['close'].ewm(span=26, min_periods=26).mean()
    signal_line = macd_line.ewm(span=9, min_periods=9).mean()
    atr14       = _atr(df, 14)
    out['macd_hist'] = (macd_line - signal_line) / atr14.replace(0, np.nan)

    # ── Volatilität ────────────────────────────────────────────────────────────
    out['atr_ratio']  = atr14 / df['close'].replace(0, np.nan)
    atr28 = _atr(df, 28)
    out['atr_change'] = (atr14 / atr28.replace(0, np.nan)) - 1
    log_ret = np.log(df['close'] / df['close'].shift(1).replace(0, np.nan))
    out['hv20'] = log_ret.rolling(20).std() * np.sqrt(252)

    # ── Fraktal ────────────────────────────────────────────────────────────────
    hurst_vals = np.full(n, np.nan)
    fd_vals    = np.full(n, np.nan)
    ent_vals   = np.full(n, np.nan)
    sha_vals   = np.full(n, np.nan)
    log_rets   = np.diff(np.log(np.clip(closes, 1e-10, None)), prepend=np.nan)

    for i in range(HURST_WINDOW, n):
        window_c    = closes[i - HURST_WINDOW:i]
        window_r    = log_rets[i - HURST_WINDOW:i]
        hurst_vals[i] = _hurst_rs(window_c)
        fd_vals[i]    = _petrosian_fd(window_c)
        ent_vals[i]   = _permutation_entropy(window_c)
        sha_vals[i]   = _shannon_entropy(window_r[~np.isnan(window_r)])

    out['hurst']   = hurst_vals
    out['fd']      = fd_vals
    out['entropy'] = ent_vals
    out['shannon'] = sha_vals

    # ── Kerze ──────────────────────────────────────────────────────────────────
    hl = (df['high'] - df['low']).replace(0, np.nan)
    upper_body = df[['open', 'close']].max(axis=1)
    lower_body = df[['open', 'close']].min(axis=1)
    out['body_ratio']  = (df['close'] - df['open']) / hl
    out['upper_wick']  = (df['high'] - upper_body) / hl
    out['lower_wick']  = (lower_body - df['low'])  / hl
    out['gap']         = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)

    # ── Orderflow ──────────────────────────────────────────────────────────────
    cvd       = _cvd(df)
    cvd_std   = cvd.rolling(50).std().replace(0, np.nan)
    out['cvd_slope'] = (cvd - cvd.shift(5)) / (cvd_std + 1e-10)

    # ── Volumen ────────────────────────────────────────────────────────────────
    vol_ma          = df['volume'].rolling(20).mean().replace(0, np.nan)
    out['rel_volume'] = df['volume'] / vol_ma

    obv        = _obv(df)
    obv_std    = obv.rolling(20).std().replace(0, np.nan)
    out['obv_slope'] = (obv - obv.shift(5)) / (obv_std + 1e-10)

    # Delta ratio (Kaeufer vs. Verkaeufer)
    delta_raw  = df['volume'] * (2 * df['close'] - df['high'] - df['low']) / hl
    delta_abs  = delta_raw.abs().rolling(20).mean().replace(0, np.nan)
    out['delta_ratio'] = delta_raw / (delta_abs + 1e-10)

    return out


def get_feature_vector(df_features: pd.DataFrame, idx: int) -> dict | None:
    row = df_features.iloc[idx]
    vec = {col: float(row[col]) for col in FEATURE_COLS}
    if any(np.isnan(v) for v in vec.values()):
        return None
    return vec


def feature_dict_to_array(fvec: dict) -> np.ndarray:
    return np.array([fvec[c] for c in FEATURE_COLS], dtype=np.float64)
