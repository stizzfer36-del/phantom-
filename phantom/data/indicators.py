"""
Technical indicators implemented with numpy only.

All functions accept array-like inputs (list or np.ndarray) and return
a scalar float or a dict of floats. They return None when the input is
too short for the requested period and log a warning.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _arr(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _ema_series(values: np.ndarray, period: int) -> np.ndarray:
    """
    Return a full EMA series the same length as values.
    The first period-1 values are filled with np.nan.
    Seed is the simple mean of the first `period` values.
    """
    k = 2.0 / (period + 1)
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return result
    result[period - 1] = values[:period].mean()
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(closes, period: int = 14) -> float | None:
    """
    Wilder's RSI for the most recent bar.

    Uses the standard Wilder smoothing (RMA / SMMA):
      avg_gain = ((prev_avg_gain × (period-1)) + current_gain) / period
    """
    c = _arr(closes)
    if len(c) < period + 1:
        logger.warning("rsi: need at least %d bars, got %d", period + 1, len(c))
        return None

    deltas = np.diff(c)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple averages over first `period` deltas
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    # Wilder smoothing over remaining bars
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(float(100.0 - 100.0 / (1.0 + rs)), 4)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def ema(closes, period: int) -> float | None:
    """Latest EMA value for an arbitrary period."""
    c = _arr(closes)
    if len(c) < period:
        logger.warning("ema(%d): need at least %d bars, got %d", period, period, len(c))
        return None
    series = _ema_series(c, period)
    val = series[~np.isnan(series)]
    return round(float(val[-1]), 8) if len(val) else None


def ema9(closes) -> float | None:
    """Latest 9-period EMA."""
    return ema(closes, 9)


def ema21(closes) -> float | None:
    """Latest 21-period EMA."""
    return ema(closes, 21)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    closes,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float] | None:
    """
    MACD indicator.

    Returns:
        {
          "macd":      float,   # fast EMA − slow EMA (latest)
          "signal":    float,   # signal-line EMA of the MACD series (latest)
          "histogram": float,   # macd − signal
        }
    """
    c = _arr(closes)
    min_len = slow + signal
    if len(c) < min_len:
        logger.warning("macd: need at least %d bars, got %d", min_len, len(c))
        return None

    fast_ema  = _ema_series(c, fast)
    slow_ema  = _ema_series(c, slow)
    macd_line = fast_ema - slow_ema          # element-wise; nan where either is nan

    # Compute signal line only over the non-nan part of macd_line
    valid_mask = ~np.isnan(macd_line)
    valid_macd = macd_line[valid_mask]
    if len(valid_macd) < signal:
        logger.warning("macd: not enough valid MACD values for signal line")
        return None

    sig_series = _ema_series(valid_macd, signal)
    sig_valid  = sig_series[~np.isnan(sig_series)]
    if not len(sig_valid):
        return None

    macd_val = float(valid_macd[-1])
    sig_val  = float(sig_valid[-1])
    return {
        "macd":      round(macd_val, 8),
        "signal":    round(sig_val, 8),
        "histogram": round(macd_val - sig_val, 8),
    }


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    closes,
    period: int = 20,
    num_std: float = 2.0,
) -> dict[str, float] | None:
    """
    Bollinger Bands around a simple moving average.

    Returns:
        {
          "upper":     float,
          "middle":    float,   # SMA
          "lower":     float,
          "bandwidth": float,   # (upper - lower) / middle
          "pct_b":     float,   # (price - lower) / (upper - lower)
        }
    """
    c = _arr(closes)
    if len(c) < period:
        logger.warning("bollinger_bands: need at least %d bars, got %d", period, len(c))
        return None

    window = c[-period:]
    sma    = float(window.mean())
    std    = float(window.std(ddof=1))
    upper  = sma + num_std * std
    lower  = sma - num_std * std
    band_width = (upper - lower) / sma if sma != 0 else 0.0
    price  = float(c[-1])
    pct_b  = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

    return {
        "upper":     round(upper, 8),
        "middle":    round(sma, 8),
        "lower":     round(lower, 8),
        "bandwidth": round(band_width, 6),
        "pct_b":     round(pct_b, 4),
    }


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def atr(highs, lows, closes, period: int = 14) -> float | None:
    """
    Wilder-smoothed Average True Range.

    True range = max(high-low, |high-prev_close|, |low-prev_close|).
    Seeded with the simple mean of the first `period` TR values, then
    Wilder-smoothed for the remainder.
    """
    h, l, c = _arr(highs), _arr(lows), _arr(closes)
    if len(c) < period + 1:
        logger.warning("atr: need at least %d bars, got %d", period + 1, len(c))
        return None

    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )
    result = float(tr[:period].mean())
    for i in range(period, len(tr)):
        result = (result * (period - 1) + float(tr[i])) / period
    return round(result, 8)


# ---------------------------------------------------------------------------
# VWAP (Volume Weighted Average Price)
# ---------------------------------------------------------------------------

def vwap(highs, lows, closes, volumes) -> float | None:
    """
    VWAP computed over the provided bars.  typical_price = (H+L+C)/3.
    """
    h, l, c, v = _arr(highs), _arr(lows), _arr(closes), _arr(volumes)
    if len(c) == 0:
        logger.warning("vwap: empty input")
        return None
    total_vol = float(v.sum())
    if total_vol == 0.0:
        return None
    typical = (h + l + c) / 3.0
    return round(float((typical * v).sum() / total_vol), 8)


# ---------------------------------------------------------------------------
# Volume ratio
# ---------------------------------------------------------------------------

def volume_ratio(volumes, period: int = 20) -> float | None:
    """
    Ratio of the most recent volume bar to the simple average of the
    prior `period` bars.

    A value > 1.0 means above-average volume; > 3.0 is a strong spike.
    """
    v = _arr(volumes)
    if len(v) < period + 1:
        logger.warning("volume_ratio: need at least %d bars, got %d", period + 1, len(v))
        return None

    avg = v[-period - 1 : -1].mean()
    if avg == 0.0:
        return None
    return round(float(v[-1]) / float(avg), 4)


# ---------------------------------------------------------------------------
# Convenience: full signal dict for a price/volume series
# ---------------------------------------------------------------------------

def compute_all(closes, volumes=None) -> dict:
    """
    Compute all indicators in one call.

    Returns a dict with keys: rsi, ema9, ema21, macd, bollinger, volume_ratio.
    Any indicator that cannot be computed is set to None.
    """
    return {
        "rsi":          rsi(closes),
        "ema9":         ema9(closes),
        "ema21":        ema21(closes),
        "macd":         macd(closes),
        "bollinger":    bollinger_bands(closes),
        "volume_ratio": volume_ratio(volumes) if volumes is not None else None,
    }
