"""
Strategy: Momentum Breakout
Pool:     momentum_tracker  (15% of bankroll)
Assets:   US equities  (via Alpaca Data API)

Entry conditions (all required):
  - Latest close > 20-day high (prior 20 bars)
  - Today's volume ≥ 2× 20-day average volume
  - RSI(14) > 55  (momentum confirmation, not overbought)
  - EMA9 > EMA21  (trend alignment)

Risk:
  - Stop loss   : 2× ATR(14) below entry
  - Trailing stop: 2× ATR(14) trailing from the highest close reached
  - Position size: $6 per position, max 2 open positions
"""

import logging
from typing import Any

import httpx
import numpy as np

from phantom.data.indicators import rsi, ema9, ema21, atr
from phantom.data.prices import _alpaca_headers, _ALPACA_DATA, _TIMEOUT
from phantom.data.scanner import _DEFAULT_EQUITY_UNIVERSE

logger = logging.getLogger(__name__)

POOL          = "momentum_tracker"
LOOKBACK      = 20
MIN_VOL_RATIO = 2.0
MIN_RSI       = 55.0
POSITION_SIZE = 6.00
MAX_POSITIONS = 2
ATR_MULT      = 2.0   # stop and trail width in ATR multiples
ATR_PERIOD    = 14


def _get_bars(symbol: str, limit: int, feed: str = "iex") -> list[dict] | None:
    try:
        resp = httpx.get(
            f"{_ALPACA_DATA}/stocks/{symbol}/bars",
            headers=_alpaca_headers(),
            params={"timeframe": "1Day", "limit": limit, "feed": feed, "sort": "asc"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("bars", [])
    except Exception as exc:
        logger.warning("momentum_breakout._get_bars(%s): %s", symbol, exc)
        return None


def _open_position_count(state: dict, pool: str) -> int:
    return len(state["pools"][pool]["positions"])


def generate_signals(
    state: dict[str, Any],
    symbols: list[str] | None = None,
    feed: str = "iex",
) -> list[dict[str, Any]]:
    """
    Scan equities for 20-day high breakouts confirmed by volume and momentum.

    symbols — optional watchlist; defaults to _DEFAULT_EQUITY_UNIVERSE.
    Fetches LOOKBACK + ATR_PERIOD + 1 bars per symbol to satisfy all indicators.
    """
    pool_cash  = state["pools"][POOL]["cash"]
    open_slots = MAX_POSITIONS - _open_position_count(state, POOL)
    held       = set(state["pools"][POOL]["positions"].keys())

    if open_slots <= 0:
        logger.info("momentum_breakout: pool full (%d positions)", MAX_POSITIONS)
        return []
    if pool_cash < POSITION_SIZE:
        logger.info("momentum_breakout: insufficient cash %.2f", pool_cash)
        return []

    universe = [s for s in (symbols or _DEFAULT_EQUITY_UNIVERSE) if s not in held]
    needed   = LOOKBACK + ATR_PERIOD + 2   # enough for all indicators
    signals: list[dict] = []

    for sym in universe:
        bars = _get_bars(sym, limit=needed, feed=feed)
        if not bars or len(bars) < needed:
            continue

        closes  = np.array([float(b["c"]) for b in bars])
        highs   = np.array([float(b["h"]) for b in bars])
        lows    = np.array([float(b["l"]) for b in bars])
        volumes = np.array([float(b["v"]) for b in bars])

        entry       = float(closes[-1])
        period_high = float(closes[-(LOOKBACK + 1):-1].max())  # prior 20 closes

        # Hard filters first (cheapest checks)
        if entry <= period_high:
            continue

        avg_vol   = float(volumes[-(LOOKBACK + 1):-1].mean())
        today_vol = float(volumes[-1])
        if avg_vol == 0 or today_vol / avg_vol < MIN_VOL_RATIO:
            continue

        rsi_val  = rsi(closes)
        ema9_val = ema9(closes)
        ema21_val= ema21(closes)
        atr_val  = atr(highs, lows, closes, period=ATR_PERIOD)

        if rsi_val is None or rsi_val < MIN_RSI:
            continue
        if ema9_val is None or ema21_val is None or ema9_val <= ema21_val:
            continue
        if atr_val is None or atr_val == 0:
            continue

        vol_ratio    = round(today_vol / avg_vol, 2)
        breakout_pct = round((entry - period_high) / period_high * 100, 2)
        stop_dist    = ATR_MULT * atr_val

        # ── Scoring ────────────────────────────────────────────────────────
        score   = 0
        reasons = []

        reasons.append(f"20d high breakout +{breakout_pct:.1f}%")
        score += min(int(breakout_pct * 5), 25)   # up to 25 pts for breakout size

        if vol_ratio >= 4:
            score += 35; reasons.append(f"vol {vol_ratio:.1f}× avg")
        elif vol_ratio >= 3:
            score += 25; reasons.append(f"vol {vol_ratio:.1f}× avg")
        else:
            score += 15; reasons.append(f"vol {vol_ratio:.1f}× avg")

        if rsi_val >= 65:
            score += 20; reasons.append(f"RSI strong ({rsi_val:.1f})")
        elif rsi_val >= 55:
            score += 12; reasons.append(f"RSI momentum ({rsi_val:.1f})")

        score += 15; reasons.append("EMA9 > EMA21")

        if score < 55:
            continue

        confidence = min(score, 95)
        stop_loss  = round(entry - stop_dist, 4)
        # Take profit: fixed 3× ATR above entry (informational; real exit uses trailing)
        take_profit = round(entry + ATR_MULT * 3 * atr_val, 4)

        signals.append({
            "asset":             sym,
            "direction":         "long",
            "pool":              POOL,
            "confidence":        confidence,
            "reason":            "; ".join(reasons),
            "entry_price":       round(entry, 4),
            "stop_loss":         stop_loss,
            "take_profit":       take_profit,
            "position_size_usd": POSITION_SIZE,
            # ATR trailing stop metadata
            "atr":               round(atr_val, 4),
            "trail_atr_mult":    ATR_MULT,
            "trail_stop_dist":   round(stop_dist, 4),
            # Context
            "period_high":       round(period_high, 4),
            "breakout_pct":      breakout_pct,
            "vol_ratio":         vol_ratio,
            "rsi":               round(rsi_val, 2),
            "ema9":              round(ema9_val, 4),
            "ema21":             round(ema21_val, 4),
            "asset_type":        "stock",
        })

    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals[:open_slots]
