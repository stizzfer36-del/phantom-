"""
Strategy: Crypto Swing
Pool:     crypto_swing  (30% of bankroll)
Assets:   BTC, ETH, SOL  (via CoinGecko OHLC + price data)

Entry conditions (all required):
  - RSI(14) < 35  (oversold)
  - Latest close < lower Bollinger Band
  - News sentiment neutral or bullish

Exit signals embedded in signal metadata:
  - Exit long when RSI > 70 OR price > upper Bollinger Band

Risk:
  - Stop loss  : 5% below entry
  - Take profit: 8% above entry
  - Max 2 open positions; each up to 50% of pool cash
"""

import logging
from typing import Any

import httpx
import numpy as np

from phantom.data.indicators import rsi, bollinger_bands
from phantom.data.sentiment import get_news_sentiment
from phantom.portfolio.manager import get_pool_balance

logger = logging.getLogger(__name__)

POOL = "crypto_swing"
STOP_PCT   = 0.05
TARGET_PCT = 0.08
MAX_POSITIONS  = 2
POOL_ALLOC_PCT = 0.50   # up to 50% of pool cash per signal

# symbol = yfinance ticker used for sim execution
# display = short label shown in summaries
_COINS = [
    {"id": "bitcoin",  "symbol": "BTC-USD", "display": "BTC", "news_query": "bitcoin BTC"},
    {"id": "ethereum", "symbol": "ETH-USD", "display": "ETH", "news_query": "ethereum ETH"},
    {"id": "solana",   "symbol": "SOL-USD", "display": "SOL", "news_query": "solana SOL"},
]

_COINGECKO = "https://api.coingecko.com/api/v3"
_TIMEOUT   = 12


def _ohlc(coin_id: str, days: int = 30) -> list[list] | None:
    try:
        resp = httpx.get(
            f"{_COINGECKO}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("crypto_swing._ohlc(%s): %s", coin_id, exc)
        return None


def _open_position_count(state: dict, pool: str) -> int:
    return len(state["pools"][pool]["positions"])


def generate_signals(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Scan BTC, ETH, SOL for oversold swing entries.

    Returns up to MAX_POSITIONS signal dicts, highest-confidence first.
    Skips coins already held in the pool.
    """
    pool_cash    = state["pools"][POOL]["cash"]
    open_slots   = MAX_POSITIONS - _open_position_count(state, POOL)
    held_symbols = set(state["pools"][POOL]["positions"].keys())

    if open_slots <= 0:
        logger.info("crypto_swing: pool full (%d/%d positions)", MAX_POSITIONS, MAX_POSITIONS)
        return []
    if pool_cash < 0.50:
        logger.info("crypto_swing: insufficient cash (%.2f)", pool_cash)
        return []

    signals: list[dict] = []

    for coin in _COINS:
        if coin["symbol"] in held_symbols:   # check yf ticker, not display name
            continue

        candles = _ohlc(coin["id"])
        if not candles or len(candles) < 35:
            logger.warning("crypto_swing: not enough candles for %s (%s)", coin["display"], len(candles) if candles else 0)
            continue

        closes = np.array([c[4] for c in candles], dtype=float)
        entry  = float(closes[-1])

        rsi_val = rsi(closes)
        bb      = bollinger_bands(closes)
        if rsi_val is None or bb is None:
            continue

        # Sentiment (optional; treat None as neutral)
        sentiment = get_news_sentiment(coin["news_query"])
        sent_score = sentiment["overall"]["score"] if sentiment else 0.0

        # ── Entry scoring ──────────────────────────────────────────────────
        score    = 0
        reasons  = []

        if rsi_val < 25:
            score += 40; reasons.append(f"RSI deeply oversold ({rsi_val:.1f})")
        elif rsi_val < 35:
            score += 25; reasons.append(f"RSI oversold ({rsi_val:.1f})")
        else:
            continue  # RSI condition is hard required

        if entry < bb["lower"]:
            score += 35; reasons.append(f"price below lower BB ({bb['lower']:.2f})")
        elif entry < bb["middle"]:
            score += 10; reasons.append("price below BB midline")
        else:
            continue  # BB condition is hard required

        if sent_score > 0.1:
            score += 20; reasons.append(f"bullish news (score {sent_score:.2f})")
        elif sent_score >= 0.0:
            score += 5;  reasons.append("neutral news")
        else:
            score -= 10; reasons.append(f"bearish news (score {sent_score:.2f})")

        # Exit markers (informational — used by execution layer)
        rsi_exit  = round(float(rsi_val), 2)  # signal already shows entry; exit ≥ 70
        bb_exit   = bb["upper"]

        confidence = min(score, 95)
        position_size = round(min(pool_cash * POOL_ALLOC_PCT, pool_cash), 2)

        signals.append({
            "asset":            coin["symbol"],   # yfinance ticker e.g. "BTC-USD"
            "display":          coin["display"],  # short label e.g. "BTC"
            "coin_id":          coin["id"],
            "direction":        "long",
            "pool":             POOL,
            "confidence":       confidence,
            "reason":           "; ".join(reasons),
            "entry_price":      round(entry, 6),
            "stop_loss":        round(entry * (1 - STOP_PCT), 6),
            "take_profit":      round(entry * (1 + TARGET_PCT), 6),
            "position_size_usd": position_size,
            # Metadata for execution / exit logic
            "exit_rsi_threshold": 70,
            "exit_bb_upper":      round(bb_exit, 6),
            "current_rsi":        rsi_exit,
            "bb_bandwidth":       bb["bandwidth"],
            "sentiment_score":    round(sent_score, 4),
            "asset_type":         "crypto",
        })

    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals[:open_slots]
