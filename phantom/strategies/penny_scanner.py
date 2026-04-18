"""
Strategy: Penny Scanner
Pool:     penny_scanner  (25% of bankroll)
Assets:   US equities priced under $5

Entry conditions (all required):
  - Volume ≥ 3× previous-day volume
  - Price within ±3% of VWAP
  - Bullish or neutral news sentiment

Risk:
  - Stop loss      : 8% below entry
  - Take profit    : 15% above entry
  - Trailing stop  : activates at +8% gain, trails by 4%
  - Position size  : $5 per position, max 4 open positions
"""

import logging
from typing import Any

import httpx

from phantom.data.indicators import rsi, vwap
from phantom.data.prices import _alpaca_headers, _ALPACA_DATA, _TIMEOUT, get_stock_snapshots, alpaca_configured
from phantom.data.sentiment import get_news_sentiment
from phantom.data.scanner import _DEFAULT_EQUITY_UNIVERSE

logger = logging.getLogger(__name__)

POOL            = "penny_scanner"
STOP_PCT        = 0.08
TARGET_PCT      = 0.15
TRAIL_TRIGGER   = 0.08   # trailing stop activates after 8% gain
TRAIL_PCT       = 0.04   # trails by 4%
POSITION_SIZE   = 5.00   # $5 fixed per position
MAX_POSITIONS   = 4
MAX_PRICE       = 5.00
MIN_VOL_RATIO   = 3.0
VWAP_BAND_PCT   = 0.03   # price must be within ±3% of VWAP

# Penny-stock focused watchlist (extends the general equity universe)
_PENNY_UNIVERSE = [
    "SNDL", "ACB", "CGC", "TLRY", "HEXO", "MVIS", "EYES", "IDEX",
    "ILUS", "CLOV", "SPCE", "WISH", "SKLZ", "RKT", "OPEN", "BB",
    "NOK", "AMC", "HOOD", "SOFI", "LCID", "RIVN",
]


def _get_bars(symbol: str, limit: int = 15, feed: str = "iex") -> list[dict] | None:
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
        logger.warning("penny_scanner._get_bars(%s): %s", symbol, exc)
        return None


def _open_position_count(state: dict, pool: str) -> int:
    return len(state["pools"][pool]["positions"])


def generate_signals(
    state: dict[str, Any],
    symbols: list[str] | None = None,
    feed: str = "iex",
) -> list[dict[str, Any]]:
    """
    Scan penny stocks for high-volume, sentiment-confirmed entries.

    symbols  — custom watchlist; defaults to _PENNY_UNIVERSE.
    """
    if not alpaca_configured():
        logger.warning("penny_scanner: ALPACA_API_KEY not set — strategy skipped")
        return []

    pool_cash  = state["pools"][POOL]["cash"]
    open_slots = MAX_POSITIONS - _open_position_count(state, POOL)
    held       = set(state["pools"][POOL]["positions"].keys())

    if open_slots <= 0:
        logger.info("penny_scanner: pool full (%d positions)", MAX_POSITIONS)
        return []
    if pool_cash < POSITION_SIZE:
        logger.info("penny_scanner: insufficient cash %.2f", pool_cash)
        return []

    universe = [s for s in (symbols or _PENNY_UNIVERSE) if s not in held]
    signals: list[dict] = []

    # Batch snapshots (Alpaca 100 per request)
    for i in range(0, len(universe), 100):
        batch = universe[i : i + 100]
        snapshots = get_stock_snapshots(batch, feed=feed)
        if not snapshots:
            continue

        for sym, snap in snapshots.items():
            daily = snap.get("dailyBar") or {}
            prev  = snap.get("prevDailyBar") or {}

            price     = float(daily.get("c") or 0)
            today_vol = float(daily.get("v") or 0)
            prev_vol  = float(prev.get("v") or 0)

            if price <= 0 or price > MAX_PRICE:
                continue
            if prev_vol <= 0:
                continue
            vol_ratio = today_vol / prev_vol
            if vol_ratio < MIN_VOL_RATIO:
                continue

            # VWAP — prefer the bar's built-in vw field; fall back to calculated
            bar_vwap = float(daily.get("vw") or 0)
            if bar_vwap == 0:
                h = float(daily.get("h") or price)
                l = float(daily.get("l") or price)
                bar_vwap = (h + l + price) / 3

            vwap_diff_pct = abs(price - bar_vwap) / bar_vwap if bar_vwap else 1.0
            if vwap_diff_pct > VWAP_BAND_PCT:
                continue

            # RSI from historical bars (graceful skip if unavailable)
            bars = _get_bars(sym)
            rsi_val = None
            if bars and len(bars) >= 15:
                closes  = [float(b["c"]) for b in bars]
                rsi_val = rsi(closes)

            # Sentiment
            sentiment  = get_news_sentiment(sym)
            sent_score = sentiment["overall"]["score"] if sentiment else 0.0

            # ── Scoring ──────────────────────────────────────────────────
            score   = 0
            reasons = []

            if vol_ratio >= 5:
                score += 35; reasons.append(f"vol spike {vol_ratio:.1f}×")
            elif vol_ratio >= 3:
                score += 20; reasons.append(f"vol spike {vol_ratio:.1f}×")

            near_vwap = vwap_diff_pct <= 0.01
            if near_vwap:
                score += 30; reasons.append(f"price at VWAP ({bar_vwap:.3f})")
            else:
                score += 15; reasons.append(f"price near VWAP ({vwap_diff_pct*100:.1f}% off)")

            if sent_score > 0.1:
                score += 25; reasons.append(f"bullish news ({sent_score:.2f})")
            elif sent_score >= 0.0:
                score += 10; reasons.append("neutral news")
            else:
                continue  # bearish news: skip

            if rsi_val is not None:
                if rsi_val < 60:
                    score += 10; reasons.append(f"RSI {rsi_val:.1f} (room to run)")
                elif rsi_val > 75:
                    score -= 15; reasons.append(f"RSI overbought ({rsi_val:.1f})")

            if price < 1.0:
                score += 5; reasons.append(f"sub-dollar (${price:.3f})")

            if score < 50:
                continue

            confidence = min(score, 95)
            signals.append({
                "asset":             sym,
                "direction":         "long",
                "pool":              POOL,
                "confidence":        confidence,
                "reason":            "; ".join(reasons),
                "entry_price":       round(price, 4),
                "stop_loss":         round(price * (1 - STOP_PCT), 4),
                "take_profit":       round(price * (1 + TARGET_PCT), 4),
                "position_size_usd": POSITION_SIZE,
                # Trailing stop metadata
                "trailing_stop_trigger_pct": TRAIL_TRIGGER,
                "trailing_stop_pct":         TRAIL_PCT,
                # Context
                "vol_ratio":         round(vol_ratio, 2),
                "vwap":              round(bar_vwap, 4),
                "vwap_diff_pct":     round(vwap_diff_pct * 100, 2),
                "rsi":               round(rsi_val, 2) if rsi_val else None,
                "sentiment_score":   round(sent_score, 4),
                "asset_type":        "stock",
            })

    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals[:open_slots]
