"""
Market scanners:
  scan_new_tokens      – DexScreener Solana pairs < 24 h old, filtered by liquidity
  scan_trending_crypto – CoinGecko trending coins
  scan_penny_stocks    – US equities under $5 at ≥ 3× previous-day volume
  scan_momentum        – US equities breaking 20-day highs on above-average volume
"""

import logging
import time
from typing import Any

import httpx

from phantom.data.prices import (
    _alpaca_headers,
    _ALPACA_DATA,
    _COINGECKO,
    _DEXSCREENER,
    _TIMEOUT,
    get_active_equity_symbols,
    get_stock_bars,
    get_stock_snapshots,
)

logger = logging.getLogger(__name__)

_24H_MS = 24 * 60 * 60 * 1_000

# Fallback universe when Alpaca assets endpoint is unavailable
_DEFAULT_EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD", "INTC", "NFLX",
    "ORCL", "CRM", "ADBE", "PYPL", "SQ", "SNAP", "PINS", "UBER", "LYFT", "COIN",
    "HOOD", "SOFI", "PLTR", "RIVN", "LCID", "NIO", "F", "GM", "T", "VZ",
    "AMC", "GME", "BB", "NOK", "CLOV", "SPCE", "WISH", "SKLZ", "RKT", "OPEN",
    "SNDL", "ACB", "CGC", "TLRY", "HEXO", "MVIS", "EYES", "IDEX", "ILUS", "ENDP",
]


# ---------------------------------------------------------------------------
# DexScreener – new Solana tokens
# ---------------------------------------------------------------------------

def scan_new_tokens(
    min_liquidity_usd: float = 5_000,
    max_age_hours: float = 24,
) -> list[dict] | None:
    """
    Return Solana trading pairs created within max_age_hours that have at
    least min_liquidity_usd in liquidity, sorted by 24-hour volume descending.

    Uses the DexScreener search endpoint (no API key required).
    """
    try:
        resp = httpx.get(
            f"{_DEXSCREENER}/search",
            params={"q": "solana"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        all_pairs: list[dict] = resp.json().get("pairs") or []

        cutoff_ms = (time.time() - max_age_hours * 3600) * 1_000
        results = []
        for pair in all_pairs:
            if pair.get("chainId") != "solana":
                continue
            created_at = pair.get("pairCreatedAt")
            if not created_at or created_at < cutoff_ms:
                continue
            liq = float(pair.get("liquidity", {}).get("usd") or 0)
            if liq < min_liquidity_usd:
                continue
            base = pair.get("baseToken", {})
            results.append({
                "address":       base.get("address", ""),
                "symbol":        base.get("symbol", ""),
                "name":          base.get("name", ""),
                "price_usd":     float(pair.get("priceUsd") or 0),
                "liquidity_usd": liq,
                "volume_24h":    float(pair.get("volume", {}).get("h24") or 0),
                "change_24h":    pair.get("priceChange", {}).get("h24"),
                "pair_address":  pair.get("pairAddress", ""),
                "pair_created_at": created_at,
                "age_hours":     round((time.time() * 1_000 - created_at) / 3_600_000, 2),
            })

        results.sort(key=lambda x: x["volume_24h"], reverse=True)
        logger.info("scan_new_tokens: %d results (liq>$%s, age<%sh)", len(results), min_liquidity_usd, max_age_hours)
        return results
    except Exception as exc:
        logger.warning("scan_new_tokens failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CoinGecko – trending crypto
# ---------------------------------------------------------------------------

def scan_trending_crypto() -> list[dict] | None:
    """
    Return CoinGecko's current trending coins (top-7 by search volume in
    the past 24 hours), enriched with price and 24-hour change.
    """
    try:
        resp = httpx.get(f"{_COINGECKO}/search/trending", timeout=_TIMEOUT)
        resp.raise_for_status()
        coins = resp.json().get("coins") or []

        results = []
        for entry in coins:
            item = entry.get("item", {})
            data = item.get("data", {})
            results.append({
                "id":            item.get("id"),
                "symbol":        item.get("symbol"),
                "name":          item.get("name"),
                "market_cap_rank": item.get("market_cap_rank"),
                "score":         item.get("score"),
                "price_usd":     data.get("price"),
                "change_24h":    data.get("price_change_percentage_24h", {}).get("usd") if isinstance(data.get("price_change_percentage_24h"), dict) else data.get("price_change_percentage_24h"),
                "market_cap":    data.get("market_cap"),
                "volume_24h":    data.get("total_volume"),
            })
        logger.info("scan_trending_crypto: %d trending coins", len(results))
        return results
    except Exception as exc:
        logger.warning("scan_trending_crypto failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Alpaca – penny stocks
# ---------------------------------------------------------------------------

def scan_penny_stocks(
    symbols: list[str] | None = None,
    max_price: float = 5.0,
    min_vol_ratio: float = 3.0,
    feed: str = "iex",
) -> list[dict] | None:
    """
    Return US equities priced below max_price whose today's volume is at
    least min_vol_ratio × the previous trading day's volume.

    symbols defaults to the top 500 active Alpaca equities (falling back to
    an internal watchlist if the assets endpoint is unavailable).
    """
    try:
        universe = symbols or get_active_equity_symbols(500) or _DEFAULT_EQUITY_UNIVERSE

        results = []
        # Alpaca snapshot batch limit is 100 per request
        for i in range(0, len(universe), 100):
            batch = universe[i : i + 100]
            snapshots = get_stock_snapshots(batch, feed=feed)
            if not snapshots:
                continue
            for sym, snap in snapshots.items():
                daily = snap.get("dailyBar") or {}
                prev  = snap.get("prevDailyBar") or {}

                price      = float(daily.get("c") or 0)
                today_vol  = float(daily.get("v") or 0)
                prev_vol   = float(prev.get("v") or 0)

                if price <= 0 or price > max_price:
                    continue
                if prev_vol <= 0:
                    continue
                vol_ratio = today_vol / prev_vol
                if vol_ratio < min_vol_ratio:
                    continue

                results.append({
                    "symbol":    sym,
                    "price":     round(price, 4),
                    "today_vol": int(today_vol),
                    "prev_vol":  int(prev_vol),
                    "vol_ratio": round(vol_ratio, 2),
                    "change_pct": round(
                        (price - float(prev.get("c") or price)) / float(prev.get("c") or price) * 100, 2
                    ) if prev.get("c") else None,
                })

        results.sort(key=lambda x: x["vol_ratio"], reverse=True)
        logger.info("scan_penny_stocks: %d results (price<$%s, vol>%sx)", len(results), max_price, min_vol_ratio)
        return results
    except Exception as exc:
        logger.warning("scan_penny_stocks failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Alpaca – momentum (20-day high breakouts)
# ---------------------------------------------------------------------------

def scan_momentum(
    symbols: list[str] | None = None,
    lookback_days: int = 20,
    min_vol_ratio: float = 1.5,
    feed: str = "iex",
) -> list[dict] | None:
    """
    Return equities whose latest close is a lookback_days-day high AND whose
    today's volume exceeds min_vol_ratio × the lookback_days average.

    Fetches lookback_days + 1 daily bars per symbol; processes sequentially
    to respect Alpaca rate limits.
    """
    try:
        universe = symbols or _DEFAULT_EQUITY_UNIVERSE

        results = []
        for sym in universe:
            bars = get_stock_bars(sym, timeframe="1Day", limit=lookback_days + 1, feed=feed)
            if not bars or len(bars) < lookback_days + 1:
                continue

            closes  = [float(b["c"]) for b in bars]
            volumes = [float(b["v"]) for b in bars]

            latest_close  = closes[-1]
            latest_vol    = volumes[-1]
            period_high   = max(closes[:-1])          # high of the prior N days
            avg_vol       = sum(volumes[:-1]) / len(volumes[:-1])

            if latest_close <= period_high:
                continue
            if avg_vol <= 0 or latest_vol / avg_vol < min_vol_ratio:
                continue

            results.append({
                "symbol":       sym,
                "close":        round(latest_close, 4),
                "period_high":  round(period_high, 4),
                "breakout_pct": round((latest_close - period_high) / period_high * 100, 2),
                "today_vol":    int(latest_vol),
                "avg_vol":      int(avg_vol),
                "vol_ratio":    round(latest_vol / avg_vol, 2),
                "lookback_days": lookback_days,
            })

        results.sort(key=lambda x: x["breakout_pct"], reverse=True)
        logger.info("scan_momentum: %d breakouts from %d symbols", len(results), len(universe))
        return results
    except Exception as exc:
        logger.warning("scan_momentum failed: %s", exc)
        return None
