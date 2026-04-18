"""
Live price feeds:
  - Alpaca Data API  →  stocks (bars, snapshots, latest quotes)
  - CoinGecko free API  →  major crypto
  - DexScreener API  →  Solana tokens
"""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_ALPACA_DATA  = "https://data.alpaca.markets/v2"
_ALPACA_TRADE = "https://api.alpaca.markets/v2"
_COINGECKO    = "https://api.coingecko.com/api/v3"
_DEXSCREENER  = "https://api.dexscreener.com/latest/dex"
_TIMEOUT      = 12


# ---------------------------------------------------------------------------
# Alpaca helpers
# ---------------------------------------------------------------------------

def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", ""),
    }


def _alpaca_get(path: str, *, base: str = _ALPACA_DATA, params: dict | None = None) -> Any:
    resp = httpx.get(
        f"{base}{path}",
        headers=_alpaca_headers(),
        params=params or {},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Stock prices
# ---------------------------------------------------------------------------

def get_stock_price(symbol: str, feed: str = "iex") -> float | None:
    """Latest ask price for a US equity via Alpaca."""
    try:
        data = _alpaca_get(f"/stocks/{symbol}/quotes/latest", params={"feed": feed})
        return float(data["quote"]["ap"])
    except Exception as exc:
        logger.warning("get_stock_price(%s) failed: %s", symbol, exc)
        return None


def get_stock_bars(
    symbol: str,
    timeframe: str = "1Day",
    limit: int = 50,
    feed: str = "iex",
) -> list[dict] | None:
    """Historical OHLCV bars for a single stock symbol."""
    try:
        data = _alpaca_get(
            f"/stocks/{symbol}/bars",
            params={"timeframe": timeframe, "limit": limit, "feed": feed, "sort": "asc"},
        )
        return data.get("bars", [])
    except Exception as exc:
        logger.warning("get_stock_bars(%s) failed: %s", symbol, exc)
        return None


def get_stock_snapshot(symbol: str, feed: str = "iex") -> dict | None:
    """
    Full snapshot for one symbol: latestTrade, latestQuote, minuteBar,
    dailyBar, prevDailyBar.
    """
    try:
        return _alpaca_get(f"/stocks/{symbol}/snapshot", params={"feed": feed})
    except Exception as exc:
        logger.warning("get_stock_snapshot(%s) failed: %s", symbol, exc)
        return None


def get_stock_snapshots(symbols: list[str], feed: str = "iex") -> dict[str, dict] | None:
    """Batch snapshots for up to 100 symbols."""
    try:
        return _alpaca_get(
            "/stocks/snapshots",
            params={"symbols": ",".join(symbols), "feed": feed},
        )
    except Exception as exc:
        logger.warning("get_stock_snapshots failed for %d symbols: %s", len(symbols), exc)
        return None


def get_active_equity_symbols(limit: int = 500) -> list[str] | None:
    """Pull active, tradable US equity symbols from Alpaca (for scan universes)."""
    try:
        data = _alpaca_get(
            "/assets",
            base=_ALPACA_TRADE,
            params={"status": "active", "asset_class": "us_equity", "tradable": "true"},
        )
        symbols = [a["symbol"] for a in data if a.get("tradable")]
        return symbols[:limit]
    except Exception as exc:
        logger.warning("get_active_equity_symbols failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Crypto prices  (CoinGecko)
# ---------------------------------------------------------------------------

def get_crypto_price(coin_id: str, vs_currency: str = "usd") -> float | None:
    """Latest price for a single CoinGecko coin ID (e.g. 'bitcoin')."""
    try:
        resp = httpx.get(
            f"{_COINGECKO}/simple/price",
            params={"ids": coin_id, "vs_currencies": vs_currency},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return float(resp.json()[coin_id][vs_currency])
    except Exception as exc:
        logger.warning("get_crypto_price(%s) failed: %s", coin_id, exc)
        return None


def get_crypto_prices(
    coin_ids: list[str],
    vs_currency: str = "usd",
    include_24h_change: bool = False,
) -> dict[str, Any] | None:
    """
    Batch price lookup for multiple CoinGecko coin IDs.
    Returns {coin_id: price} or {coin_id: {price, change_24h}} when
    include_24h_change=True.
    """
    try:
        params: dict[str, Any] = {
            "ids": ",".join(coin_ids),
            "vs_currencies": vs_currency,
        }
        if include_24h_change:
            params["include_24hr_change"] = "true"

        resp = httpx.get(f"{_COINGECKO}/simple/price", params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()

        if not include_24h_change:
            return {cid: float(raw[cid][vs_currency]) for cid in coin_ids if cid in raw}

        return {
            cid: {
                "price":     float(raw[cid][vs_currency]),
                "change_24h": raw[cid].get(f"{vs_currency}_24h_change"),
            }
            for cid in coin_ids
            if cid in raw
        }
    except Exception as exc:
        logger.warning("get_crypto_prices failed: %s", exc)
        return None


def get_crypto_market_data(coin_id: str) -> dict | None:
    """
    Extended market data for a coin: price, market cap, 24h volume,
    24h change, 7d change.
    """
    try:
        resp = httpx.get(
            f"{_COINGECKO}/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": coin_id,
                "price_change_percentage": "24h,7d",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json()
        if not items:
            raise ValueError(f"No market data for {coin_id!r}")
        d = items[0]
        return {
            "id":          d["id"],
            "symbol":      d["symbol"],
            "name":        d["name"],
            "price":       d["current_price"],
            "market_cap":  d["market_cap"],
            "volume_24h":  d["total_volume"],
            "change_24h":  d.get("price_change_percentage_24h"),
            "change_7d":   d.get("price_change_percentage_7d_in_currency"),
        }
    except Exception as exc:
        logger.warning("get_crypto_market_data(%s) failed: %s", coin_id, exc)
        return None


# ---------------------------------------------------------------------------
# Solana token prices  (DexScreener)
# ---------------------------------------------------------------------------

def _best_sol_pair(pairs: list[dict]) -> dict | None:
    """Return highest-liquidity Solana pair from a DexScreener pairs list."""
    sol = [p for p in pairs if p.get("chainId") == "solana" and p.get("priceUsd")]
    if not sol:
        return None
    return max(sol, key=lambda p: float(p.get("liquidity", {}).get("usd") or 0))


def get_token_price(token_address: str) -> float | None:
    """USD price of a Solana token via DexScreener (highest-liquidity pair)."""
    try:
        resp = httpx.get(f"{_DEXSCREENER}/tokens/{token_address}", timeout=_TIMEOUT)
        resp.raise_for_status()
        pair = _best_sol_pair(resp.json().get("pairs") or [])
        if not pair:
            raise ValueError(f"No Solana pairs for {token_address!r}")
        return float(pair["priceUsd"])
    except Exception as exc:
        logger.warning("get_token_price(%s) failed: %s", token_address, exc)
        return None


def get_token_info(token_address: str) -> dict | None:
    """
    Extended info for a Solana token: symbol, name, price, liquidity,
    24h volume, 24h price change, pair address.
    """
    try:
        resp = httpx.get(f"{_DEXSCREENER}/tokens/{token_address}", timeout=_TIMEOUT)
        resp.raise_for_status()
        pair = _best_sol_pair(resp.json().get("pairs") or [])
        if not pair:
            raise ValueError(f"No Solana pairs for {token_address!r}")
        base = pair.get("baseToken", {})
        return {
            "address":      token_address,
            "symbol":       base.get("symbol", "UNKNOWN"),
            "name":         base.get("name", ""),
            "price_usd":    float(pair["priceUsd"]),
            "liquidity_usd": float(pair.get("liquidity", {}).get("usd") or 0),
            "volume_24h":   float(pair.get("volume", {}).get("h24") or 0),
            "change_24h":   pair.get("priceChange", {}).get("h24"),
            "pair_address": pair.get("pairAddress", ""),
            "pair_created_at": pair.get("pairCreatedAt"),
        }
    except Exception as exc:
        logger.warning("get_token_info(%s) failed: %s", token_address, exc)
        return None
