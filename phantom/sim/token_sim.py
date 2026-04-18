"""
Simulated Solana token trading against live DexScreener prices.
Slippage: 1.5% on buys, 2.0% on sells.
"""

from datetime import datetime, timezone
from typing import Any

import httpx

from phantom.portfolio.manager import save_state

SLIPPAGE_BUY = 0.015   # 1.5%
SLIPPAGE_SELL = 0.020  # 2.0%
SIM_TYPE = "token"

_DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{address}"
_REQUEST_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def get_price(token_address: str) -> float:
    """
    Fetch the USD price of a Solana token via DexScreener.

    Uses the highest-liquidity Solana trading pair for the given token address.
    Raises ValueError if no Solana pair is found or price is unavailable.
    """
    url = _DEXSCREENER_TOKEN_URL.format(address=token_address)
    resp = httpx.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    pairs = resp.json().get("pairs") or []

    sol_pairs = [
        p for p in pairs
        if p.get("chainId") == "solana" and p.get("priceUsd")
    ]
    if not sol_pairs:
        raise ValueError(
            f"No Solana trading pairs found for token address {token_address!r}"
        )

    best = max(
        sol_pairs,
        key=lambda p: float(p.get("liquidity", {}).get("usd") or 0),
    )
    return float(best["priceUsd"])


def get_token_info(token_address: str) -> dict[str, Any]:
    """
    Return the best Solana pair metadata (symbol, name, liquidity, volume)
    for a token address.
    """
    url = _DEXSCREENER_TOKEN_URL.format(address=token_address)
    resp = httpx.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    pairs = resp.json().get("pairs") or []

    sol_pairs = [p for p in pairs if p.get("chainId") == "solana" and p.get("priceUsd")]
    if not sol_pairs:
        raise ValueError(f"No Solana pairs found for {token_address!r}")

    best = max(sol_pairs, key=lambda p: float(p.get("liquidity", {}).get("usd") or 0))
    base = best.get("baseToken", {})
    return {
        "address": token_address,
        "symbol": base.get("symbol", "UNKNOWN"),
        "name": base.get("name", ""),
        "pair_address": best.get("pairAddress", ""),
        "price_usd": float(best["priceUsd"]),
        "liquidity_usd": float(best.get("liquidity", {}).get("usd") or 0),
        "volume_24h": float(best.get("volume", {}).get("h24") or 0),
    }


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _buy_price(market_price: float) -> float:
    return round(market_price * (1 + SLIPPAGE_BUY), 12)


def _sell_price(market_price: float) -> float:
    return round(market_price * (1 - SLIPPAGE_SELL), 12)


def _position_key(token_address: str) -> str:
    return f"token:{token_address}"


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def simulate_buy(
    state: dict[str, Any],
    pool: str,
    token_address: str,
    quantity: float,
    symbol: str | None = None,
) -> dict[str, Any]:
    """
    Open a simulated Solana token position.

    Fetches a live DexScreener price and applies 1.5% buy slippage.
    symbol is optional display name; if omitted it is resolved from DexScreener.
    Saves state before returning.
    """
    if symbol is None:
        info = get_token_info(token_address)
        symbol = info["symbol"]
        market_price = info["price_usd"]
    else:
        market_price = get_price(token_address)

    exec_price = _buy_price(market_price)
    cost = round(quantity * exec_price, 6)

    pool_data = state["pools"][pool]
    if cost > pool_data["cash"]:
        raise ValueError(
            f"Insufficient cash in {pool!r}: need {cost:.6f}, have {pool_data['cash']:.6f}"
        )

    pool_data["cash"] = round(pool_data["cash"] - cost, 6)
    key = _position_key(token_address)
    pos = {
        "symbol": symbol,
        "token_address": token_address,
        "quantity": quantity,
        "entry_price": exec_price,
        "current_price": exec_price,
        "unrealized_pnl": 0.0,
        "side": "long",
        "sim_type": SIM_TYPE,
        "opened_at": _now(),
    }
    pool_data["positions"][key] = pos

    trade = {
        "action": "buy",
        "symbol": symbol,
        "token_address": token_address,
        "side": "long",
        "quantity": quantity,
        "market_price": market_price,
        "exec_price": exec_price,
        "slippage_pct": SLIPPAGE_BUY,
        "cost": cost,
        "pnl": None,
        "sim_type": SIM_TYPE,
        "timestamp": _now(),
    }
    pool_data["trades"].append(trade)
    save_state(state)
    return pos


def simulate_sell(
    state: dict[str, Any],
    pool: str,
    token_address: str,
    quantity: float | None = None,
) -> dict[str, Any]:
    """
    Close all or part of a Solana token position.

    quantity=None closes the full position. Applies 2.0% sell slippage.
    Updates realized P&L on the pool. Saves state before returning.
    """
    pool_data = state["pools"][pool]
    key = _position_key(token_address)
    if key not in pool_data["positions"]:
        raise KeyError(f"No open token position for {token_address!r} in pool {pool!r}")

    pos = pool_data["positions"][key]
    close_qty = quantity if quantity is not None else pos["quantity"]
    if close_qty > pos["quantity"]:
        raise ValueError(
            f"Cannot sell {close_qty} of {pos['symbol']!r}; only {pos['quantity']} held"
        )

    market_price = get_price(token_address)
    exec_price = _sell_price(market_price)
    proceeds = round(close_qty * exec_price, 6)
    cost_basis = round(close_qty * pos["entry_price"], 6)
    pnl = round(proceeds - cost_basis, 6)

    pool_data["cash"] = round(pool_data["cash"] + proceeds, 6)
    pool_data["realized_pnl"] = round(pool_data["realized_pnl"] + pnl, 6)

    remaining = round(pos["quantity"] - close_qty, 8)
    if remaining <= 0:
        pool_data["positions"].pop(key)
    else:
        pos["quantity"] = remaining

    trade = {
        "action": "sell",
        "symbol": pos["symbol"],
        "token_address": token_address,
        "side": "long",
        "quantity": close_qty,
        "market_price": market_price,
        "exec_price": exec_price,
        "slippage_pct": SLIPPAGE_SELL,
        "proceeds": proceeds,
        "pnl": pnl,
        "sim_type": SIM_TYPE,
        "timestamp": _now(),
    }
    pool_data["trades"].append(trade)
    save_state(state)
    return trade


def get_positions(
    state: dict[str, Any],
    pool: str,
    live: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Return all open token positions for the pool.

    With live=True (default), fetches the current price for each position
    from DexScreener and updates unrealized P&L in-place.
    """
    positions = {
        k: v for k, v in state["pools"][pool]["positions"].items()
        if v.get("sim_type") == SIM_TYPE
    }
    if not live:
        return positions

    for pos in positions.values():
        try:
            current = get_price(pos["token_address"])
        except Exception:
            continue
        pos["current_price"] = current
        pos["unrealized_pnl"] = round(
            pos["quantity"] * (current - pos["entry_price"]), 6
        )

    return positions
