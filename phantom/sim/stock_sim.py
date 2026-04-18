"""
Simulated stock and major-crypto trading against live yfinance prices.
Slippage: 0.5% on both buys and sells.
"""

from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from phantom.portfolio.manager import open_position as _pm_open, record_trade, save_state

SLIPPAGE = 0.005  # 0.5%
SIM_TYPE = "stock"


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def get_price(symbol: str) -> float:
    """Return the latest market price for a stock or major-crypto ticker."""
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.last_price
    if price is None or price != price:  # None or NaN
        raise ValueError(f"Could not fetch price for {symbol!r}")
    return float(price)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _buy_price(market_price: float) -> float:
    return round(market_price * (1 + SLIPPAGE), 8)


def _sell_price(market_price: float) -> float:
    return round(market_price * (1 - SLIPPAGE), 8)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def simulate_buy(
    state: dict[str, Any],
    pool: str,
    symbol: str,
    quantity: float,
) -> dict[str, Any]:
    """
    Open a simulated long position.

    Applies 0.5% buy-side slippage. Raises ValueError if the pool has
    insufficient cash. Saves state before returning.
    """
    market_price = get_price(symbol)
    exec_price = _buy_price(market_price)
    cost = round(quantity * exec_price, 6)

    pool_data = state["pools"][pool]
    if cost > pool_data["cash"]:
        raise ValueError(
            f"Insufficient cash in {pool!r}: need {cost:.4f}, have {pool_data['cash']:.4f}"
        )

    pool_data["cash"] = round(pool_data["cash"] - cost, 6)
    pos = {
        "symbol": symbol,
        "quantity": quantity,
        "entry_price": exec_price,
        "current_price": exec_price,
        "unrealized_pnl": 0.0,
        "side": "long",
        "sim_type": SIM_TYPE,
        "opened_at": _now(),
    }
    pool_data["positions"][symbol] = pos

    trade = {
        "action": "buy",
        "symbol": symbol,
        "side": "long",
        "quantity": quantity,
        "market_price": market_price,
        "exec_price": exec_price,
        "slippage_pct": SLIPPAGE,
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
    symbol: str,
    quantity: float | None = None,
) -> dict[str, Any]:
    """
    Close all or part of an open position.

    quantity=None closes the full position. Applies 0.5% sell-side slippage.
    Updates realized P&L on the pool. Saves state before returning.
    """
    pool_data = state["pools"][pool]
    if symbol not in pool_data["positions"]:
        raise KeyError(f"No open position for {symbol!r} in pool {pool!r}")

    pos = pool_data["positions"][symbol]
    close_qty = quantity if quantity is not None else pos["quantity"]
    if close_qty > pos["quantity"]:
        raise ValueError(
            f"Cannot sell {close_qty} of {symbol!r}; only {pos['quantity']} held"
        )

    market_price = get_price(symbol)
    exec_price = _sell_price(market_price)
    proceeds = round(close_qty * exec_price, 6)
    cost_basis = round(close_qty * pos["entry_price"], 6)
    pnl = round(proceeds - cost_basis, 6)

    pool_data["cash"] = round(pool_data["cash"] + proceeds, 6)
    pool_data["realized_pnl"] = round(pool_data["realized_pnl"] + pnl, 6)

    remaining = round(pos["quantity"] - close_qty, 8)
    if remaining <= 0:
        pool_data["positions"].pop(symbol)
    else:
        pos["quantity"] = remaining

    trade = {
        "action": "sell",
        "symbol": symbol,
        "side": "long",
        "quantity": close_qty,
        "market_price": market_price,
        "exec_price": exec_price,
        "slippage_pct": SLIPPAGE,
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
    Return all open positions for the pool.

    With live=True (default), fetches the current market price for each
    position and recalculates unrealized P&L in-place.
    """
    positions = state["pools"][pool]["positions"]
    if not live:
        return positions

    for symbol, pos in positions.items():
        if pos.get("sim_type") != SIM_TYPE:
            continue
        try:
            current = get_price(symbol)
        except Exception:
            continue
        pos["current_price"] = current
        pos["unrealized_pnl"] = round(
            pos["quantity"] * (current - pos["entry_price"]), 6
        )

    return positions
