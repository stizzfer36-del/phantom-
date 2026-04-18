import json
import os
from datetime import datetime, timezone
from typing import Any

BANKROLL = 80.0

POOL_ALLOCATIONS: dict[str, float] = {
    "crypto_swing":     0.30,
    "penny_scanner":    0.25,
    "new_token_tracker": 0.20,
    "momentum_tracker": 0.15,
    "micro_tracker":    0.10,
}

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "portfolio_state.json")


def _state_path() -> str:
    return os.path.abspath(STATE_FILE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict[str, Any]:
    pools: dict[str, Any] = {}
    for name, pct in POOL_ALLOCATIONS.items():
        initial = round(BANKROLL * pct, 2)
        pools[name] = {
            "allocation_pct": pct,
            "initial_balance": initial,
            "cash": initial,
            "positions": {},
            "trades": [],
            "realized_pnl": 0.0,
        }
    return {
        "bankroll": BANKROLL,
        "pools": pools,
        "created_at": _now(),
        "updated_at": _now(),
    }


def load_state() -> dict[str, Any]:
    path = _state_path()
    if not os.path.exists(path):
        state = _default_state()
        _write(path, state)
        return state
    with open(path) as f:
        return json.load(f)


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    _write(_state_path(), state)


def _write(path: str, state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def get_pool_balance(state: dict[str, Any], pool: str) -> float:
    """Return total value of the pool: cash + unrealised position cost basis."""
    _require_pool(pool)
    p = state["pools"][pool]
    position_value = sum(
        pos["quantity"] * pos["entry_price"]
        for pos in p["positions"].values()
    )
    return round(p["cash"] + position_value, 6)


def update_positions(
    state: dict[str, Any],
    pool: str,
    positions: dict[str, dict[str, Any]],
) -> None:
    """Replace the positions dict for a pool. Caller owns validation."""
    _require_pool(pool)
    state["pools"][pool]["positions"] = positions


def open_position(
    state: dict[str, Any],
    pool: str,
    symbol: str,
    quantity: float,
    price: float,
    side: str = "long",
) -> dict[str, Any]:
    """Simulate buying into a position. Deducts cost from pool cash."""
    _require_pool(pool)
    p = state["pools"][pool]
    cost = round(quantity * price, 6)
    if cost > p["cash"]:
        raise ValueError(
            f"Insufficient cash in {pool}: need {cost:.4f}, have {p['cash']:.4f}"
        )
    p["cash"] = round(p["cash"] - cost, 6)
    pos = {
        "symbol": symbol,
        "quantity": quantity,
        "entry_price": price,
        "side": side,
        "opened_at": _now(),
    }
    p["positions"][symbol] = pos
    trade = {
        "action": "buy",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "cost": cost,
        "pnl": None,
        "timestamp": _now(),
    }
    p["trades"].append(trade)
    return pos


def close_position(
    state: dict[str, Any],
    pool: str,
    symbol: str,
    price: float,
) -> dict[str, Any]:
    """Simulate selling an entire position. Credits proceeds and records P&L."""
    _require_pool(pool)
    p = state["pools"][pool]
    if symbol not in p["positions"]:
        raise KeyError(f"No open position for {symbol!r} in pool {pool!r}")
    pos = p["positions"].pop(symbol)
    proceeds = round(pos["quantity"] * price, 6)
    cost_basis = round(pos["quantity"] * pos["entry_price"], 6)
    pnl = round(proceeds - cost_basis, 6)
    p["cash"] = round(p["cash"] + proceeds, 6)
    p["realized_pnl"] = round(p["realized_pnl"] + pnl, 6)
    trade = {
        "action": "sell",
        "symbol": symbol,
        "side": pos["side"],
        "quantity": pos["quantity"],
        "price": price,
        "proceeds": proceeds,
        "pnl": pnl,
        "timestamp": _now(),
    }
    p["trades"].append(trade)
    return trade


def record_trade(
    state: dict[str, Any],
    pool: str,
    trade: dict[str, Any],
) -> None:
    """Append an arbitrary trade record to a pool's history (manual override)."""
    _require_pool(pool)
    state["pools"][pool]["trades"].append(trade)


def pool_summary(state: dict[str, Any], pool: str) -> dict[str, Any]:
    """Return a snapshot of a pool's key metrics."""
    _require_pool(pool)
    p = state["pools"][pool]
    return {
        "pool": pool,
        "allocation_pct": p["allocation_pct"],
        "initial_balance": p["initial_balance"],
        "cash": p["cash"],
        "open_positions": len(p["positions"]),
        "total_balance": get_pool_balance(state, pool),
        "realized_pnl": p["realized_pnl"],
        "total_trades": len(p["trades"]),
    }


def portfolio_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Aggregate summary across all pools."""
    summaries = {p: pool_summary(state, p) for p in POOL_ALLOCATIONS}
    total_balance = sum(s["total_balance"] for s in summaries.values())
    total_realized_pnl = sum(s["realized_pnl"] for s in summaries.values())
    return {
        "bankroll": state["bankroll"],
        "total_balance": round(total_balance, 6),
        "total_realized_pnl": round(total_realized_pnl, 6),
        "pools": summaries,
        "updated_at": state["updated_at"],
    }


def _require_pool(pool: str) -> None:
    if pool not in POOL_ALLOCATIONS:
        raise KeyError(f"Unknown pool {pool!r}. Valid pools: {list(POOL_ALLOCATIONS)}")
