#!/usr/bin/env python3
"""
Phantom – Market Analysis & Paper Trading Simulator
────────────────────────────────────────────────────
Run once per cycle:

  python main.py

Each cycle:
  1. Load portfolio state from portfolio_state.json
  2. Refresh prices on all open positions; update unrealized P&L
  3. Auto-close positions that hit stop-loss or take-profit
  4. Run all 5 strategy scanners to collect fresh signals
  5. Filter signals (capacity, cash, duplicate checks)
  6. Execute qualifying signals as simulated trades
  7. Persist updated state
  8. Print a full cycle summary
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Load .env before touching any os.getenv calls
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually if not installed

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,         # set to INFO/DEBUG for verbose output
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("phantom")

# ── Portfolio ──────────────────────────────────────────────────────────────
from phantom.portfolio.manager import (
    load_state,
    save_state,
    portfolio_summary,
    POOL_ALLOCATIONS,
)

# ── Simulators ─────────────────────────────────────────────────────────────
from phantom.sim.stock_sim import (
    simulate_buy  as _stock_buy,
    simulate_sell as _stock_sell,
    get_price     as _stock_price,
    SLIPPAGE      as _STOCK_SLIPPAGE,
)
from phantom.sim.token_sim import (
    simulate_buy  as _token_buy,
    simulate_sell as _token_sell,
    get_price     as _token_price,
    SLIPPAGE_BUY  as _TOKEN_SLIPPAGE,
)

# ── Strategies ─────────────────────────────────────────────────────────────
from phantom.strategies.crypto_swing    import generate_signals as _crypto_swing_signals
from phantom.strategies.penny_scanner   import generate_signals as _penny_scanner_signals
from phantom.strategies.new_token_tracker import generate_signals as _new_token_signals
from phantom.strategies.momentum_breakout import generate_signals as _momentum_signals
from phantom.strategies.micro_tracker   import generate_signals as _micro_tracker_signals

# Map pool → (strategy fn, max open positions)
_STRATEGIES: list[tuple[str, Any, int]] = [
    ("crypto_swing",      _crypto_swing_signals,   2),
    ("penny_scanner",     _penny_scanner_signals,  4),
    ("new_token_tracker", _new_token_signals,      5),
    ("momentum_tracker",  _momentum_signals,       2),
    ("micro_tracker",     _micro_tracker_signals,  4),
]


# ── Position management ────────────────────────────────────────────────────

def _live_price(pos: dict) -> float | None:
    """Fetch current market price for any position type."""
    try:
        if pos.get("sim_type") == "stock":
            return _stock_price(pos["symbol"])
        if pos.get("sim_type") == "token":
            return _token_price(pos["token_address"])
    except Exception as exc:
        logger.warning("price fetch failed for %s: %s", pos.get("symbol", "?"), exc)
    return None


def refresh_positions(state: dict) -> list[dict]:
    """
    Update current_price / unrealized_pnl on every open position.
    Auto-close positions that have hit their stop_loss or take_profit.

    Returns a list of close records for cycle reporting.
    """
    closes: list[dict] = []

    for pool_name, pool in state["pools"].items():
        for pos_key, pos in list(pool["positions"].items()):
            price = _live_price(pos)
            if price is None:
                continue

            pos["current_price"] = price
            pos["unrealized_pnl"] = round(
                pos["quantity"] * (price - pos["entry_price"]), 6
            )

            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            reason = None
            if sl is not None and price <= sl:
                reason = "stop_loss"
            elif tp is not None and price >= tp:
                reason = "take_profit"

            if reason is None:
                continue

            try:
                if pos["sim_type"] == "stock":
                    trade = _stock_sell(state, pool_name, pos["symbol"])
                else:
                    trade = _token_sell(state, pool_name, pos["token_address"])
                trade.update({"close_reason": reason, "pool": pool_name})
                closes.append(trade)
                logger.info(
                    "[%s] auto-closed %s via %s pnl=%.4f",
                    pool_name, pos.get("symbol", pos_key), reason, trade.get("pnl", 0),
                )
            except Exception as exc:
                logger.warning("auto-close failed for %s/%s: %s", pool_name, pos_key, exc)

    return closes


# ── Signal collection ──────────────────────────────────────────────────────

def collect_signals(state: dict) -> list[dict]:
    """Run every strategy and merge their signals into one sorted list."""
    all_signals: list[dict] = []
    for pool_name, strategy_fn, _ in _STRATEGIES:
        try:
            sigs = strategy_fn(state)
            all_signals.extend(sigs or [])
            logger.info("[%s] %d signal(s)", pool_name, len(sigs or []))
        except Exception as exc:
            logger.warning("[%s] strategy error: %s", pool_name, exc)
    # Highest confidence first
    all_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)
    return all_signals


def filter_signals(signals: list[dict], state: dict) -> list[dict]:
    """
    Drop signals that:
    - Target a pool already at max open positions
    - Target a pool with insufficient cash for the position
    - Duplicate an asset already held in the same pool
    """
    # Build current slot counts to track within-cycle capacity
    slot_used: dict[str, int] = {
        pool: len(state["pools"][pool]["positions"])
        for pool in POOL_ALLOCATIONS
    }
    max_slots: dict[str, int] = {
        pool: max_pos for pool, _, max_pos in _STRATEGIES
    }

    kept: list[dict] = []
    for sig in signals:
        pool = sig["pool"]
        pool_data = state["pools"][pool]

        if slot_used.get(pool, 0) >= max_slots.get(pool, 4):
            continue
        if pool_data["cash"] < sig.get("position_size_usd", 0):
            continue

        # Duplicate check (stock key = ticker, token key = "token:{address}")
        if sig.get("asset_type") == "token":
            pos_key = f"token:{sig.get('token_address', '')}"
        else:
            pos_key = sig["asset"]
        if pos_key in pool_data["positions"]:
            continue

        kept.append(sig)
        slot_used[pool] = slot_used.get(pool, 0) + 1  # reserve slot optimistically

    return kept


# ── Trade execution ────────────────────────────────────────────────────────

def execute_signal(state: dict, signal: dict) -> dict | None:
    """
    Execute one signal as a simulated trade.

    Quantity is derived from position_size_usd with a slippage buffer so
    the sim's internal cost check doesn't reject the order.
    Stop-loss and take-profit are injected into the position after opening.

    Returns an execution record or None on failure.
    """
    pool     = signal["pool"]
    asset    = signal["asset"]
    pos_usd  = signal["position_size_usd"]
    entry    = signal["entry_price"]
    atype    = signal.get("asset_type", "stock")

    if entry <= 0:
        return None

    # Reserve a slippage buffer so the sim never rejects on cost overflow
    slippage_buffer = _TOKEN_SLIPPAGE if atype == "token" else _STOCK_SLIPPAGE
    safe_usd = min(pos_usd, state["pools"][pool]["cash"]) * (1 - slippage_buffer - 0.005)
    quantity  = safe_usd / entry

    if quantity <= 0:
        return None

    try:
        if atype == "token":
            pos = _token_buy(
                state, pool,
                token_address=signal["token_address"],
                quantity=quantity,
                symbol=signal.get("display", asset),
            )
            pos_key = f"token:{signal['token_address']}"
        else:
            pos = _stock_buy(state, pool, asset, quantity=quantity)
            pos_key = asset

        # Persist SL / TP inside the position dict for auto-close logic
        live_pos = state["pools"][pool]["positions"].get(pos_key)
        if live_pos is not None:
            live_pos["stop_loss"]   = signal["stop_loss"]
            live_pos["take_profit"] = signal["take_profit"]

        display = signal.get("display", asset)
        logger.info(
            "[%s] opened %-10s qty=%.6g entry=%.6g sl=%.6g tp=%.6g $%.2f",
            pool, display, quantity, pos.get("entry_price", entry),
            signal["stop_loss"], signal["take_profit"], pos_usd,
        )
        return {
            "action":           "opened",
            "asset":            asset,
            "display":          signal.get("display", asset),
            "pool":             pool,
            "quantity":         quantity,
            "entry_price":      pos.get("entry_price", entry),
            "stop_loss":        signal["stop_loss"],
            "take_profit":      signal["take_profit"],
            "position_size_usd": pos_usd,
            "confidence":       signal.get("confidence", 0),
            "reason":           signal.get("reason", ""),
        }

    except Exception as exc:
        logger.warning("execute_signal failed for %s/%s: %s", pool, asset, exc)
        return None


# ── Summary printing ───────────────────────────────────────────────────────

_W = 66   # terminal width for the summary box

def _pnl_str(v: float) -> str:
    return f"{'+' if v >= 0 else ''}${v:.4f}"


def print_summary(state: dict, closes: list[dict], opens: list[dict]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary = portfolio_summary(state)

    total_val   = summary["total_balance"]
    bankroll    = summary["bankroll"]
    realized    = summary["total_realized_pnl"]
    unrealized  = sum(
        pos.get("unrealized_pnl", 0)
        for pool in state["pools"].values()
        for pos in pool["positions"].values()
    )
    net_pnl = realized + unrealized

    print()
    print("═" * _W)
    print(f"  PHANTOM  ·  {now}")
    print("═" * _W)
    print(f"  Portfolio     ${total_val:.2f}   "
          f"(bankroll ${bankroll:.2f}  net {_pnl_str(net_pnl)})")
    print(f"  Realized P&L  {_pnl_str(realized)}    "
          f"Unrealized  {_pnl_str(unrealized)}")
    print("─" * _W)

    for pool_name in POOL_ALLOCATIONS:
        ps       = summary["pools"][pool_name]
        positions = state["pools"][pool_name]["positions"]
        pool_unr  = sum(p.get("unrealized_pnl", 0) for p in positions.values())
        n_pos     = ps["open_positions"]
        alloc_pct = int(ps["allocation_pct"] * 100)

        print(f"\n  [{pool_name}]  {alloc_pct}%  "
              f"${ps['total_balance']:.2f}  "
              f"(${ps['cash']:.2f} cash · {n_pos} open)")
        print(f"    realized {_pnl_str(ps['realized_pnl'])}  "
              f"unrealized {_pnl_str(pool_unr)}  "
              f"trades {ps['total_trades']}")

        for key, pos in positions.items():
            sym   = pos.get("symbol", key)
            qty   = pos.get("quantity", 0)
            ep    = pos.get("entry_price", 0)
            cp    = pos.get("current_price", ep)
            upnl  = pos.get("unrealized_pnl", 0)
            sl    = pos.get("stop_loss")
            tp    = pos.get("take_profit")
            sl_s  = f"{sl:.6g}" if sl is not None else "—"
            tp_s  = f"{tp:.6g}" if tp is not None else "—"
            print(f"    ▸ {sym:<14} qty={qty:.5g}  "
                  f"in={ep:.5g}  now={cp:.5g}  "
                  f"P&L {_pnl_str(upnl)}  sl={sl_s}  tp={tp_s}")

    print()
    print("─" * _W)
    cycle_events = len(closes) + len(opens)
    print(f"  Cycle activity: {cycle_events} event(s)")

    for c in closes:
        sym    = c.get("symbol", "?")
        reason = c.get("close_reason", "?")
        pnl    = c.get("pnl", 0)
        print(f"    ✕ CLOSED  {sym:<14} [{c['pool']:<20}]  "
              f"{reason:<12}  P&L {_pnl_str(pnl)}")

    for o in opens:
        display = o.get("display", o.get("asset", "?"))
        reason  = o.get("reason", "")[:48]
        print(f"    ✓ OPENED  {display:<14} [{o['pool']:<20}]  "
              f"${o['position_size_usd']:.2f}  conf={o['confidence']}  {reason}")

    if cycle_events == 0:
        print("    (no trades — market conditions did not trigger signals)")

    print("═" * _W)
    print()


# ── Main cycle ─────────────────────────────────────────────────────────────

def run_cycle() -> None:
    logger.info("=== Phantom cycle start ===")

    # 1. Load state
    state = load_state()
    logger.info("State loaded. Pools: %s",
                {p: len(state["pools"][p]["positions"]) for p in POOL_ALLOCATIONS})

    # 2. Refresh prices + 3. Auto-close SL/TP hits
    closes = refresh_positions(state)
    if closes:
        save_state(state)  # checkpoint after closes before running strategies

    # 4+5. Collect and filter signals
    signals  = collect_signals(state)
    filtered = filter_signals(signals, state)
    logger.info("%d raw signal(s) → %d after filtering", len(signals), len(filtered))

    # 6. Execute trades
    opens: list[dict] = []
    for sig in filtered:
        result = execute_signal(state, sig)
        if result:
            opens.append(result)

    # 7. Persist final state
    save_state(state)

    # 8. Print summary
    print_summary(state, closes, opens)
    logger.info("=== Phantom cycle complete ===")


if __name__ == "__main__":
    run_cycle()
