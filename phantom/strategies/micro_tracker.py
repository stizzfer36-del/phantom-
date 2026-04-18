"""
Strategy: Micro Tracker
Pool:     micro_tracker  (10% of bankroll = $8)
Assets:   Solana tokens under 1 hour old  (via DexScreener)

Purpose: Study the early lifecycle of new tokens — observe price action,
liquidity growth, and typical outcome patterns from the first minutes.

Entry conditions:
  - Token age: < 1 hour
  - Liquidity ≥ $5,000
  - At least 1 on-chain transaction in the last 5 minutes (txns.m5 > 0)

Risk:
  - Position size  : $1–$2 (confidence-scaled)
  - Stop loss      : 40% below entry  (tokens can rug quickly)
  - Take profit    : 80% above entry
  - Max 4 open positions
  - Hard circuit breaker: stop trading if pool cash < $1
"""

import logging
import time
from typing import Any

import httpx

from phantom.data.prices import _DEXSCREENER, _TIMEOUT

logger = logging.getLogger(__name__)

POOL          = "micro_tracker"
STOP_PCT      = 0.40
TARGET_PCT    = 0.80
MAX_POSITIONS = 4
MAX_AGE_H     = 1.0
MIN_LIQUIDITY = 5_000
CIRCUIT_BREAKER_CASH = 1.00   # stop trading if pool cash drops below this

_MIN_POSITION = 1.00
_MAX_POSITION = 2.00


def _fetch_ultra_new_pairs() -> list[dict] | None:
    """Fetch Solana pairs and filter to those created within the last hour."""
    try:
        resp = httpx.get(
            f"{_DEXSCREENER}/search",
            params={"q": "solana"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        pairs     = resp.json().get("pairs") or []
        cutoff_ms = (time.time() - MAX_AGE_H * 3600) * 1_000
        return [
            p for p in pairs
            if p.get("chainId") == "solana"
            and p.get("pairCreatedAt") is not None
            and p["pairCreatedAt"] >= cutoff_ms
        ]
    except Exception as exc:
        logger.warning("micro_tracker._fetch_ultra_new_pairs: %s", exc)
        return None


def _open_position_count(state: dict, pool: str) -> int:
    return len(state["pools"][pool]["positions"])


def _already_held(state: dict, pool: str, token_address: str) -> bool:
    return f"token:{token_address}" in state["pools"][pool]["positions"]


def generate_signals(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Scan for very-new Solana tokens as lifecycle observation entries.

    Circuit breaker: returns [] immediately if pool cash < CIRCUIT_BREAKER_CASH.
    """
    pool_cash  = state["pools"][POOL]["cash"]
    open_slots = MAX_POSITIONS - _open_position_count(state, POOL)

    # Circuit breaker
    if pool_cash < CIRCUIT_BREAKER_CASH:
        logger.warning(
            "micro_tracker: CIRCUIT BREAKER — pool cash %.2f < $%.2f, no new entries",
            pool_cash, CIRCUIT_BREAKER_CASH,
        )
        return []

    if open_slots <= 0:
        logger.info("micro_tracker: pool full (%d positions)", MAX_POSITIONS)
        return []
    if pool_cash < _MIN_POSITION:
        logger.info("micro_tracker: insufficient cash %.2f", pool_cash)
        return []

    pairs = _fetch_ultra_new_pairs()
    if pairs is None:
        return []

    now_ms   = time.time() * 1_000
    signals: list[dict] = []

    for pair in pairs:
        base    = pair.get("baseToken", {})
        address = base.get("address", "")
        symbol  = base.get("symbol", "?")

        if not address or _already_held(state, POOL, address):
            continue

        liq   = float(pair.get("liquidity", {}).get("usd") or 0)
        price = float(pair.get("priceUsd") or 0)
        age_h = (now_ms - pair["pairCreatedAt"]) / 3_600_000
        age_m = age_h * 60

        if liq < MIN_LIQUIDITY or price <= 0:
            continue

        # Recent transaction activity (txns.m5 = buys+sells in last 5 min)
        txns_m5 = pair.get("txns", {}).get("m5", {})
        txn_count_5m = (int(txns_m5.get("buys") or 0) + int(txns_m5.get("sells") or 0))

        vol_5m  = float(pair.get("volume", {}).get("m5") or 0)
        vol_h1  = float(pair.get("volume", {}).get("h1") or 0)
        change  = pair.get("priceChange", {}).get("m5") or 0

        # ── Scoring ────────────────────────────────────────────────────────
        score   = 0
        reasons = []

        # Liquidity safety
        if liq >= 50_000:
            score += 30; reasons.append(f"liq ${liq/1000:.0f}k")
        elif liq >= 20_000:
            score += 22; reasons.append(f"liq ${liq/1000:.0f}k")
        elif liq >= 5_000:
            score += 12; reasons.append(f"liq ${liq/1000:.0f}k")

        # Recent activity (last 5 min)
        if txn_count_5m >= 20:
            score += 25; reasons.append(f"{txn_count_5m} txns/5min")
        elif txn_count_5m >= 5:
            score += 15; reasons.append(f"{txn_count_5m} txns/5min")
        elif txn_count_5m > 0:
            score += 5;  reasons.append(f"{txn_count_5m} txns/5min")

        # 5-min volume
        if vol_5m >= 10_000:
            score += 20; reasons.append(f"5m vol ${vol_5m:.0f}")
        elif vol_5m >= 1_000:
            score += 10; reasons.append(f"5m vol ${vol_5m:.0f}")

        # Freshness bonus
        if age_m <= 10:
            score += 20; reasons.append(f"very fresh ({age_m:.0f}min)")
        elif age_m <= 30:
            score += 10; reasons.append(f"fresh ({age_m:.0f}min)")
        else:
            reasons.append(f"age {age_m:.0f}min")

        # Positive 5-min price movement
        if change > 0:
            score += 5;  reasons.append(f"5m +{change:.1f}%")
        elif change < -10:
            score -= 10; reasons.append(f"5m {change:.1f}%")

        if score < 35:
            continue  # lower threshold — we're studying even marginal setups

        confidence = min(score, 95)
        pos_size   = round(
            _MIN_POSITION + (_MAX_POSITION - _MIN_POSITION) * (confidence / 95),
            2,
        )
        pos_size = min(pos_size, pool_cash)

        signals.append({
            "asset":            symbol,
            "token_address":    address,
            "direction":        "long",
            "pool":             POOL,
            "confidence":       confidence,
            "reason":           "; ".join(reasons),
            "entry_price":      round(price, 12),
            "stop_loss":        round(price * (1 - STOP_PCT), 12),
            "take_profit":      round(price * (1 + TARGET_PCT), 12),
            "position_size_usd": pos_size,
            # Lifecycle observation metadata
            "age_minutes":       round(age_m, 1),
            "liquidity_usd":     round(liq, 2),
            "volume_5m":         round(vol_5m, 2),
            "volume_1h":         round(vol_h1, 2),
            "txns_5m":           txn_count_5m,
            "change_5m_pct":     round(float(change), 2),
            "circuit_breaker_at": CIRCUIT_BREAKER_CASH,
            "asset_type":        "token",
        })

    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals[:open_slots]
