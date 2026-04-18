"""
Strategy: New Token Tracker
Pool:     new_token_tracker  (20% of bankroll)
Assets:   Solana tokens aged 1–24 hours  (via DexScreener)

Entry conditions (all required):
  - Token age: 1–24 hours old
  - Liquidity ≥ $10,000
  - 24h volume ≥ $50,000
  - 24h price change: +30% to +500%  (momentum without mania)

Risk:
  - Stop loss      : 40% below entry
  - Take profit    : 100% above entry (take half at 100%)
  - Trailing stop  : 30% below peak after first TP leg hit
  - Position size  : $1–$3 per trade (scales with confidence)
  - Max positions  : 5
"""

import logging
import time
from typing import Any

import httpx

from phantom.data.prices import _DEXSCREENER, _TIMEOUT

logger = logging.getLogger(__name__)

POOL          = "new_token_tracker"
STOP_PCT      = 0.40
FIRST_TP_PCT  = 1.00   # 100% gain → take half
TRAIL_PCT     = 0.30   # then trail remaining by 30% from peak
MAX_POSITIONS = 5
MIN_LIQUIDITY = 10_000
MIN_VOLUME    = 50_000
MIN_AGE_H     = 1.0
MAX_AGE_H     = 24.0
MIN_CHANGE    = 30.0   # % 24h price change
MAX_CHANGE    = 500.0  # % — avoid tokens that already mooned

_MIN_POSITION = 1.00
_MAX_POSITION = 3.00


def _fetch_new_pairs(max_age_h: float = MAX_AGE_H) -> list[dict] | None:
    """Pull recent Solana pairs from DexScreener search."""
    try:
        resp = httpx.get(
            f"{_DEXSCREENER}/search",
            params={"q": "solana"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        pairs = resp.json().get("pairs") or []
        cutoff_ms = (time.time() - max_age_h * 3600) * 1_000
        return [
            p for p in pairs
            if p.get("chainId") == "solana"
            and p.get("pairCreatedAt") is not None
            and p["pairCreatedAt"] >= cutoff_ms
        ]
    except Exception as exc:
        logger.warning("new_token_tracker._fetch_new_pairs: %s", exc)
        return None


def _open_position_count(state: dict, pool: str) -> int:
    return len(state["pools"][pool]["positions"])


def _already_held(state: dict, pool: str, token_address: str) -> bool:
    key = f"token:{token_address}"
    return key in state["pools"][pool]["positions"]


def generate_signals(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Scan DexScreener for newly launched Solana tokens with strong early momentum.

    Returns up to (MAX_POSITIONS - open_positions) signals, sorted by confidence.
    """
    pool_cash  = state["pools"][POOL]["cash"]
    open_slots = MAX_POSITIONS - _open_position_count(state, POOL)

    if open_slots <= 0:
        logger.info("new_token_tracker: pool full (%d positions)", MAX_POSITIONS)
        return []
    if pool_cash < _MIN_POSITION:
        logger.info("new_token_tracker: insufficient cash %.2f", pool_cash)
        return []

    pairs = _fetch_new_pairs()
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

        liq    = float(pair.get("liquidity", {}).get("usd") or 0)
        vol    = float(pair.get("volume", {}).get("h24") or 0)
        change = pair.get("priceChange", {}).get("h24")
        price  = float(pair.get("priceUsd") or 0)
        age_h  = (now_ms - pair["pairCreatedAt"]) / 3_600_000

        if liq < MIN_LIQUIDITY or vol < MIN_VOLUME or price <= 0:
            continue
        if change is None:
            continue
        change = float(change)
        if not (MIN_CHANGE <= change <= MAX_CHANGE):
            continue
        if not (MIN_AGE_H <= age_h <= MAX_AGE_H):
            continue

        # ── Scoring ────────────────────────────────────────────────────────
        score   = 0
        reasons = []

        # Liquidity depth
        if liq >= 100_000:
            score += 30; reasons.append(f"liq ${liq/1000:.0f}k")
        elif liq >= 50_000:
            score += 22; reasons.append(f"liq ${liq/1000:.0f}k")
        elif liq >= 10_000:
            score += 12; reasons.append(f"liq ${liq/1000:.0f}k")

        # Volume strength
        if vol >= 500_000:
            score += 30; reasons.append(f"vol ${vol/1000:.0f}k")
        elif vol >= 200_000:
            score += 22; reasons.append(f"vol ${vol/1000:.0f}k")
        elif vol >= 50_000:
            score += 12; reasons.append(f"vol ${vol/1000:.0f}k")

        # Price momentum (sweet spot: meaningful move but not exhausted)
        if 100 <= change <= 300:
            score += 25; reasons.append(f"+{change:.0f}% 24h (strong)")
        elif 50 <= change < 100:
            score += 18; reasons.append(f"+{change:.0f}% 24h")
        elif 30 <= change < 50:
            score += 10; reasons.append(f"+{change:.0f}% 24h")
        elif change > 300:
            score += 5;  reasons.append(f"+{change:.0f}% (extended)")

        # Age bonus — fresher tokens catch more of the move
        if age_h <= 3:
            score += 15; reasons.append(f"very new ({age_h:.1f}h)")
        elif age_h <= 8:
            score += 8;  reasons.append(f"new ({age_h:.1f}h)")
        else:
            reasons.append(f"age {age_h:.1f}h")

        # Volume/liquidity ratio — healthy turnover signal
        vol_liq = vol / liq if liq else 0
        if vol_liq >= 5:
            score += 10; reasons.append(f"vol/liq {vol_liq:.1f}×")

        if score < 45:
            continue

        confidence = min(score, 95)

        # Scale position size $1–$3 with confidence
        pos_size = round(
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
            "take_profit":      round(price * (1 + FIRST_TP_PCT), 12),
            "position_size_usd": pos_size,
            # Multi-leg TP metadata
            "tp1_pct":          FIRST_TP_PCT,        # sell 50% here
            "tp1_qty_pct":      0.50,
            "trailing_stop_pct": TRAIL_PCT,
            # Context
            "age_hours":        round(age_h, 2),
            "liquidity_usd":    round(liq, 2),
            "volume_24h":       round(vol, 2),
            "change_24h_pct":   round(change, 2),
            "vol_liq_ratio":    round(vol_liq, 2),
            "asset_type":       "token",
        })

    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals[:open_slots]
