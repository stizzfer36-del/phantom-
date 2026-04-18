# Phantom

A market analysis and **paper trading simulator** built in Python.  
Phantom scans live market data across stocks, major crypto, and Solana tokens,
generates trade signals from five independent strategies, and simulates execution
against a virtual $80 bankroll — all without risking real money.

---

## Security

**Real API keys must never be committed to this repository.**

- Copy `.env.example` to `.env` and fill in your keys. `.env` is gitignored; `.env.example` is not — keep it blank placeholders only.
- If you run Phantom inside Claude Code's web environment, any value you put in the environment-variables field is visible to anyone who has access to that environment. Use **paper-trading-only** Alpaca keys there — never live or production credentials.
- The NewsAPI key is sent as an HTTP header (`X-Api-Key`), not as a URL query parameter, so it cannot leak into server access logs or exception messages.
- Alpaca keys are sent as request headers (`APCA-API-KEY-ID` / `APCA-API-SECRET-KEY`), not embedded in URLs.
- If you suspect a key has been exposed, revoke it immediately at the issuing service and generate a new one.

---

## How it works

Run one cycle with:

```bash
python main.py
```

Each cycle does exactly this:

1. **Load state** from `portfolio_state.json`
2. **Refresh prices** on every open position and update unrealized P&L
3. **Auto-close** any position that has hit its stop-loss or take-profit
4. **Run all five strategies** to collect fresh signals
5. **Filter signals** — drop any that exceed position limits or available cash
6. **Execute simulated trades** with realistic slippage
7. **Save state** back to `portfolio_state.json`
8. **Print a cycle summary** showing portfolio value, pool breakdown, and all activity

State persists between runs, so you can schedule `main.py` as a cron job or run it manually whenever you want a new cycle.

---

## Setup

```bash
git clone <repo>
cd phantom-
pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your API keys
python main.py
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | For stocks | Alpaca paper-trading API key |
| `ALPACA_API_SECRET` | For stocks | Alpaca API secret |
| `NEWSAPI_KEY` | Optional | NewsAPI.org key (100 req/day free) |
| `STARTING_BALANCE` | No | Initial bankroll in USD (default `80`) |

**Free tiers that need no key:** CoinGecko (crypto prices), DexScreener (Solana tokens), Reddit public JSON API.

---

## Architecture

```
main.py                   ← entry point, one cycle per run

phantom/
  portfolio/
    manager.py            ← state engine: load/save, pool balances, P&L tracking

  sim/
    stock_sim.py          ← simulated stock/crypto trades via yfinance (0.5% slippage)
    token_sim.py          ← simulated Solana token trades via DexScreener (1.5/2.0%)

  data/
    prices.py             ← Alpaca (stocks), CoinGecko (crypto), DexScreener (tokens)
    scanner.py            ← scan_new_tokens, scan_trending_crypto,
                            scan_penny_stocks, scan_momentum
    sentiment.py          ← Reddit scraper + NewsAPI + keyword sentiment scoring
    indicators.py         ← RSI, EMA9/21, MACD, Bollinger Bands, ATR, VWAP, volume ratio

  strategies/
    crypto_swing.py       ← BTC/ETH/SOL swing entries
    penny_scanner.py      ← high-volume penny stock entries
    new_token_tracker.py  ← new Solana token momentum entries
    momentum_breakout.py  ← 20-day high breakout entries
    micro_tracker.py      ← ultra-new token lifecycle observation

portfolio_state.json      ← persisted simulation state (auto-created)
```

---

## Portfolio pools

The $80 bankroll is split into five isolated pools. Each pool runs one strategy
and tracks its own positions, cash, and P&L independently.

| Pool | Allocation | USD | Strategy |
|---|---|---|---|
| `crypto_swing` | 30% | $24 | BTC / ETH / SOL swing trades |
| `penny_scanner` | 25% | $20 | Penny stock volume spikes |
| `new_token_tracker` | 20% | $16 | New Solana token momentum |
| `momentum_tracker` | 15% | $12 | Stock 20-day breakouts |
| `micro_tracker` | 10% | $8 | Ultra-new token lifecycle study |

---

## Strategies

### crypto_swing — `crypto_swing` pool
Scans BTC, ETH, and SOL using 30-day CoinGecko OHLC data.

**Entry:** RSI(14) < 35 **and** price below lower Bollinger Band **and** news sentiment neutral or bullish  
**Exit signals:** RSI > 70 or price above upper BB  
**Risk:** 5% stop-loss · 8% take-profit · max 2 positions at up to 50% of pool each

---

### penny_scanner — `penny_scanner` pool
Scans US equities under $5 using Alpaca batch snapshots.

**Entry:** Volume ≥ 3× previous day **and** price within 3% of daily VWAP **and** bullish/neutral news  
**Risk:** 8% stop-loss · 15% take-profit · trailing stop activates at +8% (trails 4%) · $5 fixed per position · max 4

---

### new_token_tracker — `new_token_tracker` pool
Scans DexScreener for Solana tokens aged 1–24 hours.

**Entry:** Liquidity > $10k **and** 24h volume > $50k **and** 24h price change +30% to +500%  
**Risk:** 40% stop-loss · 100% first take-profit (sell half) · 30% trailing on remainder · $1–$3 per position · max 5

---

### momentum_breakout — `momentum_tracker` pool
Scans equities for 20-day high breakouts using Alpaca daily bars.

**Entry:** Close > 20-day high **and** volume ≥ 2× average **and** RSI > 55 **and** EMA9 > EMA21  
**Risk:** Stop-loss at 2× ATR(14) below entry · trailing stop at 2× ATR · $6 fixed per position · max 2

---

### micro_tracker — `micro_tracker` pool
Tracks Solana tokens under 1 hour old to study early lifecycle behaviour.

**Entry:** Liquidity > $5k **and** active transactions in the last 5 minutes  
**Risk:** 40% stop-loss · 80% take-profit · $1–$2 per position · max 4  
**Circuit breaker:** halts all new entries if pool cash drops below $1

---

## Data sources

| Source | Used for | Auth |
|---|---|---|
| Alpaca Data API | Stock prices, bars, snapshots | `ALPACA_API_KEY` + `ALPACA_API_SECRET` |
| CoinGecko (free) | Crypto prices, OHLC, trending | None |
| DexScreener | Solana token prices, new pairs | None |
| yfinance | Stock + crypto price execution | None |
| NewsAPI | Financial headlines | `NEWSAPI_KEY` |
| Reddit (public JSON) | WSB / pennystocks / crypto posts | None |

---

## Technical indicators

All implemented in `phantom/data/indicators.py` using **NumPy only** (no TA-Lib dependency).

- **RSI(14)** — Wilder smoothing
- **EMA(9)** and **EMA(21)**
- **MACD** — 12/26/9 with histogram
- **Bollinger Bands** — 20-period SMA ± 2σ with `%B` and bandwidth
- **ATR(14)** — Wilder smoothing, used for dynamic stop sizing
- **VWAP** — typical-price weighted
- **Volume ratio** — current bar vs N-period average

---

## Slippage model

| Asset type | Buy slippage | Sell slippage |
|---|---|---|
| Stocks / major crypto | 0.5% | 0.5% |
| Solana tokens | 1.5% | 2.0% |

---

## Resetting the simulation

Delete `portfolio_state.json` and re-run. A fresh $80 bankroll will be created.

```bash
rm portfolio_state.json
python main.py
```
