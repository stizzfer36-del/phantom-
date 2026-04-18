"""
Sentiment analysis from NewsAPI headlines and Reddit posts.

Reddit scraping uses the public JSON API (no auth required).
Scoring uses bullish/bearish keyword matching against post titles
and headlines.

Required env vars:
  NEWS_API_KEY  – NewsAPI.org key (free tier: 100 req/day)
"""

import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_REDDIT_URL  = "https://www.reddit.com/r/{subreddit}/hot.json"
_TIMEOUT     = 12
_REDDIT_UA   = "python:phantom-sim:0.1 (market analysis tool)"

DEFAULT_SUBREDDITS = ["wallstreetbets", "pennystocks", "cryptocurrency", "solana"]

_BULLISH = {
    "buy", "long", "bull", "bullish", "moon", "rocket", "pump", "breakout",
    "hold", "strong", "upside", "gain", "profit", "rally", "surge", "squeeze",
    "ath", "undervalued", "accumulate", "support", "bounce", "recovery",
    "explosive", "potential", "gem", "opportunity", "calls",
}

_BEARISH = {
    "sell", "short", "bear", "bearish", "dump", "crash", "drop", "down",
    "weak", "loss", "decline", "fail", "plunge", "collapse", "overvalued",
    "resistance", "correction", "bubble", "overbought", "puts", "rug",
    "scam", "avoid", "caution", "warning", "risk", "exit",
}

_TOKEN_RE = re.compile(r"[a-zA-Z$#@]+")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    return [t.lower().strip("$#@") for t in _TOKEN_RE.findall(text) if len(t) > 1]


def score_text(text: str) -> dict[str, Any]:
    """
    Score a single piece of text for bullish/bearish sentiment.

    Returns:
        {
          "score": float,       # (bullish - bearish) / total_signals; 0.0 when no signals
          "bullish": int,       # count of bullish keyword hits
          "bearish": int,       # count of bearish keyword hits
          "label": str,         # "bullish" | "bearish" | "neutral"
        }
    """
    words = _tokenise(text)
    bull = sum(1 for w in words if w in _BULLISH)
    bear = sum(1 for w in words if w in _BEARISH)
    total = bull + bear
    score = round((bull - bear) / total, 4) if total else 0.0
    label = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
    return {"score": score, "bullish": bull, "bearish": bear, "label": label}


def _aggregate_scores(scored_items: list[dict]) -> dict[str, Any]:
    """Aggregate a list of scored items into a single summary."""
    if not scored_items:
        return {"score": 0.0, "bullish": 0, "bearish": 0, "label": "neutral", "count": 0}
    total_bull = sum(x["bullish"] for x in scored_items)
    total_bear = sum(x["bearish"] for x in scored_items)
    total_sig  = total_bull + total_bear
    score = round((total_bull - total_bear) / total_sig, 4) if total_sig else 0.0
    label = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
    return {
        "score":   score,
        "bullish": total_bull,
        "bearish": total_bear,
        "label":   label,
        "count":   len(scored_items),
    }


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

def _fetch_reddit_posts(subreddit: str, limit: int = 25) -> list[dict] | None:
    """
    Fetch hot posts from a subreddit via the public Reddit JSON API.
    Returns a list of {title, score, url, subreddit} dicts or None on failure.
    """
    try:
        resp = httpx.get(
            _REDDIT_URL.format(subreddit=subreddit),
            params={"limit": limit},
            headers={"User-Agent": _REDDIT_UA},
            follow_redirects=True,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        children = resp.json()["data"]["children"]
        return [
            {
                "title":     c["data"]["title"],
                "score":     c["data"]["score"],
                "url":       c["data"]["url"],
                "subreddit": subreddit,
            }
            for c in children
            if c.get("data")
        ]
    except Exception as exc:
        logger.warning("_fetch_reddit_posts(%s) failed: %s", subreddit, exc)
        return None


def get_reddit_sentiment(
    subreddits: list[str] | None = None,
    limit: int = 25,
) -> dict[str, Any] | None:
    """
    Fetch and score posts from each subreddit.

    Returns:
        {
          "overall": {score, bullish, bearish, label, count},
          "by_subreddit": {subreddit: {score, bullish, bearish, label, count}},
          "posts": [{title, score, url, subreddit, sentiment}, ...],
        }
    or None if all subreddits fail.
    """
    targets = subreddits or DEFAULT_SUBREDDITS
    all_posts: list[dict] = []
    by_sub: dict[str, Any] = {}

    for sub in targets:
        posts = _fetch_reddit_posts(sub, limit=limit)
        if posts is None:
            continue
        scored = []
        for post in posts:
            sentiment = score_text(post["title"])
            post["sentiment"] = sentiment
            scored.append(sentiment)
            all_posts.append(post)
        by_sub[sub] = _aggregate_scores(scored)

    if not all_posts:
        logger.warning("get_reddit_sentiment: no posts retrieved from any subreddit")
        return None

    return {
        "overall":      _aggregate_scores([p["sentiment"] for p in all_posts]),
        "by_subreddit": by_sub,
        "posts":        all_posts,
    }


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

def _fetch_news(query: str, page_size: int = 20, language: str = "en") -> list[dict] | None:
    """
    Fetch recent headlines from NewsAPI for the given query string.
    Requires NEWS_API_KEY env var.
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        logger.warning("_fetch_news: NEWS_API_KEY not set")
        return None
    try:
        resp = httpx.get(
            _NEWSAPI_URL,
            params={
                "q":        query,
                "pageSize": page_size,
                "language": language,
                "sortBy":   "publishedAt",
                "apiKey":   api_key,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles") or []
        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "source":      a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "url":         a.get("url", ""),
            }
            for a in articles
        ]
    except Exception as exc:
        logger.warning("_fetch_news(%r) failed: %s", query, exc)
        return None


def get_news_sentiment(
    query: str,
    page_size: int = 20,
) -> dict[str, Any] | None:
    """
    Fetch and score news headlines for a query term (ticker, coin name, etc.).

    Returns:
        {
          "query": str,
          "overall": {score, bullish, bearish, label, count},
          "articles": [{title, description, source, published_at, url, sentiment}, ...],
        }
    or None on failure.
    """
    articles = _fetch_news(query, page_size=page_size)
    if articles is None:
        return None

    scored = []
    for article in articles:
        text = f"{article['title']} {article.get('description', '')}"
        sentiment = score_text(text)
        article["sentiment"] = sentiment
        scored.append(sentiment)

    return {
        "query":    query,
        "overall":  _aggregate_scores(scored),
        "articles": articles,
    }


# ---------------------------------------------------------------------------
# Combined signal
# ---------------------------------------------------------------------------

def get_combined_sentiment(
    symbol: str,
    subreddits: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Merge Reddit + News sentiment for a symbol/query into a single signal.

    Returns:
        {
          "symbol": str,
          "combined_score": float,   # weighted average (reddit 40%, news 60%)
          "label": str,
          "reddit": {...} | None,
          "news": {...} | None,
        }
    or None if both sources fail.
    """
    reddit = get_reddit_sentiment(subreddits)
    news   = get_news_sentiment(symbol)

    if reddit is None and news is None:
        logger.warning("get_combined_sentiment(%s): both sources failed", symbol)
        return None

    reddit_score = reddit["overall"]["score"] if reddit else 0.0
    news_score   = news["overall"]["score"]   if news   else 0.0

    # Weight news more heavily (60/40) since it's more directly targeted
    if reddit and news:
        combined = round(reddit_score * 0.4 + news_score * 0.6, 4)
    elif news:
        combined = news_score
    else:
        combined = reddit_score

    label = "bullish" if combined > 0 else ("bearish" if combined < 0 else "neutral")
    return {
        "symbol":         symbol,
        "combined_score": combined,
        "label":          label,
        "reddit":         reddit,
        "news":           news,
    }
