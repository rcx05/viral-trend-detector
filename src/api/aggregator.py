"""
Viral Trend Detector — Data Aggregator
Merges data from all API sources into a unified normalized series.
Auto-detects whether API keys exist: uses official APIs if available,
falls back to free public endpoints otherwise.
"""

import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .google_trends  import fetch_google_trends, fetch_related_queries, fetch_trending_searches
from .reddit_api     import fetch_subreddit_trends, fetch_reddit_hot_topics
from .reddit_free    import fetch_subreddit_trends_free, fetch_reddit_hot_free
from .news_api       import fetch_news_mentions, fetch_top_news
from .news_free      import fetch_news_mentions_free, fetch_top_news_free
from .youtube_api    import fetch_youtube_mentions, fetch_youtube_trending
from .youtube_free   import fetch_youtube_mentions_free, fetch_youtube_trending_free

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────── Helper: check if API key is configured ─────────────

def _has_key(env_var: str, placeholder: str) -> bool:
    val = os.getenv(env_var, "")
    return bool(val) and val != placeholder


def _reddit_has_key() -> bool:
    return _has_key("REDDIT_CLIENT_ID", "your_reddit_client_id")

def _news_has_key() -> bool:
    return _has_key("NEWS_API_KEY", "your_newsapi_key")

def _youtube_has_key() -> bool:
    return _has_key("YOUTUBE_API_KEY", "your_youtube_api_key")


# ───────────────────────── Caching Helpers ────────────────────────────────

def _cache_key(prefix: str, **kwargs) -> str:
    raw = json.dumps(kwargs, sort_keys=True)
    return f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:8]}"


def _load_cache(key: str, ttl_seconds: int = 300) -> Optional[pd.DataFrame]:
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
        if age < ttl_seconds:
            try:
                return pd.read_pickle(path)
            except Exception:
                pass
    return None


def _save_cache(key: str, df: pd.DataFrame):
    try:
        df.to_pickle(CACHE_DIR / f"{key}.pkl")
    except Exception:
        pass


# ───────────────────────── Source Fetchers ────────────────────────────────

def get_google_trend_series(keyword: str, timeframe: str = "today 3-m") -> Optional[pd.Series]:
    ckey = _cache_key("gt", keyword=keyword, timeframe=timeframe)
    cached = _load_cache(ckey)
    if cached is not None and keyword in cached.columns:
        return cached[keyword]

    df = fetch_google_trends([keyword], timeframe=timeframe)
    if df is None or keyword not in df.columns:
        return None
    _save_cache(ckey, df)
    return df[keyword]


def get_reddit_series(keyword: str) -> Optional[pd.Series]:
    ckey = _cache_key("reddit", keyword=keyword)
    cached = _load_cache(ckey)
    if cached is not None and "mentions" in cached.columns:
        return cached["mentions"]

    # Try official API first, then fallback to free
    if _reddit_has_key():
        df = fetch_subreddit_trends(keyword)
    else:
        df = fetch_subreddit_trends_free(keyword)

    if df is None or "mentions" not in df.columns:
        return None
    _save_cache(ckey, df)
    return df["mentions"]


def get_news_series(keyword: str) -> Optional[pd.Series]:
    ckey = _cache_key("news", keyword=keyword)
    cached = _load_cache(ckey)
    if cached is not None and "articles" in cached.columns:
        return cached["articles"]

    # Try official API first, then fallback to free
    if _news_has_key():
        df = fetch_news_mentions(keyword)
    else:
        df = fetch_news_mentions_free(keyword)

    if df is None or "articles" not in df.columns:
        return None
    _save_cache(ckey, df)
    return df["articles"]


def get_youtube_series(keyword: str) -> Optional[pd.Series]:
    ckey = _cache_key("yt", keyword=keyword)
    cached = _load_cache(ckey)
    if cached is not None and "videos" in cached.columns:
        return cached["videos"]

    # Try official API first, then fallback to free
    if _youtube_has_key():
        df = fetch_youtube_mentions(keyword)
    else:
        df = fetch_youtube_mentions_free(keyword)

    if df is None or "videos" not in df.columns:
        return None
    _save_cache(ckey, df)
    return df["videos"]


def _get_reddit_hot(limit: int = 10) -> List[dict]:
    if _reddit_has_key():
        return fetch_reddit_hot_topics("all", limit=limit)
    return fetch_reddit_hot_free("all", limit=limit)


def _get_top_news(keyword: str, limit: int = 5) -> List[dict]:
    if _news_has_key():
        return fetch_top_news(keyword, limit=limit)
    return fetch_top_news_free(keyword, limit=limit)


# ───────────────────────── Unified Aggregator ─────────────────────────────

def aggregate_sources(
    keyword: str,
    sources: List[str] = None,
    timeframe: str = "today 3-m",
) -> Dict[str, Any]:
    """
    Fetch data from all selected sources and return:
      - per-source Series (normalized 0–100)
      - combined weighted series
      - source availability flags
      - raw supplemental data

    Automatically uses free endpoints when API keys are not configured.

    Args:
        keyword:  Search keyword / hashtag
        sources:  List from ['google_trends', 'reddit', 'news', 'youtube']
        timeframe: Google Trends timeframe string

    Returns:
        Dict with 'combined', 'sources', 'available', 'supplemental'
    """
    if sources is None:
        sources = ["google_trends", "reddit", "news", "youtube"]

    result = {
        "keyword":      keyword,
        "sources":      {},
        "available":    {},
        "combined":     None,
        "supplemental": {},
    }

    all_series = []

    # ── Google Trends ────────────────────────────────────────────────────
    if "google_trends" in sources:
        clean_kw = keyword.lstrip("#")
        s = get_google_trend_series(clean_kw, timeframe)
        result["available"]["google_trends"] = s is not None
        if s is not None:
            # Already 0–100 scale
            s = s.astype(float)
            result["sources"]["google_trends"] = s
            all_series.append(("google_trends", s, 1.0))

            # Related queries
            rq = fetch_related_queries(clean_kw)
            result["supplemental"]["related_queries"] = rq

    # ── Reddit ───────────────────────────────────────────────────────────
    if "reddit" in sources:
        clean_kw = keyword.lstrip("#")
        s = get_reddit_series(clean_kw)
        result["available"]["reddit"] = s is not None
        if s is not None:
            s = _normalize(s)
            result["sources"]["reddit"] = s
            all_series.append(("reddit", s, 0.8))

            hot = _get_reddit_hot(limit=10)
            result["supplemental"]["reddit_hot"] = hot

    # ── News ──────────────────────────────────────────────────────────────
    if "news" in sources:
        clean_kw = keyword.lstrip("#")
        s = get_news_series(clean_kw)
        result["available"]["news"] = s is not None
        if s is not None:
            s = _normalize(s)
            result["sources"]["news"] = s
            all_series.append(("news", s, 0.7))

            top_news = _get_top_news(clean_kw, limit=5)
            result["supplemental"]["top_news"] = top_news

    # ── YouTube ───────────────────────────────────────────────────────────
    if "youtube" in sources:
        clean_kw = keyword.lstrip("#")
        s = get_youtube_series(clean_kw)
        result["available"]["youtube"] = s is not None
        if s is not None:
            s = _normalize(s)
            result["sources"]["youtube"] = s
            all_series.append(("youtube", s, 0.9))

    # ── Combine ───────────────────────────────────────────────────────────
    if all_series:
        result["combined"] = _weighted_combine(all_series)

    return result


def _normalize(s: pd.Series) -> pd.Series:
    """Scale a series to 0–100."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    return ((s - mn) / (mx - mn) * 100).astype(float)


def _weighted_combine(series_list) -> pd.Series:
    """
    Align all series to a common date index and compute a weighted average.
    """
    # Build a full date range spanning all series
    all_dates = pd.DatetimeIndex([])
    for _, s, _ in series_list:
        all_dates = all_dates.union(s.index)

    combined = pd.Series(0.0, index=all_dates)
    weight_sum = pd.Series(0.0, index=all_dates)

    for _, s, w in series_list:
        reindexed = s.reindex(all_dates, fill_value=0)
        combined  += reindexed * w
        weight_sum += w

    result = combined / weight_sum.replace(0, 1)
    return result.sort_index()


def get_trending_topics() -> Dict[str, List[str]]:
    """
    Pull currently trending topics from Google Trends.
    """
    return {
        "worldwide": fetch_trending_searches("united_states"),
        "india":     fetch_trending_searches("india"),
    }
