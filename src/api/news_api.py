"""
Viral Trend Detector — NewsAPI Fetcher
Free tier: 100 requests/day, last 30 days of news.
API key: https://newsapi.org/register
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException


def _build_client() -> Optional[NewsApiClient]:
    key = os.getenv("NEWS_API_KEY", "")
    if not key or key == "your_newsapi_key":
        return None
    return NewsApiClient(api_key=key)


def fetch_news_mentions(
    keyword: str,
    days: int = 30,
    language: str = "en",
    page_size: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Fetch article counts mentioning a keyword, grouped by day.

    Args:
        keyword:   Search query (supports AND, OR, NOT, quotes)
        days:      How many days back to fetch (max 30 on free tier)
        language:  Language code e.g. 'en', 'hi'
        page_size: Articles per page (max 100)

    Returns:
        DataFrame with DatetimeIndex and 'articles' column.
        None if credentials missing.
    """
    client = _build_client()
    if client is None:
        return None

    from_date = (datetime.now() - timedelta(days=min(days, 30))).strftime("%Y-%m-%d")
    to_date   = datetime.now().strftime("%Y-%m-%d")

    try:
        response = client.get_everything(
            q           = keyword,
            from_param  = from_date,
            to          = to_date,
            language    = language,
            sort_by     = "publishedAt",
            page_size   = page_size,
        )
        articles = response.get("articles", [])
        if not articles:
            return None

        rows = []
        for article in articles:
            pub_at = article.get("publishedAt", "")
            if pub_at:
                date = pd.to_datetime(pub_at).normalize()
                rows.append({
                    "date":   date,
                    "title":  article.get("title", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "url":    article.get("url", ""),
                })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        daily = df.groupby("date").agg(articles=("title", "count")).sort_index()
        return daily

    except NewsAPIException as e:
        print(f"[NewsAPI] Error: {e}")
        return None
    except Exception as e:
        print(f"[NewsAPI] Unexpected error: {e}")
        return None


def fetch_top_news(keyword: str, limit: int = 10) -> List[dict]:
    """
    Fetch the top news articles for a keyword.

    Returns:
        List of article dicts with title, source, url, publishedAt.
    """
    client = _build_client()
    if client is None:
        return []
    try:
        response = client.get_everything(
            q=keyword, sort_by="relevancy", page_size=limit, language="en"
        )
        return [
            {
                "title":       a.get("title", ""),
                "source":      a.get("source", {}).get("name", ""),
                "url":         a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "description": a.get("description", ""),
            }
            for a in response.get("articles", [])[:limit]
        ]
    except Exception as e:
        print(f"[NewsAPI Top] Error: {e}")
        return []
