"""
Viral Trend Detector — API-Free News Fetcher
Uses Google News RSS feed (no API key required).
"""

import re
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET


HEADERS = {
    "User-Agent": "ViralTrendDetector/1.0 (College Project)"
}


def fetch_news_mentions_free(
    keyword: str,
    language: str = "en",
) -> Optional[pd.DataFrame]:
    """
    Fetch news article counts from Google News RSS feed.
    No API key required.

    Args:
        keyword:  Search query
        language: Language code e.g. 'en'

    Returns:
        DataFrame with DatetimeIndex and 'articles' column.
    """
    encoded = quote_plus(keyword)
    url = f"https://news.google.com/rss/search?q={encoded}&hl={language}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[News Free] Error fetching RSS: {e}")
        return None

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"[News Free] XML parse error: {e}")
        return None

    rows = []
    for item in root.iter("item"):
        title   = item.findtext("title", "")
        link    = item.findtext("link", "")
        source  = item.findtext("source", "")
        pub_str = item.findtext("pubDate", "")

        if pub_str:
            try:
                pub_date = pd.to_datetime(pub_str, utc=True).normalize()
                rows.append({
                    "date":   pub_date,
                    "title":  _clean_title(title),
                    "source": source,
                    "url":    link,
                })
            except Exception:
                continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    daily = df.groupby("date").agg(articles=("title", "count")).sort_index()
    # Remove timezone info to match other sources
    daily.index = daily.index.tz_localize(None)
    return daily


def fetch_top_news_free(keyword: str, limit: int = 10) -> List[dict]:
    """
    Fetch latest news articles from Google News RSS.
    No API key required.
    """
    encoded = quote_plus(keyword)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"[News Free Top] Error: {e}")
        return []

    results = []
    for item in list(root.iter("item"))[:limit]:
        title   = item.findtext("title", "")
        link    = item.findtext("link", "")
        source  = item.findtext("source", "")
        pub_str = item.findtext("pubDate", "")

        results.append({
            "title":        _clean_title(title),
            "source":       source,
            "url":          link,
            "published_at": pub_str[:16] if pub_str else "",
            "description":  "",
        })
    return results


def _clean_title(title: str) -> str:
    """Remove the ' - SourceName' suffix Google News appends."""
    parts = title.rsplit(" - ", 1)
    return parts[0].strip() if len(parts) > 1 else title.strip()
