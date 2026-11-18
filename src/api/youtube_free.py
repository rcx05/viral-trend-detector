"""
Viral Trend Detector — API-Free YouTube Fetcher
Uses YouTube's public RSS feeds and oembed endpoint (no API key required).
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET


HEADERS = {
    "User-Agent": "ViralTrendDetector/1.0 (College Project)"
}

# YouTube RSS namespace
YT_NS  = {"yt": "http://www.youtube.com/xml/schemas/2015",
           "media": "http://search.yahoo.com/mrss/",
           "atom": "http://www.w3.org/2005/Atom"}


def fetch_youtube_mentions_free(
    keyword: str,
    max_results: int = 50,
) -> Optional[pd.DataFrame]:
    """
    Search YouTube via scraping the search results page or RSS,
    and count daily video uploads mentioning the keyword.
    No API key required.

    Uses the invidious API (public YouTube proxy) as a fallback.
    """
    # Try multiple public Invidious instances
    instances = [
        "https://vid.puffyan.us",
        "https://invidious.snopyta.org",
        "https://inv.nadeko.net",
        "https://invidious.nerdvpn.de",
    ]

    for base_url in instances:
        try:
            url = f"{base_url}/api/v1/search"
            params = {
                "q":     keyword,
                "sort":  "upload_date",
                "type":  "video",
            }
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data:
                continue

            rows = []
            for video in data[:max_results]:
                if video.get("type") != "video":
                    continue
                published = video.get("published", 0)
                if published:
                    date = pd.to_datetime(datetime.utcfromtimestamp(published)).normalize()
                    rows.append({
                        "date":  date,
                        "title": video.get("title", ""),
                        "views": video.get("viewCount", 0),
                        "url":   f"https://youtube.com/watch?v={video.get('videoId', '')}",
                    })

            if not rows:
                continue

            df = pd.DataFrame(rows)
            daily = df.groupby("date").agg(
                videos      = ("title", "count"),
                total_views = ("views", "sum"),
            ).sort_index()
            return daily

        except Exception as e:
            continue

    # All instances failed — return None gracefully
    print("[YouTube Free] All Invidious instances unavailable")
    return None


def fetch_youtube_trending_free(region_code: str = "IN") -> List[dict]:
    """
    Fetch trending videos using public Invidious API.
    No API key required.
    """
    instances = [
        "https://vid.puffyan.us",
        "https://inv.nadeko.net",
        "https://invidious.nerdvpn.de",
    ]

    for base_url in instances:
        try:
            url = f"{base_url}/api/v1/trending"
            params = {"region": region_code}
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            return [
                {
                    "title":   v.get("title", ""),
                    "channel": v.get("author", ""),
                    "views":   v.get("viewCount", 0),
                    "url":     f"https://youtube.com/watch?v={v.get('videoId', '')}",
                }
                for v in data[:20]
                if v.get("type") == "video"
            ]
        except Exception:
            continue
    return []
