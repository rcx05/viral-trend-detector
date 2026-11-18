"""
Viral Trend Detector — YouTube Data API v3 Fetcher
Free quota: 10,000 units/day.
API key: https://console.cloud.google.com → Enable YouTube Data API v3
"""

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import requests


YT_SEARCH_URL   = "https://www.googleapis.com/youtube/v3/search"
YT_VIDEOS_URL   = "https://www.googleapis.com/youtube/v3/videos"
YT_TRENDING_URL = "https://www.googleapis.com/youtube/v3/videos"


def _api_key() -> Optional[str]:
    key = os.getenv("YOUTUBE_API_KEY", "")
    if not key or key == "your_youtube_api_key":
        return None
    return key


def fetch_youtube_mentions(
    keyword: str,
    days: int = 30,
    max_results: int = 50,
) -> Optional[pd.DataFrame]:
    """
    Search YouTube for videos matching a keyword and return daily upload counts.

    Args:
        keyword:    Search query / hashtag (without #)
        days:       How many days back to search
        max_results: Max videos per request (up to 50)

    Returns:
        DataFrame with DatetimeIndex and 'videos' + 'total_views' columns.
        None if API key missing.
    """
    key = _api_key()
    if not key:
        return None

    published_after = (
        datetime.now(tz=timezone.utc) - timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "part":           "snippet",
        "q":              keyword,
        "type":           "video",
        "order":          "date",
        "publishedAfter": published_after,
        "maxResults":     max_results,
        "key":            key,
    }

    try:
        resp = requests.get(YT_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items:
            return None

        # Fetch view counts for each video
        video_ids = [item["id"]["videoId"] for item in items if "videoId" in item.get("id", {})]
        view_map  = _fetch_view_counts(video_ids, key)

        rows = []
        for item in items:
            snippet = item.get("snippet", {})
            pub_at  = snippet.get("publishedAt", "")
            vid_id  = item.get("id", {}).get("videoId", "")
            if pub_at:
                date = pd.to_datetime(pub_at).normalize()
                rows.append({
                    "date":   date,
                    "title":  snippet.get("title", ""),
                    "views":  view_map.get(vid_id, 0),
                    "url":    f"https://youtube.com/watch?v={vid_id}",
                })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        daily = df.groupby("date").agg(
            videos      = ("title", "count"),
            total_views = ("views", "sum"),
        ).sort_index()
        return daily

    except requests.RequestException as e:
        print(f"[YouTube] Network error: {e}")
        return None
    except Exception as e:
        print(f"[YouTube] Error: {e}")
        return None


def _fetch_view_counts(video_ids: List[str], key: str) -> dict:
    """Batch-fetch view counts for a list of video IDs."""
    if not video_ids:
        return {}
    params = {
        "part": "statistics",
        "id":   ",".join(video_ids),
        "key":  key,
    }
    try:
        resp = requests.get(YT_VIDEOS_URL, params=params, timeout=10)
        resp.raise_for_status()
        return {
            item["id"]: int(item.get("statistics", {}).get("viewCount", 0))
            for item in resp.json().get("items", [])
        }
    except Exception:
        return {}


def fetch_youtube_trending(region_code: str = "IN") -> List[dict]:
    """
    Fetch currently trending videos on YouTube for a country.

    Args:
        region_code: ISO 3166-1 alpha-2 country code e.g. 'US', 'IN'

    Returns:
        List of video dicts with title, views, channel, url.
    """
    key = _api_key()
    if not key:
        return []

    params = {
        "part":        "snippet,statistics",
        "chart":       "mostPopular",
        "regionCode":  region_code,
        "maxResults":  20,
        "key":         key,
    }
    try:
        resp = requests.get(YT_TRENDING_URL, params=params, timeout=10)
        resp.raise_for_status()
        return [
            {
                "title":   item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "views":   int(item.get("statistics", {}).get("viewCount", 0)),
                "url":     f"https://youtube.com/watch?v={item['id']}",
            }
            for item in resp.json().get("items", [])
        ]
    except Exception as e:
        print(f"[YouTube Trending] Error: {e}")
        return []
