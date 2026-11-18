"""
Viral Trend Detector — API-Free Reddit Fetcher
Uses Reddit's public JSON endpoints (no authentication required).
Appends .json to Reddit URLs to get structured data.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List


HEADERS = {
    "User-Agent": "ViralTrendDetector/1.0 (College Project; No Auth)"
}


def fetch_subreddit_trends_free(
    keyword: str,
    limit: int = 100,
    sort: str = "new",
) -> Optional[pd.DataFrame]:
    """
    Search Reddit for posts mentioning a keyword using public JSON API.
    No API key required.

    Args:
        keyword:  Search term
        limit:    Max posts per request (max 100 per page, we do multiple pages)
        sort:     'new', 'relevance', 'hot', 'top', 'comments'

    Returns:
        DataFrame with DatetimeIndex and 'mentions' column.
    """
    all_posts = []
    after = None
    pages = 3  # Fetch up to 3 pages (300 posts)

    for _ in range(pages):
        url = "https://www.reddit.com/search.json"
        params = {
            "q":     keyword,
            "sort":  sort,
            "limit": min(limit, 100),
            "t":     "month",        # time filter: month
        }
        if after:
            params["after"] = after

        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                time.sleep(3)
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[Reddit Free] Error: {e}")
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            created = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)
            all_posts.append({
                "date":      created.date(),
                "title":     post.get("title", ""),
                "score":     post.get("score", 0),
                "comments":  post.get("num_comments", 0),
                "subreddit": post.get("subreddit", ""),
                "url":       f"https://reddit.com{post.get('permalink', '')}",
            })

        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(1.5)  # Rate limit: be polite

    if not all_posts:
        return None

    df = pd.DataFrame(all_posts)
    daily = df.groupby("date").agg(
        mentions       = ("title", "count"),
        avg_score      = ("score", "mean"),
        total_comments = ("comments", "sum"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index()
    return daily


def fetch_reddit_hot_free(subreddit: str = "all", limit: int = 25) -> List[dict]:
    """
    Fetch hot posts from a subreddit using public JSON API.
    No API key required.
    """
    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    params = {"limit": limit}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[Reddit Hot Free] Error: {e}")
        return []

    results = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        if post.get("stickied"):
            continue
        results.append({
            "title":     post.get("title", ""),
            "score":     post.get("score", 0),
            "comments":  post.get("num_comments", 0),
            "subreddit": post.get("subreddit", ""),
            "url":       f"https://reddit.com{post.get('permalink', '')}",
        })
    return results
