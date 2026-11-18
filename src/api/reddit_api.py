"""
Viral Trend Detector — Reddit API Fetcher
Uses PRAW (Python Reddit API Wrapper).
API keys: https://www.reddit.com/prefs/apps
"""

import os
import time
import pandas as pd
from datetime import datetime, timezone
from collections import Counter
from typing import List, Optional
import praw
from praw.exceptions import RedditAPIException


def _build_client() -> praw.Reddit:
    """Create a Reddit read-only client from environment variables."""
    return praw.Reddit(
        client_id     = os.getenv("REDDIT_CLIENT_ID", ""),
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", ""),
        user_agent    = os.getenv("REDDIT_USER_AGENT", "viral_trend_detector/1.0"),
        read_only     = True,
    )


def fetch_subreddit_trends(
    keyword: str,
    subreddits: List[str] = None,
    limit: int = 500,
    time_filter: str = "month",
) -> Optional[pd.DataFrame]:
    """
    Search Reddit for posts mentioning a keyword and return daily mention counts.

    Args:
        keyword:     Search term / hashtag (without #)
        subreddits:  List of subreddits to search (None = all of Reddit)
        limit:       Max posts to fetch
        time_filter: 'day','week','month','year','all'

    Returns:
        DataFrame with DatetimeIndex and 'mentions' column.
        None if credentials are missing/invalid.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    if not client_id or client_id == "your_reddit_client_id":
        return None   # Credentials not configured

    try:
        reddit = _build_client()
        sub_str = "+".join(subreddits) if subreddits else "all"
        subreddit = reddit.subreddit(sub_str)

        posts = []
        for submission in subreddit.search(
            keyword, sort="new", time_filter=time_filter, limit=limit
        ):
            created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            posts.append({
                "date":   created.date(),
                "title":  submission.title,
                "score":  submission.score,
                "comments": submission.num_comments,
                "url":    submission.url,
            })
            time.sleep(0.05)  # be kind to the API

        if not posts:
            return None

        df = pd.DataFrame(posts)
        daily = df.groupby("date").agg(
            mentions   = ("title", "count"),
            avg_score  = ("score", "mean"),
            total_comments = ("comments", "sum"),
        ).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date").sort_index()
        return daily

    except RedditAPIException as e:
        print(f"[Reddit] API error: {e}")
        return None
    except Exception as e:
        print(f"[Reddit] Error: {e}")
        return None


def fetch_reddit_hot_topics(subreddit_name: str = "all", limit: int = 25) -> List[dict]:
    """
    Fetch currently hot posts from a subreddit.

    Returns:
        List of dicts with title, score, comments, url.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    if not client_id or client_id == "your_reddit_client_id":
        return []

    try:
        reddit = _build_client()
        subreddit = reddit.subreddit(subreddit_name)
        results = []
        for post in subreddit.hot(limit=limit):
            results.append({
                "title":    post.title,
                "score":    post.score,
                "comments": post.num_comments,
                "url":      f"https://reddit.com{post.permalink}",
                "subreddit": str(post.subreddit),
            })
        return results
    except Exception as e:
        print(f"[Reddit Hot] Error: {e}")
        return []
