"""
Viral Trend Detector — Google Trends API Fetcher
Uses pytrends (no API key required).
"""

import time
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from typing import List, Optional


def _build_client(retries: int = 3) -> TrendReq:
    """Build a pytrends client with backoff-ready settings."""
    return TrendReq(hl="en-US", tz=330, timeout=(10, 25), retries=retries, backoff_factor=0.5)


def fetch_google_trends(
    keywords: List[str],
    timeframe: str = "today 3-m",
    geo: str = "",
) -> Optional[pd.DataFrame]:
    """
    Fetch interest-over-time data from Google Trends.

    Args:
        keywords: Up to 5 keywords / hashtags
        timeframe: e.g. 'today 1-m', 'today 3-m', 'today 12-m', 'now 7-d'
        geo:       Country code e.g. 'US', 'IN'. Empty = worldwide.

    Returns:
        DataFrame indexed by date with one column per keyword (0–100 scale).
        Returns None on failure.
    """
    keywords = keywords[:5]  # Google Trends max 5 at a time
    pytrends = _build_client()

    try:
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if df.empty:
            return None
        # Drop the 'isPartial' column if present
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        return df
    except TooManyRequestsError:
        time.sleep(5)
        return None
    except Exception as e:
        print(f"[Google Trends] Error: {e}")
        return None


def fetch_related_queries(keyword: str) -> dict:
    """
    Fetch rising & top related queries for a keyword.

    Returns:
        dict with 'top' and 'rising' DataFrames (may be None).
    """
    pytrends = _build_client()
    try:
        pytrends.build_payload([keyword], timeframe="today 3-m")
        related = pytrends.related_queries()
        return {
            "top":    related.get(keyword, {}).get("top"),
            "rising": related.get(keyword, {}).get("rising"),
        }
    except Exception as e:
        print(f"[Google Trends Related] Error: {e}")
        return {"top": None, "rising": None}


def fetch_trending_searches(geo: str = "united_states") -> List[str]:
    """
    Fetch today's trending searches for a country.

    Args:
        geo: Country name e.g. 'united_states', 'india'

    Returns:
        List of trending search topics.
    """
    pytrends = _build_client()
    try:
        df = pytrends.trending_searches(pn=geo)
        return df[0].tolist()[:20]
    except Exception as e:
        print(f"[Google Trends Trending] Error: {e}")
        return []
