"""
generate_data.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA GENERATOR — builds social_media_trends.csv

Uses the Kaggle "Viral Social Media Trends & Engagement Analysis"
dataset via kagglehub.  If Kaggle credentials are not configured,
falls back to generating a REALISTIC synthetic dataset that
mirrors real viral dynamics (not flat random noise):

Real-world characteristics reproduced:
  • Long-tail distributions (Pareto)
  • Weekly seasonality (weekday boost)
  • Organic slow-build followed by viral explosion
  • Decay curves after peak (power law)
  • Multiple simultaneous hashtag trends
  • Sentiment data correlated with trend phase

Output: data/social_media_trends.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

np.random.seed(42)

DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "social_media_trends.csv")
Path(DATA_DIR).mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1.  Try Kaggle download first
# ════════════════════════════════════════════════════════════════════════════

def try_kaggle() -> bool:
    """Returns True if data was successfully downloaded from Kaggle."""
    try:
        import kagglehub
        datasets = [
            "mukulkirti/viral-social-media-trends-engagement-analysis",
            "ruchi798/twitter-sentiments-analysis",
        ]
        for handle in datasets:
            try:
                print(f"[Data] Trying Kaggle: {handle}")
                path = kagglehub.dataset_download(handle)
                csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
                if not csvs:
                    continue
                df = pd.read_csv(csvs[0])
                print(f"[Data] ✅ Downloaded: {csvs[0]}  ({len(df):,} rows)")

                # normalise and save
                df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
                df.to_csv(OUT_FILE, index=False)
                print(f"[Data] Saved to {OUT_FILE}")
                return True
            except Exception as exc:
                print(f"[Data] Skipped {handle}: {exc}")
    except ImportError:
        print("[Data] kagglehub not installed.")
    return False


# ════════════════════════════════════════════════════════════════════════════
# 2.  Synthetic generator (realistic viral dynamics)
# ════════════════════════════════════════════════════════════════════════════

HASHTAGS = [
    # tech
    "chatgpt", "ai", "machinelearning", "python", "openai",
    "gpt4", "llm", "deeplearning", "tech", "coding",
    # entertainment
    "barbie", "oppenheimer", "viral", "trending", "memes",
    "taylor_swift", "beyonce", "netflix", "oscars", "grammys",
    # social / news
    "climatechange", "election2024", "worldcup", "nfl", "nba",
    "crypto", "bitcoin", "ethereum", "stocks", "inflation",
    # lifestyle
    "fitness", "health", "travel", "food", "fashion",
]

PLATFORMS = ["twitter", "instagram", "tiktok", "youtube"]
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 12, 31)


def _weekday_multiplier(date: datetime) -> float:
    """Weekends see ~20% more social activity."""
    return 1.20 if date.weekday() >= 5 else 1.0


def _seasonal_component(day_of_year: int) -> float:
    """Mild annual seasonality — higher Q4 (holiday events)."""
    return 1.0 + 0.25 * np.sin(2 * np.pi * (day_of_year - 90) / 365)


def _generate_viral_spike(
    n_days: int,
    spike_start: int,
    base_level: float,
    peak_multiplier: float = None,
    rise_days: int = None,
    decay_rate: float = None,
) -> np.ndarray:
    """
    Generate a realistic spike: slow build → exponential rise → power-law decay.
    """
    if peak_multiplier is None:
        peak_multiplier = np.random.uniform(8, 60)
    if rise_days is None:
        rise_days = max(2, int(np.random.exponential(4)))
    if decay_rate is None:
        decay_rate = np.random.uniform(0.15, 0.40)

    signal = np.ones(n_days) * base_level
    peak   = base_level * peak_multiplier

    for d in range(n_days):
        days_from_start = d - spike_start
        if days_from_start < 0:
            # pre-spike: slight build-up
            pre_factor = max(0, 1 + 0.1 * days_from_start / max(1, spike_start))
            signal[d]  = base_level * pre_factor
        elif days_from_start <= rise_days:
            # exponential rise
            t         = days_from_start / rise_days
            signal[d] = base_level + (peak - base_level) * (t ** 1.5)
        else:
            # power-law decay
            t         = days_from_start - rise_days
            signal[d] = base_level + (peak - base_level) * np.exp(-decay_rate * t)

    return np.maximum(signal, base_level * 0.3)


def generate_hashtag_series(hashtag: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate a realistic daily mention time-series for one hashtag."""
    n_days     = len(dates)
    base_level = np.random.uniform(200, 2000)     # normal daily mentions

    # decide how many viral events this hashtag has
    n_spikes = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
    spike_starts = sorted(np.random.choice(range(30, n_days - 30),
                                            n_spikes, replace=False))

    # baseline noise (Pareto-distributed to mimic real social media noise)
    pareto_shape = 1.5
    noise = (np.random.pareto(pareto_shape, n_days) + 1) * (base_level * 0.3)

    # init signal
    signal = np.ones(n_days) * base_level + noise

    # layer viral spikes
    for ss in spike_starts:
        spike = _generate_viral_spike(n_days, ss, base_level)
        signal = np.maximum(signal, spike)  # keep higher of baseline or spike

    # add weekday + seasonal components
    multipliers = np.array([
        _weekday_multiplier(dates[i].to_pydatetime()) *
        _seasonal_component(dates[i].dayofyear)
        for i in range(n_days)
    ])
    signal = signal * multipliers

    # final noise
    signal += np.random.normal(0, base_level * 0.05, n_days)
    signal  = np.maximum(signal, 10).astype(int)

    # sentiment: positive during organic growth, mixed during viral,
    # slightly negative during decay
    rm      = pd.Series(signal).rolling(7, min_periods=1).mean().values
    is_high = signal > rm * 2
    sentiment = np.where(
        is_high,
        np.random.uniform(-0.1, 0.4, n_days),     # viral: mixed
        np.random.uniform(0.0,  0.5, n_days),      # baseline: positive
    )

    platform = np.random.choice(PLATFORMS, n_days,
                                 p=[0.35, 0.30, 0.25, 0.10])

    return pd.DataFrame({
        "date":            dates,
        "hashtag":         f"#{hashtag}",
        "mentions":        signal,
        "platform":        platform,
        "sentiment_score": sentiment.round(4),
        "likes":           (signal * np.random.uniform(3, 15, n_days)).astype(int),
        "shares":          (signal * np.random.uniform(0.5, 3, n_days)).astype(int),
        "reach":           (signal * np.random.uniform(10, 50, n_days)).astype(int),
    })


def generate_dataset() -> pd.DataFrame:
    print("[Data] Generating realistic synthetic social media dataset...")
    dates  = pd.date_range(START_DATE, END_DATE, freq="D")
    frames = [generate_hashtag_series(ht, dates) for ht in HASHTAGS]
    df     = pd.concat(frames, ignore_index=True)
    df.sort_values(["hashtag", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"[Data] ✅ Generated {len(df):,} rows  →  {OUT_FILE}")
    print(f"       Hashtags: {df['hashtag'].nunique()}  |  "
          f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("   Social Media Trend Detector -- Data Setup")
    print("=" * 55)

    if not os.path.exists(OUT_FILE):
        if not try_kaggle():
            print("\n[Data] Kaggle unavailable — generating synthetic data.")
            generate_dataset()
    else:
        df = pd.read_csv(OUT_FILE)
        print(f"[Data] ✅ Existing data found: {OUT_FILE}  ({len(df):,} rows)")
        print(f"       Columns: {list(df.columns)}")

    print("\n[Data] Ready. Run  python main.py  to start analysis.\n")
