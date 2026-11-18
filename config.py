"""
config.py - Global Configuration for Viral Trend Detector
"""

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR  = "data"
PLOTS_DIR = "plots"
CACHE_DIR = "cache"

# ── Kaggle dataset (pulled via kagglehub) ───────────────────────────────────
KAGGLE_DATASET_HANDLE = "shivamb/netflix-shows"   # fallback – overridden below
TRENDING_DATASET      = "ruchi798/twitter-sentiments-analysis"
VIRAL_DATASET         = "mukulkirti/viral-social-media-trends-engagement-analysis"

# ── Analysis parameters ──────────────────────────────────────────────────────
Z_SCORE_THRESHOLD   = 2.5     # Flag if Z-score > this
ROLLING_WINDOW      = 7       # 7-day rolling mean/std
SPIKE_MULTIPLIER    = 3.0     # Mention count must be > 3× rolling mean
MIN_VIRAL_SCORE     = 60      # 0-100 scale: treat as viral if ≥ 60
EARLY_DETECT_WINDOW = 3       # days to look ahead for early warning

# ── Change-point detection ───────────────────────────────────────────────────
CPD_MODEL  = "rbf"            # ruptures model: rbf | l2 | l1
CPD_PENALTY = 3               # Pelt penalty value

# ── Sentiment mapping ────────────────────────────────────────────────────────
SENTIMENT_LABELS = {
    "positive": (0.05, 1.0),
    "neutral":  (-0.05, 0.05),
    "negative": (-1.0, -0.05),
}

# ── Dashboard ────────────────────────────────────────────────────────────────
DASH_HOST = "127.0.0.1"
DASH_PORT = 8050
DASH_DEBUG = False
