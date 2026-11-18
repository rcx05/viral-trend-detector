# 🔥 Viral Trend Detector

> **College Project** — Social Media Trend & Viral Spike Detection  
> Built with Python 3.14, Streamlit, Plotly, and 4 real-time APIs

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
copy .env.example .env
# Edit .env with your keys (Google Trends works without any key!)

# 3. Run the dashboard
streamlit run app.py
```

The app opens at **http://localhost:8501** automatically.

---

## Project Structure

```
viral-trend-detector/
├── app.py                      ← Streamlit dashboard (main entry)
├── requirements.txt
├── .env.example                ← API key template
├── src/
│   ├── api/
│   │   ├── google_trends.py    ← pytrends (no key needed)
│   │   ├── reddit_api.py       ← PRAW Reddit client
│   │   ├── news_api.py         ← NewsAPI client
│   │   ├── youtube_api.py      ← YouTube Data API v3
│   │   └── aggregator.py       ← Unified data layer + caching
│   ├── analysis/
│   │   ├── engine.py           ← Core analysis pipeline
│   │   └── changepoint.py      ← CUSUM change-point detection
│   └── utils/
│       └── charts.py           ← Plotly chart factory
└── data/
    └── cache/                  ← 5-minute API response cache
```

---

## Data Sources & API Keys

| Source | Key Required | Free Tier | Where to Get |
|--------|-------------|-----------|--------------|
| **Google Trends** | No | Unlimited | Built-in via pytrends |
| **Reddit** | Yes | Unlimited reads | reddit.com/prefs/apps |
| **NewsAPI** | Yes | 100 req/day | newsapi.org/register |
| **YouTube** | Yes | 10,000 units/day | console.cloud.google.com |

---

## Techniques Used

### 1. Rolling Statistics
- Rolling Mean (configurable window, default 7 days)
- Rolling Standard Deviation for confidence band
- Momentum = % change in rolling mean per day

### 2. Z-Score Anomaly Detection
Points with |Z| > 2.5 sigma are flagged as viral spikes.

### 3. Change-Point Detection (CUSUM Binary Segmentation)
Implemented from scratch without C-extension dependencies.

### 4. Seasonal Decomposition (STL)
Decomposes signal into Trend + Seasonal + Residual.

### 5. Viral Score (Composite, 0-100)
Weighted combination of Momentum + Z-score + Acceleration.

---

## Dashboard Tabs

| Tab | Contents |
|-----|---------|
| Trend Overview | Raw signal + rolling mean + confidence band + change points |
| Viral Score | Gauge (0-100) + viral score timeline |
| Z-Score | Bar chart with threshold bands |
| Decomposition | STL 4-panel: Original / Trend / Seasonal / Residual |
| Multi-Source | Overlay comparison of all data sources |
| News and Posts | Latest news articles + Reddit hot posts |
