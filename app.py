"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          VIRAL TREND DETECTOR — Streamlit Dashboard                         ║
║          College Project: Social Media Trend & Viral Spike Detection        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run with:  streamlit run app.py
"""

import sys
import os
import time
import warnings
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.api.aggregator  import aggregate_sources, get_trending_topics
from src.analysis.engine import full_analysis
from src.utils.charts    import (
    plot_trend_overview,
    plot_zscore,
    plot_viral_gauge,
    plot_decomposition,
    plot_viral_score_timeline,
    plot_multi_source,
    plot_momentum,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "🔥 Viral Trend Detector",
    page_icon  = "🔥",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #1E1E2E; color: #CDD6F4; }
    .block-container { padding-top: 1rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #2A2A3E;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #7F849C !important; font-size: 13px; }
    [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700; }

    /* Alert box */
    .viral-alert {
        background: linear-gradient(135deg, #FF4757, #FF6B81);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        font-size: 18px;
        font-weight: 700;
        text-align: center;
        margin: 8px 0;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { opacity: 1; }
        50%      { opacity: 0.8; }
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
    }
    .status-viral   { background:#FF4757; color:white; }
    .status-rising  { background:#FA8231; color:white; }
    .status-active  { background:#6C63FF; color:white; }
    .status-quiet   { background:#45475A; color:#CDD6F4; }

    /* Section header */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #CDD6F4;
        border-left: 4px solid #6C63FF;
        padding-left: 10px;
        margin: 20px 0 10px 0;
    }

    /* News card */
    .news-card {
        background: #2A2A3E;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .news-card a { color: #89B4FA; text-decoration: none; }

    /* Source badge */
    .src-google  { color: #EA4335; font-weight:700; }
    .src-reddit  { color: #FF4500; font-weight:700; }
    .src-news    { color: #0078FF; font-weight:700; }
    .src-youtube { color: #FF0000; font-weight:700; }

    /* Hide Streamlit branding */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔥 Viral Trend Detector")
    st.markdown("*Real-time social media spike detection*")
    st.divider()

    # ── API Status ────────────────────────────────────────────────────────
    st.markdown("### 📡 Data Sources")

    has_reddit_key  = bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_ID") != "your_reddit_client_id")
    has_news_key    = bool(os.getenv("NEWS_API_KEY") and os.getenv("NEWS_API_KEY") != "your_newsapi_key")
    has_youtube_key = bool(os.getenv("YOUTUBE_API_KEY") and os.getenv("YOUTUBE_API_KEY") != "your_youtube_api_key")

    apis = {
        "🔍 Google Trends": ("pytrends",   True),
        "🤖 Reddit":        ("API" if has_reddit_key else "Public JSON", True),
        "📰 News":          ("API" if has_news_key else "Google News RSS", True),
        "▶️ YouTube":       ("API" if has_youtube_key else "Invidious", True),
    }
    for name, (mode, active) in apis.items():
        st.markdown(f"🟢 **{name}** `{mode}`")

    st.caption("✅ All sources work without API keys!")

    st.divider()

    # ── Search Input ───────────────────────────────────────────────────────
    st.markdown("### 🔎 Search Topic")
    keyword = st.text_input(
        "Keyword or #hashtag",
        value = "#AI",
        placeholder = "e.g. #WorldCup, Bitcoin, Taylor Swift",
        help = "Enter any keyword or hashtag to analyse.",
    )

    # ── Source Selection ───────────────────────────────────────────────────
    st.markdown("### ☑️ Sources to Use")
    use_google  = st.checkbox("🔍 Google Trends",  value=True,  disabled=True)
    use_reddit  = st.checkbox("🤖 Reddit",         value=True)
    use_news    = st.checkbox("📰 News",            value=True)
    use_youtube = st.checkbox("▶️ YouTube",         value=True)

    sources = ["google_trends"]
    if use_reddit:  sources.append("reddit")
    if use_news:    sources.append("news")
    if use_youtube: sources.append("youtube")

    # ── Analysis Settings ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Analysis Settings")
    timeframe_map = {
        "Last 7 days":   "now 7-d",
        "Last 1 month":  "today 1-m",
        "Last 3 months": "today 3-m",
        "Last 12 months":"today 12-m",
    }
    timeframe_label = st.selectbox("Time Range", list(timeframe_map.keys()), index=2)
    timeframe = timeframe_map[timeframe_label]

    rolling_window = st.slider("Rolling Window (days)", 3, 30, 7)
    zscore_thresh  = st.slider("Z-Score Threshold (σ)", 1.5, 4.0, 2.5, 0.1)
    cp_penalty     = st.slider("Change-Point Sensitivity", 1.0, 5.0, 3.0, 0.5,
                                help="Lower = more change points detected")

    st.divider()
    analyse_btn = st.button("🚀 Analyse Trends", type="primary", use_container_width=True)

    st.markdown("### 🌍 Today's Trending")
    show_trending = st.button("🔥 Load Trending Topics", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🔥 Viral Trend & Spike Detector")
st.markdown(
    "Detect viral trends using **Google Trends · Reddit · NewsAPI · YouTube** "
    "— with rolling averages, Z-score, change-point & seasonal decomposition."
)
st.divider()

# ── Trending Topics Panel ──────────────────────────────────────────────────
if show_trending:
    with st.spinner("🌍 Loading today's trending searches..."):
        trending = get_trending_topics()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🇺🇸 Trending (Worldwide)")
        for i, t in enumerate(trending.get("worldwide", [])[:10], 1):
            st.markdown(f"`{i}.` {t}")
    with col2:
        st.markdown("### 🇮🇳 Trending (India)")
        for i, t in enumerate(trending.get("india", [])[:10], 1):
            st.markdown(f"`{i}.` {t}")
    st.divider()


# ── Analysis ──────────────────────────────────────────────────────────────
if analyse_btn or "analysis_result" in st.session_state:

    if analyse_btn:
        with st.spinner(f"🔄 Fetching data for **{keyword}** from {len(sources)} source(s)…"):
            agg = aggregate_sources(keyword, sources=sources, timeframe=timeframe)
            st.session_state["agg"]     = agg
            st.session_state["keyword"] = keyword

        series = agg.get("combined")

        # Fallback to Google Trends only
        if series is None and "google_trends" in agg.get("sources", {}):
            series = agg["sources"]["google_trends"]

        if series is None or series.empty:
            st.error("❌ No data returned. Check your keyword or API configuration.")
            st.stop()

        with st.spinner("🧠 Running analysis engine…"):
            result = full_analysis(
                series,
                topic          = keyword,
                window         = rolling_window,
                zscore_thresh  = zscore_thresh,
                cp_penalty     = cp_penalty,
            )
            st.session_state["analysis_result"] = result

    agg    = st.session_state.get("agg", {})
    result = st.session_state["analysis_result"]
    kw     = st.session_state.get("keyword", keyword)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    # ── Viral Alert Banner ─────────────────────────────────────────────────
    if result["is_viral"]:
        st.markdown(
            f'<div class="viral-alert">🚨 VIRAL TREND DETECTED: <b>{kw}</b> — '
            f'Z-Score: {result["latest_zscore"]:.1f}σ | Viral Score: {result["latest_viral_score"]:.0f}/100</div>',
            unsafe_allow_html=True,
        )

    # ── KPI Cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("🎯 Status",         result["status"])
    with c2: st.metric("💯 Viral Score",     f"{result['latest_viral_score']:.0f}/100")
    with c3: st.metric("📊 Z-Score",         f"{result['latest_zscore']:.2f}σ")
    with c4: st.metric("🏔️ Peak Value",      f"{result['peak_value']:.0f}")
    with c5: st.metric("🔄 Change Points",   len(result["changepoints"]))

    st.divider()

    # ── Active Sources ─────────────────────────────────────────────────────
    avail = agg.get("available", {})
    src_icons = {
        "google_trends": ("🔍 Google Trends", "#EA4335"),
        "reddit":        ("🤖 Reddit",        "#FF4500"),
        "news":          ("📰 NewsAPI",        "#0078FF"),
        "youtube":       ("▶️ YouTube",        "#FF0000"),
    }
    active_srcs = [src_icons[k] for k, v in avail.items() if v and k in src_icons]
    if active_srcs:
        badge_html = " &nbsp;".join(
            f'<span style="background:{c};color:white;padding:4px 10px;border-radius:12px;font-size:13px;font-weight:700">{n}</span>'
            for n, c in active_srcs
        )
        st.markdown(f"**Active Data Sources:** {badge_html}", unsafe_allow_html=True)

    st.markdown("")

    # ── Tab Layout ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Trend Overview",
        "🔥 Viral Score",
        "📊 Z-Score",
        "🌊 Decomposition",
        "🔗 Multi-Source",
        "📰 News & Posts",
    ])

    with tab1:
        st.markdown('<div class="section-title">Trend Overview with Change Points</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_trend_overview(result), use_container_width=True)
        st.plotly_chart(plot_momentum(result), use_container_width=True)

        # Spike summary
        if result["spike_dates"]:
            st.markdown("#### 🚨 Detected Spikes")
            spike_df = pd.DataFrame({
                "Date":        [str(d.date()) for d in result["spike_dates"]],
                "Value":       [f"{result['series'][d]:.1f}" for d in result["spike_dates"]],
                "Z-Score":     [f"{result['anomaly']['zscore'][d]:.2f}σ" for d in result["spike_dates"]],
            })
            st.dataframe(spike_df, use_container_width=True, hide_index=True)

    with tab2:
        col_gauge, col_timeline = st.columns([1, 2])
        with col_gauge:
            st.markdown('<div class="section-title">Current Viral Score</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_viral_gauge(result["latest_viral_score"], kw),
                use_container_width=True,
            )
            # Score breakdown
            st.markdown("#### 📖 Score Interpretation")
            for score, label, desc in [
                (75, "🔥 VIRAL",   "Explosive growth. Trending everywhere."),
                (50, "📈 RISING",  "Above average. Clear upward momentum."),
                (25, "📊 ACTIVE",  "Normal activity. Worth monitoring."),
                (0,  "😴 QUIET",   "Below baseline. No significant activity."),
            ]:
                current = result["latest_viral_score"]
                is_curr = (current >= score) and (current < score + 25 or score == 75)
                prefix  = "→ **" if is_curr else "   "
                suffix  = "** ← *Current*" if is_curr else ""
                st.markdown(f"{prefix}{score}+ = {label}: {desc}{suffix}")

        with col_timeline:
            st.markdown('<div class="section-title">Viral Score Timeline</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(plot_viral_score_timeline(result), use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title">Z-Score Anomaly Detection</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"Bars beyond **±{zscore_thresh}σ** are flagged as anomalies. "
            f"Red bars = viral spikes. Orange = drops."
        )
        st.plotly_chart(plot_zscore(result), use_container_width=True)

        # Z-score stats table
        z = result["anomaly"]["zscore"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Max Z-Score",   f"{z.max():.2f}σ")
        col_b.metric("Min Z-Score",   f"{z.min():.2f}σ")
        col_c.metric("# Anomalies",   str(result["anomaly"]["is_anomaly"].sum()))

    with tab4:
        st.markdown('<div class="section-title">Seasonal Decomposition (STL)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Breaks the signal into **Trend** (long-term direction), "
            "**Seasonal** (repeating pattern), and **Residual** (noise/spikes)."
        )
        st.plotly_chart(plot_decomposition(result), use_container_width=True)

        # Change-point list
        if result["changepoints"]:
            st.markdown("#### 🔀 Structural Change Points Detected")
            for i, cp in enumerate(result["changepoints"], 1):
                val_at_cp = result["series"].get(cp, "N/A")
                val_str   = f"{val_at_cp:.1f}" if isinstance(val_at_cp, float) else str(val_at_cp)
                st.markdown(f"**{i}.** `{cp.strftime('%Y-%m-%d')}` — Value at change: **{val_str}**")
        else:
            st.info("No structural change points detected with current sensitivity.")

    with tab5:
        src_dict = {k: v for k, v in agg.get("sources", {}).items()}
        combined = agg.get("combined")
        if combined is not None:
            src_dict["combined"] = combined

        if len(src_dict) > 1:
            st.markdown('<div class="section-title">Multi-Source Interest Comparison</div>',
                        unsafe_allow_html=True)
            st.markdown("All sources are normalized to 0–100 for fair comparison.")
            st.plotly_chart(plot_multi_source(src_dict, kw), use_container_width=True)
        else:
            st.info(
                "Only one data source is active. Configure additional API keys "
                "in `.env` to see multi-source comparison."
            )

        # Related queries from Google Trends
        rq = agg.get("supplemental", {}).get("related_queries", {})
        if rq.get("rising") is not None and not rq["rising"].empty:
            st.markdown("#### 📈 Rising Related Queries (Google)")
            st.dataframe(rq["rising"].head(10), use_container_width=True, hide_index=True)
        if rq.get("top") is not None and not rq["top"].empty:
            st.markdown("#### 🔝 Top Related Queries (Google)")
            st.dataframe(rq["top"].head(10), use_container_width=True, hide_index=True)

    with tab6:
        # News articles
        top_news = agg.get("supplemental", {}).get("top_news", [])
        if top_news:
            st.markdown('<div class="section-title">📰 Latest News Articles</div>',
                        unsafe_allow_html=True)
            for article in top_news:
                st.markdown(f"""
<div class="news-card">
<b><a href="{article['url']}" target="_blank">{article['title']}</a></b><br>
<small><span class="src-news">📰 {article['source']}</span> · {article['published_at'][:10]}</small><br>
<small>{article.get('description', '')[:200]}…</small>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No news articles. Add a NewsAPI key to `.env` to see news.")

        # Reddit hot posts
        reddit_hot = agg.get("supplemental", {}).get("reddit_hot", [])
        if reddit_hot:
            st.markdown('<div class="section-title">🤖 Hot on Reddit Right Now</div>',
                        unsafe_allow_html=True)
            for post in reddit_hot[:8]:
                st.markdown(f"""
<div class="news-card">
<b><a href="{post['url']}" target="_blank">{post['title']}</a></b><br>
<small><span class="src-reddit">r/{post['subreddit']}</span> · 
⬆ {post['score']:,} · 💬 {post['comments']:,}</small>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No Reddit data. Add Reddit API credentials to `.env`.")

else:
    # ── Welcome Screen ─────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center; padding: 60px 0;">
<div style="font-size:80px;">🔥</div>
<h2 style="color:#CDD6F4">Enter a keyword or hashtag and click <b>Analyse Trends</b></h2>
<p style="color:#7F849C; font-size:16px;">
  Powered by <b>Google Trends · Reddit · NewsAPI · YouTube</b><br>
  Using <b>Rolling Averages · Z-Score · Change-Point Detection · STL Decomposition</b>
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Example Topics to Try")
    example_cols = st.columns(4)
    examples = [
        ("🤖 #AI", "#AI"),
        ("₿ Bitcoin",  "Bitcoin"),
        ("🏏 #IPL2024","#IPL2024"),
        ("🎬 Squid Game","Squid Game"),
        ("🌍 Climate Change","Climate Change"),
        ("🎵 Taylor Swift","Taylor Swift"),
        ("🚀 SpaceX","SpaceX"),
        ("🎮 GTA6","GTA6"),
    ]
    for i, (label, kw_ex) in enumerate(examples):
        with example_cols[i % 4]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state["auto_keyword"] = kw_ex
                st.rerun()

    # ── How It Works ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧠 How It Works")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown("""
**📡 Data Collection**
- Google Trends (free)
- Reddit (PRAW API)
- NewsAPI articles
- YouTube video counts
        """)
    with h2:
        st.markdown("""
**📊 Rolling Statistics**
- 7-day rolling mean
- Rolling ±2σ band
- Momentum (% change)
- Acceleration
        """)
    with h3:
        st.markdown("""
**🚨 Anomaly Detection**
- Z-score analysis
- Threshold: ±2.5σ
- Spike classification
- Drop detection
        """)
    with h4:
        st.markdown("""
**🌊 Advanced Techniques**
- STL decomposition
- Trend / Seasonal / Residual
- Change-point detection
- Viral score (0–100)
        """)

# auto-fill keyword from quick buttons
if "auto_keyword" in st.session_state:
    # Cannot directly set text_input value after render; show a hint
    kw_hint = st.session_state.pop("auto_keyword")
    st.sidebar.info(f"Type **{kw_hint}** in the search box and click Analyse!")
