"""
Viral Trend Detector — Plotly Visualization Factory
All charts are created here and returned as Plotly figures.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional


# ─── Colour Palette ──────────────────────────────────────────────────────────
PALETTE = {
    "primary":    "#6C63FF",
    "secondary":  "#FF6584",
    "success":    "#43E97B",
    "warning":    "#FA8231",
    "danger":     "#FF4757",
    "dark":       "#1E1E2E",
    "card":       "#2A2A3E",
    "text":       "#CDD6F4",
    "subtext":    "#7F849C",
    "google":     "#EA4335",
    "reddit":     "#FF4500",
    "news":       "#0078FF",
    "youtube":    "#FF0000",
}

LAYOUT_BASE = dict(
    paper_bgcolor = "#1E1E2E",
    plot_bgcolor  = "#1E1E2E",
    font          = dict(family="Inter, sans-serif", color="#CDD6F4"),
    margin        = dict(l=50, r=30, t=50, b=50),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)


# ──────────────────────────── Main Trend Chart ───────────────────────────────

def plot_trend_overview(analysis: Dict[str, Any]) -> go.Figure:
    """
    Main chart: raw value + rolling mean band + anomaly markers + changepoints.
    """
    series    = analysis["series"]
    rolling   = analysis["rolling"]
    anomaly   = analysis["anomaly"]
    changepoints = analysis.get("changepoints", [])
    topic     = analysis["topic"]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x    = rolling.index.tolist() + rolling.index.tolist()[::-1],
        y    = rolling["rolling_upper"].tolist() + rolling["rolling_lower"].tolist()[::-1],
        fill = "toself",
        fillcolor = "rgba(108,99,255,0.12)",
        line = dict(width=0),
        name = "±2σ Band",
        showlegend = True,
    ))

    # Raw values
    fig.add_trace(go.Scatter(
        x    = series.index,
        y    = series.values,
        mode = "lines",
        line = dict(color=PALETTE["primary"], width=2),
        name = "Interest Score",
    ))

    # Rolling mean
    fig.add_trace(go.Scatter(
        x    = rolling.index,
        y    = rolling["rolling_mean"],
        mode = "lines",
        line = dict(color=PALETTE["success"], width=2, dash="dot"),
        name = "Rolling Mean",
    ))

    # Anomaly spikes
    spikes = anomaly[anomaly["is_spike"]]
    if not spikes.empty:
        fig.add_trace(go.Scatter(
            x       = spikes.index,
            y       = spikes["value"],
            mode    = "markers",
            marker  = dict(color=PALETTE["danger"], size=10, symbol="star",
                           line=dict(color="white", width=1)),
            name    = "🔥 Viral Spike",
        ))

    # Change-points — add as shapes + annotations (more compatible)
    for cp in changepoints:
        fig.add_shape(
            type       = "line",
            x0         = cp, x1 = cp,
            y0         = 0, y1  = 1,
            yref       = "paper",
            line       = dict(color=PALETTE["warning"], width=2, dash="dash"),
        )
        fig.add_annotation(
            x          = cp,
            y          = 1,
            yref       = "paper",
            text       = "⚑ Change",
            showarrow  = False,
            font       = dict(color=PALETTE["warning"], size=10),
            bgcolor    = "rgba(0,0,0,0.5)",
            borderpad  = 2,
        )

    fig.update_layout(
        title = dict(text=f"📈 Trend Analysis: <b>{topic}</b>", font=dict(size=18)),
        xaxis = dict(title="Date", gridcolor="#313244", showgrid=True),
        yaxis = dict(title="Interest Score (0–100)", gridcolor="#313244", showgrid=True),
        hovermode = "x unified",
        **LAYOUT_BASE,
    )
    return fig


# ──────────────────────────── Z-Score Chart ──────────────────────────────────

def plot_zscore(analysis: Dict[str, Any]) -> go.Figure:
    """Z-score timeline with threshold bands."""
    anomaly = analysis["anomaly"]
    topic   = analysis["topic"]
    thresh  = 2.5

    fig = go.Figure()

    # Threshold bands (using shapes + annotations for compatibility)
    for y, label, color in [
        (thresh,  f"+{thresh}s (Viral)",   PALETTE["danger"]),
        (-thresh, f"-{thresh}s (Drop)",    PALETTE["warning"]),
    ]:
        fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                      y0=y, y1=y, line=dict(color=color, dash="dash", width=1.5))
        fig.add_annotation(x=1, xref="paper", y=y, text=label, showarrow=False,
                           xanchor="left", font=dict(color=color, size=10))

    # Zero line
    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                  y0=0, y1=0, line=dict(color=PALETTE["subtext"], width=1))

    # Z-score bars
    colors = [
        PALETTE["danger"] if z > thresh else
        PALETTE["warning"] if z < -thresh else
        PALETTE["primary"]
        for z in anomaly["zscore"]
    ]
    fig.add_trace(go.Bar(
        x           = anomaly.index,
        y           = anomaly["zscore"],
        marker_color = colors,
        name         = "Z-Score",
        hovertemplate = "<b>%{x}</b><br>Z-Score: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title  = dict(text=f"📊 Z-Score Anomaly Detection: <b>{topic}</b>", font=dict(size=16)),
        xaxis  = dict(title="Date", gridcolor="#313244"),
        yaxis  = dict(title="Z-Score (σ)", gridcolor="#313244"),
        **LAYOUT_BASE,
    )
    return fig


# ──────────────────────────── Viral Score Gauge ──────────────────────────────

def plot_viral_gauge(viral_score: float, topic: str) -> go.Figure:
    """Gauge chart showing current viral score."""
    color = (
        PALETTE["danger"]   if viral_score >= 75 else
        PALETTE["warning"]  if viral_score >= 50 else
        PALETTE["primary"]  if viral_score >= 25 else
        PALETTE["subtext"]
    )

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = viral_score,
        title = {"text": f"🔥 Viral Score<br><sub>{topic}</sub>", "font": {"size": 16, "color": "#CDD6F4"}},
        delta = {"reference": 50, "increasing": {"color": PALETTE["danger"]},
                 "decreasing": {"color": PALETTE["success"]}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#CDD6F4"},
            "bar":   {"color": color},
            "bgcolor": "#2A2A3E",
            "steps": [
                {"range": [0,  25], "color": "#313244"},
                {"range": [25, 50], "color": "#3B3B52"},
                {"range": [50, 75], "color": "#45455E"},
                {"range": [75, 100], "color": "#52526B"},
            ],
            "threshold": {
                "line": {"color": PALETTE["danger"], "width": 3},
                "thickness": 0.75,
                "value": 75,
            },
        },
        number = {"suffix": "/100", "font": {"size": 28, "color": color}},
    ))
    fig.update_layout(paper_bgcolor="#1E1E2E", font_color="#CDD6F4",
                      margin=dict(l=20, r=20, t=60, b=20))
    return fig


# ──────────────────────────── Seasonal Decomposition ────────────────────────

def plot_decomposition(analysis: Dict[str, Any]) -> go.Figure:
    """4-panel STL decomposition chart."""
    series = analysis["series"]
    decomp = analysis["decomp"]
    topic  = analysis["topic"]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=["Original Signal", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.07,
    )

    for i, (name, data, color) in enumerate([
        ("Original",  series,              PALETTE["primary"]),
        ("Trend",     decomp["trend"],     PALETTE["success"]),
        ("Seasonal",  decomp["seasonal"],  PALETTE["warning"]),
        ("Residual",  decomp["residual"],  PALETTE["secondary"]),
    ], start=1):
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values,
            mode="lines", line=dict(color=color, width=1.5),
            name=name, showlegend=(i == 1),
        ), row=i, col=1)

    fig.update_layout(
        title = dict(text=f"🌊 Seasonal Decomposition: <b>{topic}</b>", font=dict(size=16)),
        height = 600,
        **LAYOUT_BASE,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#313244", row=i, col=1)
        fig.update_yaxes(gridcolor="#313244", row=i, col=1)
    return fig


# ──────────────────────────── Viral Score Timeline ───────────────────────────

def plot_viral_score_timeline(analysis: Dict[str, Any]) -> go.Figure:
    """Coloured area chart of viral score over time."""
    vs    = analysis["viral_score"]
    topic = analysis["topic"]

    colors = [
        PALETTE["danger"]  if v >= 75 else
        PALETTE["warning"] if v >= 50 else
        PALETTE["primary"] if v >= 25 else
        PALETTE["subtext"]
        for v in vs.values
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = vs.index,
        y    = vs.values,
        mode = "lines",
        fill = "tozeroy",
        line = dict(color=PALETTE["primary"], width=2),
        fillcolor = "rgba(108,99,255,0.25)",
        name = "Viral Score",
        hovertemplate = "<b>%{x}</b><br>Viral Score: %{y:.1f}<extra></extra>",
    ))
    # Threshold lines (using shapes for compatibility)
    for y, label, color in [
        (75, "Viral (75)",   PALETTE["danger"]),
        (50, "Rising (50)",  PALETTE["warning"]),
        (25, "Active (25)",  PALETTE["subtext"]),
    ]:
        fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                      y0=y, y1=y, line=dict(color=color, dash="dash", width=1.5))
        fig.add_annotation(x=1.01, xref="paper", y=y, text=label, showarrow=False,
                           xanchor="left", font=dict(color=color, size=10))

    fig.update_layout(
        title = dict(text=f"💯 Viral Score Timeline: <b>{topic}</b>", font=dict(size=16)),
        xaxis = dict(title="Date", gridcolor="#313244"),
        yaxis = dict(title="Viral Score (0–100)", gridcolor="#313244", range=[0, 105]),
        **LAYOUT_BASE,
    )
    return fig


# ──────────────────────────── Multi-Source Comparison ───────────────────────

def plot_multi_source(sources_data: Dict[str, pd.Series], topic: str) -> go.Figure:
    """Compare interest trends across multiple data sources."""
    source_colors = {
        "google_trends": PALETTE["google"],
        "reddit":        PALETTE["reddit"],
        "news":          PALETTE["news"],
        "youtube":       PALETTE["youtube"],
        "combined":      PALETTE["primary"],
    }
    source_labels = {
        "google_trends": "🔍 Google Trends",
        "reddit":        "🤖 Reddit",
        "news":          "📰 NewsAPI",
        "youtube":       "▶️ YouTube",
        "combined":      "⚡ Combined",
    }

    fig = go.Figure()
    for src, s in sources_data.items():
        if s is None or s.empty:
            continue
        fig.add_trace(go.Scatter(
            x    = s.index,
            y    = s.values,
            mode = "lines",
            line = dict(color=source_colors.get(src, PALETTE["primary"]), width=2),
            name = source_labels.get(src, src),
        ))

    fig.update_layout(
        title     = dict(text=f"🔗 Multi-Source Comparison: <b>{topic}</b>", font=dict(size=16)),
        xaxis     = dict(title="Date", gridcolor="#313244"),
        yaxis     = dict(title="Normalized Interest (0–100)", gridcolor="#313244"),
        hovermode = "x unified",
        **LAYOUT_BASE,
    )
    return fig


# ──────────────────────────── Momentum Bar Chart ─────────────────────────────

def plot_momentum(analysis: Dict[str, Any]) -> go.Figure:
    """Daily momentum (% change in rolling mean)."""
    rolling = analysis["rolling"]
    topic   = analysis["topic"]

    mom = rolling["momentum"].fillna(0)
    colors = [PALETTE["success"] if v >= 0 else PALETTE["danger"] for v in mom]

    fig = go.Figure(go.Bar(
        x            = rolling.index,
        y            = mom,
        marker_color = colors,
        name         = "Momentum %",
        hovertemplate = "<b>%{x}</b><br>Momentum: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=PALETTE["subtext"])

    fig.update_layout(
        title = dict(text=f"⚡ Momentum (Rolling Mean % Change): <b>{topic}</b>", font=dict(size=16)),
        xaxis = dict(title="Date", gridcolor="#313244"),
        yaxis = dict(title="% Change", gridcolor="#313244"),
        **LAYOUT_BASE,
    )
    return fig
