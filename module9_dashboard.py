"""
module9_dashboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 9 – Interactive Dashboard (Plotly Dash)
  • Real-time-style dark-themed UI
  • Sidebar: hashtag selector + analysis mode
  • Tab 1 – Trend Overview (rolling mean / STL)
  • Tab 2 – Anomaly Detection (z-score + change-points)
  • Tab 3 – Viral Score Gauge
  • Tab 5 – Early Warning System
  • Tab 6 – Hashtag Rankings Table
  • KPI cards: peak mentions / viral score / EWS / spike count
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Dash ─────────────────────────────────────────────────────────────────────
try:
    import dash
    from dash import dcc, html, Input, Output, State, dash_table
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from config import DASH_HOST, DASH_PORT, DASH_DEBUG

# ── project modules ──────────────────────────────────────────────────────────
from module1_data_loader    import SocialMediaLoader
from module2_trend_analysis import TrendAnalyzer
from module3_anomaly_detection import AnomalyDetector
from module4_hashtag_trends import HashtagTrendDetector
from module5_viral_score    import ViralScorer
from module7_event_detection import EventDetector
from module8_early_detection import EarlyViralDetector


# ════════════════════════════════════════════════════════════════════════════
# Dark-theme helpers
# ════════════════════════════════════════════════════════════════════════════

DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
ACCENT    = "#58a6ff"
RED       = "#f85149"
GREEN     = "#3fb950"
YELLOW    = "#e3b341"

PLOT_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1),
    xaxis=dict(gridcolor=BORDER, showline=True, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, showline=True, linecolor=BORDER),
)


def _layout(**kwargs):
    d = PLOT_LAYOUT.copy()
    d.update(kwargs)
    return d


# ════════════════════════════════════════════════════════════════════════════
# Data cache (loaded once at startup)
# ════════════════════════════════════════════════════════════════════════════

class DataCache:
    def __init__(self):
        self.loader   = SocialMediaLoader()
        self.daily_df = self.loader.load()
        self.hashtags = self.loader.top_hashtags(30)
        self.ranked   = HashtagTrendDetector(top_n=30).rank_hashtags(self.daily_df)
        print(f"[Dashboard] Data cached — {len(self.hashtags)} hashtags ready.")


# ════════════════════════════════════════════════════════════════════════════
# Figure builders
# ════════════════════════════════════════════════════════════════════════════

def fig_trend(series: pd.Series, hashtag: str) -> go.Figure:
    df = TrendAnalyzer().analyze(series, hashtag)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Daily Mentions + Rolling Mean", "Growth Rate (%)"],
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=df.index, y=df["mentions"],
                             name="Mentions", line=dict(color=ACCENT, width=1.5),
                             opacity=0.8, fill="tozeroy",
                             fillcolor="rgba(88,166,255,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rm"],
                             name="Rolling Mean", line=dict(color=RED, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(df["rm"] + df["rstd"]) + list((df["rm"] - df["rstd"])[::-1]),
        fill="toself", fillcolor="rgba(248,81,73,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="±1 STD", showlegend=True), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["growth_rate"],
                         name="Growth Rate",
                         marker_color=np.where(df["growth_rate"] >= 0, GREEN, RED)),
                  row=2, col=1)

    fig.update_layout(title=f"#{hashtag} — Trend Analysis", **_layout())
    return fig


def fig_anomaly(series: pd.Series, hashtag: str) -> go.Figure:
    df = AnomalyDetector().detect(series, hashtag)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Mentions + Anomalies", "Z-Score"],
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=df.index, y=df["mentions"],
                             name="Mentions", line=dict(color=ACCENT, width=1.5),
                             opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rolling_mean"],
                             name="Rolling Mean", line=dict(color=YELLOW, width=2)),
                  row=1, col=1)

    anom = df[df["is_anomaly"]]
    fig.add_trace(go.Scatter(x=anom.index, y=anom["mentions"],
                             mode="markers", name="🚨 Anomaly",
                             marker=dict(color=RED, size=9, symbol="triangle-up",
                                         line=dict(color="white", width=1))),
                  row=1, col=1)
    # change-points
    for cp in df.index[df["is_changepoint"]]:
        fig.add_vline(x=cp, line_color="#bc8cff", line_dash="dash",
                      line_width=1.5, opacity=0.7, row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["z_score"],
                             name="Z-Score", line=dict(color="#79c0ff", width=1.5)),
                  row=2, col=1)
    fig.add_hline(y=2.5,  line_color=RED, line_dash="dash", row=2, col=1)
    fig.add_hline(y=-2.5, line_color=RED, line_dash="dash", row=2, col=1)
    fig.add_hline(y=0,    line_color="white", line_width=0.7, row=2, col=1)

    fig.update_layout(title=f"#{hashtag} — Anomaly Detection", **_layout())
    return fig


def fig_viral_score(series: pd.Series, hashtag: str) -> go.Figure:
    scorer = ViralScorer()
    df     = scorer.score(series, hashtag)
    summ   = scorer.summary(df)

    tier_color_map = {
        "🔥 MEGA VIRAL": "#ff4444",
        "🔴 HOT":        "#ff8800",
        "🟡 WARM":       "#ffcc00",
        "🔵 RISING":     "#00aaff",
        "⚪ COLD":        "#888888",
    }
    bar_colors = [tier_color_map.get(t, "#888888") for t in df["viral_tier"]]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Daily Mentions (colour = tier)",
                                        "Viral Score (0-100)"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Bar(x=df.index, y=df["mentions"],
                         name="Mentions", marker_color=bar_colors,
                         opacity=0.85), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["viral_score"],
                             name="Viral Score", fill="tozeroy",
                             line=dict(color="#f78166", width=2),
                             fillcolor="rgba(247,129,102,0.2)"), row=2, col=1)
    for thr, lbl, clr in scorer.TIERS[:-1]:
        fig.add_hline(y=thr, line_color=clr, line_dash="dot",
                      annotation_text=lbl, annotation_font_color=clr,
                      row=2, col=1)

    fig.update_yaxes(range=[0, 105], row=2, col=1)
    fig.update_layout(
        title=f"#{hashtag} — Viral Score | Peak: {summ['peak_viral_score']} on {summ['peak_viral_date']}",
        **_layout()
    )
    return fig


def fig_ews(series: pd.Series, hashtag: str) -> go.Figure:
    df = EarlyViralDetector().compute_ews(series, hashtag)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Daily Mentions + Warnings",
                                        "Early Warning Score (EWS)",
                                        "Momentum Divergence Index (MDI)"],
                        vertical_spacing=0.07)

    fig.add_trace(go.Scatter(x=df.index, y=df["mentions"],
                             name="Mentions", line=dict(color=ACCENT, width=1.5)),
                  row=1, col=1)
    warn_mask = df["ews"] > 50
    fig.add_trace(go.Scatter(x=df.index[warn_mask], y=df["mentions"][warn_mask],
                             mode="markers", name="⚠️ Early Warning",
                             marker=dict(color=YELLOW, size=10, symbol="triangle-down",
                                         line=dict(color="white", width=1))),
                  row=1, col=1)

    ews_colors = ["#ff0000" if v >= 75 else "#ff6600" if v >= 50
                  else "#ffcc00" if v >= 25 else "#3fb950"
                  for v in df["ews"]]
    fig.add_trace(go.Bar(x=df.index, y=df["ews"],
                         name="EWS", marker_color=ews_colors, opacity=0.9),
                  row=2, col=1)
    for thr, lbl, clr in [(75, "Pre-Viral", "#ff0000"),
                           (50, "Warning",   "#ff6600"),
                           (25, "Watch",     "#ffcc00")]:
        fig.add_hline(y=thr, line_color=clr, line_dash="dash",
                      annotation_text=lbl, annotation_font_color=clr,
                      row=2, col=1)

    mdi_pos = df["mdi"].clip(lower=0)
    mdi_neg = df["mdi"].clip(upper=0)
    fig.add_trace(go.Bar(x=df.index, y=mdi_pos, name="MDI+",
                         marker_color=GREEN, opacity=0.8), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=mdi_neg, name="MDI-",
                         marker_color=RED,   opacity=0.8), row=3, col=1)
    fig.add_hline(y=0, line_color="white", line_width=0.7, row=3, col=1)

    fig.update_layout(title=f"#{hashtag} — Early Viral Detection", **_layout(),
                      barmode="overlay")
    return fig


def fig_rankings(ranked_df: pd.DataFrame) -> go.Figure:
    top = ranked_df.head(20).copy().reset_index()
    top["hashtag"] = "#" + top["hashtag"]

    fig = px.bar(
        top, x="viral_composite", y="hashtag", orientation="h",
        color="viral_composite",
        color_continuous_scale="YlOrRd",
        labels={"viral_composite": "Viral Score", "hashtag": "Hashtag"},
        title="Top 20 Hashtags by Viral Score",
        text=top["viral_composite"].round(1),
    )
    fig.update_traces(textposition="outside", textfont=dict(color=TEXT))
    fig.update_layout(**_layout(), yaxis=dict(autorange="reversed",
                                              gridcolor=BORDER),
                      coloraxis_showscale=False)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Dashboard layout builder
# ════════════════════════════════════════════════════════════════════════════

def build_app(cache: DataCache) -> "dash.Dash":
    if not DASH_AVAILABLE:
        raise ImportError("Install dash + dash-bootstrap-components: "
                          "pip install dash dash-bootstrap-components")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="🔥 Viral Trend Detector",
    )

    hashtag_options = [{"label": f"#{h}", "value": h}
                       for h in cache.hashtags]
    default_tag = cache.hashtags[0] if cache.hashtags else "ai"

    # ── KPI card helper ──────────────────────────────────────────────────────
    def kpi_card(title: str, value: str, color: str = ACCENT) -> dbc.Card:
        return dbc.Card([
            dbc.CardBody([
                html.P(title, className="text-muted mb-1",
                       style={"fontSize": "0.8rem"}),
                html.H4(value, style={"color": color, "fontWeight": "bold"}),
            ])
        ], style={"backgroundColor": CARD_BG, "border": f"1px solid {BORDER}",
                  "borderTop": f"3px solid {color}"})

    # ── Layout ───────────────────────────────────────────────────────────────
    app.layout = dbc.Container(fluid=True, style={"backgroundColor": DARK_BG,
                                                    "minHeight": "100vh",
                                                    "padding": "20px"}, children=[

        # ── header ──────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.H2("🔥 Viral Trend Detector",
                        style={"color": TEXT, "fontWeight": "bold", "margin": 0}),
                html.P("Social Media Time Series Analysis · Spike Detection · Early Warning System",
                       style={"color": "#8b949e", "margin": 0, "fontSize": "0.9rem"}),
            ]), width=8),
            dbc.Col(html.Div([
                html.Span("● LIVE", style={"color": GREEN, "fontWeight": "bold",
                                           "fontSize": "0.9rem"}),
                html.Span(f"  |  {len(cache.hashtags)} hashtags tracked",
                          style={"color": "#8b949e", "fontSize": "0.85rem"}),
            ], style={"textAlign": "right", "paddingTop": "10px"}), width=4),
        ], style={"borderBottom": f"1px solid {BORDER}", "paddingBottom": "15px",
                  "marginBottom": "20px"}),

        # ── controls ────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Label("Select Hashtag", style={"color": TEXT, "fontWeight": "bold"}),
                dcc.Dropdown(
                    id="hashtag-select",
                    options=hashtag_options,
                    value=default_tag,
                    clearable=False,
                    style={"backgroundColor": CARD_BG, "color": DARK_BG,
                           "border": f"1px solid {BORDER}"},
                ),
            ], width=4),
            dbc.Col([
                html.Label("Date Range", style={"color": TEXT, "fontWeight": "bold"}),
                dcc.DatePickerRange(
                    id="date-range",
                    min_date_allowed=cache.daily_df["date"].min(),
                    max_date_allowed=cache.daily_df["date"].max(),
                    start_date=cache.daily_df["date"].min(),
                    end_date=cache.daily_df["date"].max(),
                    style={"backgroundColor": CARD_BG},
                ),
            ], width=5),
            dbc.Col([
                html.Label("Quick Filter", style={"color": TEXT, "fontWeight": "bold"}),
                dbc.ButtonGroup([
                    dbc.Button("30d",  id="btn-30d",  color="secondary", size="sm"),
                    dbc.Button("90d",  id="btn-90d",  color="secondary", size="sm"),
                    dbc.Button("180d", id="btn-180d", color="secondary", size="sm"),
                    dbc.Button("All",  id="btn-all",  color="primary",   size="sm"),
                ]),
            ], width=3),
        ], style={"backgroundColor": CARD_BG, "padding": "15px",
                  "borderRadius": "8px", "border": f"1px solid {BORDER}",
                  "marginBottom": "20px"}),

        # ── KPI row ──────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Spinner(html.Div(id="kpi-peak"),    color="info"), width=3),
            dbc.Col(dbc.Spinner(html.Div(id="kpi-viral"),   color="warning"), width=3),
            dbc.Col(dbc.Spinner(html.Div(id="kpi-ews"),     color="danger"), width=3),
            dbc.Col(dbc.Spinner(html.Div(id="kpi-spikes"),  color="success"), width=3),
        ], style={"marginBottom": "20px"}),

        # ── main tabs ────────────────────────────────────────────────────────
        dbc.Tabs(id="main-tabs", active_tab="tab-trend", children=[
            dbc.Tab(label="📈 Trend",        tab_id="tab-trend"),
            dbc.Tab(label="🚨 Anomalies",    tab_id="tab-anomaly"),
            dbc.Tab(label="🔥 Viral Score",  tab_id="tab-viral"),
            dbc.Tab(label="⚡ Early Warning", tab_id="tab-ews"),
            dbc.Tab(label="🏆 Rankings",     tab_id="tab-rankings"),
        ], style={"marginBottom": "10px"}),

        # ── graph output ────────────────────────────────────────────────────
        dbc.Spinner(
            dcc.Graph(id="main-graph", style={"height": "600px"},
                      config={"displayModeBar": True, "displaylogo": False}),
            color="info",
        ),

        # ── footer ──────────────────────────────────────────────────────────
        html.Hr(style={"borderColor": BORDER, "marginTop": "30px"}),
        html.P("Viral Trend Detector · Built with Plotly Dash · Real Kaggle Data",
               style={"color": "#8b949e", "textAlign": "center",
                      "fontSize": "0.8rem"}),
    ])

    # ════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ════════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("kpi-peak",   "children"),
         Output("kpi-viral",  "children"),
         Output("kpi-ews",    "children"),
         Output("kpi-spikes", "children"),
         Output("main-graph", "figure")],
        [Input("hashtag-select", "value"),
         Input("date-range",     "start_date"),
         Input("date-range",     "end_date"),
         Input("main-tabs",      "active_tab")],
    )
    def update_dashboard(hashtag, start_date, end_date, active_tab):
        # ── filter series ────────────────────────────────────────────────────
        series = cache.loader.get_hashtag_series(hashtag)
        if start_date and end_date:
            series = series.loc[start_date:end_date]
        series = series.resample("D").sum().fillna(0)
        series = series[series > 0] if series.sum() > 0 else series

        # ── KPIs ─────────────────────────────────────────────────────────────
        scorer  = ViralScorer()
        vs_df   = scorer.score(series, hashtag)
        vs_summ = scorer.summary(vs_df)

        anom_df = AnomalyDetector().detect(series, hashtag)
        ews_df  = EarlyViralDetector().compute_ews(series, hashtag)

        peak_kpi   = kpi_card("📊 Peak Mentions",
                               f"{int(series.max()):,}", ACCENT)
        viral_kpi  = kpi_card("🔥 Peak Viral Score",
                               f"{vs_summ['peak_viral_score']} / 100", YELLOW)
        ews_kpi    = kpi_card("⚡ Max EWS",
                               f"{ews_df['ews'].max():.0f} / 100", RED)
        spikes_kpi = kpi_card("🚨 Spike Events",
                               str(int(anom_df["is_anomaly"].sum())), GREEN)

        # ── figure ───────────────────────────────────────────────────────────
        if active_tab == "tab-trend":
            fig = fig_trend(series, hashtag)
        elif active_tab == "tab-anomaly":
            fig = fig_anomaly(series, hashtag)
        elif active_tab == "tab-viral":
            fig = fig_viral_score(series, hashtag)
        elif active_tab == "tab-ews":
            fig = fig_ews(series, hashtag)
        elif active_tab == "tab-rankings":
            fig = fig_rankings(cache.ranked)
        else:
            fig = fig_trend(series, hashtag)

        return peak_kpi, viral_kpi, ews_kpi, spikes_kpi, fig

    # ── date range quick-filters ─────────────────────────────────────────────
    @app.callback(
        [Output("date-range", "start_date"),
         Output("date-range", "end_date")],
        [Input("btn-30d",  "n_clicks"),
         Input("btn-90d",  "n_clicks"),
         Input("btn-180d", "n_clicks"),
         Input("btn-all",  "n_clicks")],
        prevent_initial_call=True,
    )
    def quick_filter(n30, n90, n180, nall):
        ctx  = dash.callback_context
        max_date = pd.to_datetime(cache.daily_df["date"].max())
        min_date = pd.to_datetime(cache.daily_df["date"].min())

        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "btn-all"
        offsets   = {"btn-30d": 30, "btn-90d": 90, "btn-180d": 180}
        if triggered in offsets:
            start = max_date - pd.Timedelta(days=offsets[triggered])
            return str(start.date()), str(max_date.date())
        return str(min_date.date()), str(max_date.date())

    return app


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║    🔥  Viral Trend Detector – Dashboard          ║")
    print("╚══════════════════════════════════════════════════╝")
    cache = DataCache()
    app   = build_app(cache)
    print(f"\n🌐 Dashboard running at  http://{DASH_HOST}:{DASH_PORT}/\n")
    app.run(host=DASH_HOST, port=DASH_PORT, debug=DASH_DEBUG)
