"""
module7_event_detection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 7 – Event Detection
  • Classifies each spike as: Event-Driven / Organic / Bot-like / Seasonal
  • Inter-event timeline
  • Co-occurring hashtag analysis (which tags spiked together?)
  • Cross-platform event matching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

from config import PLOTS_DIR, ROLLING_WINDOW

Path(PLOTS_DIR).mkdir(exist_ok=True)


EVENT_TYPES = {
    "MEGA_ORGANIC":  {"color": "#ff4444", "marker": "★"},
    "BOT_SUSPECTED": {"color": "#bc8cff", "marker": "⬟"},
    "SEASONAL":      {"color": "#79c0ff", "marker": "◆"},
    "EVENT_DRIVEN":  {"color": "#f78166", "marker": "●"},
    "ORGANIC":       {"color": "#3fb950", "marker": "▲"},
    "NORMAL":        {"color": "#888888", "marker": "·"},
}


class EventDetector:
    """
    Classifies viral spikes into event types and extracts event metadata.
    """

    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window

    # ── public ──────────────────────────────────────────────────────────────

    def classify_events(self, anomaly_df: pd.DataFrame, viral_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        anomaly_df : from Module 3 (AnomalyDetector.detect)
        viral_df   : from Module 5 (ViralScorer.score)

        Returns
        -------
        Merged DataFrame with event_type column.
        """
        df = anomaly_df.copy()
        if "viral_score" in viral_df.columns:
            df["viral_score"] = viral_df["viral_score"].reindex(df.index, fill_value=0)
            df["viral_tier"]  = viral_df["viral_tier"].reindex(df.index, fill_value="⚪ COLD")
        else:
            df["viral_score"] = 0
            df["viral_tier"]  = "⚪ COLD"

        df["event_type"] = df.apply(self._classify_row, axis=1)
        return df

    def _classify_row(self, row: pd.Series) -> str:
        """Rule-based spike classification."""
        is_anom   = bool(row.get("is_anomaly", False))
        z         = float(row.get("z_score", 0))
        growth    = float(row.get("growth_factor", 1) or 1)
        viral     = float(row.get("viral_score", 0))
        rise_days = float(row.get("rise_days", 7) or 7)
        duration  = float(row.get("duration_days", 1) or 1)

        if not is_anom:
            return "NORMAL"

        # Bot-like: huge spike that resolves in ≤2 days
        if growth > 10 and duration <= 2:
            return "BOT_SUSPECTED"

        # Mega organic: very high viral score, gradual rise
        if viral >= 80 and rise_days >= 3:
            return "MEGA_ORGANIC"

        # Seasonal: moderate z-score, repeatable pattern
        if 1.5 < z <= 3 and rise_days >= 5:
            return "SEASONAL"

        # Event-driven: sharp, large growth
        if growth > 5 and rise_days <= 3:
            return "EVENT_DRIVEN"

        return "ORGANIC"

    def co_occurrence_analysis(self, daily_df: pd.DataFrame,
                                target_hashtag: str, window_days: int = 3) -> pd.DataFrame:
        """
        Find hashtags that spiked within ±window_days of target_hashtag's spikes.
        """
        # get spike dates for target
        target = daily_df[daily_df["hashtag"] == target_hashtag].copy()
        target["date"] = pd.to_datetime(target["date"])
        rm = target.set_index("date")["mentions"].rolling(7, min_periods=1).mean()
        spike_dates = rm[target.set_index("date")["mentions"] > 3 * rm].index.tolist()

        if not spike_dates:
            return pd.DataFrame(columns=["hashtag", "co_spike_count"])

        co = []
        for ht in daily_df["hashtag"].unique():
            if ht == target_hashtag:
                continue
            ht_df = daily_df[daily_df["hashtag"] == ht].copy()
            ht_df["date"] = pd.to_datetime(ht_df["date"])
            ht_rm = ht_df.set_index("date")["mentions"].rolling(7, min_periods=1).mean()
            ht_spikes = set(ht_rm[ht_df.set_index("date")["mentions"] > 3 * ht_rm].index.tolist())

            count = 0
            for sd in spike_dates:
                nearby = {sd + pd.Timedelta(days=d) for d in range(-window_days, window_days + 1)}
                if ht_spikes & nearby:
                    count += 1

            if count > 0:
                co.append({"hashtag": ht, "co_spike_count": count})

        return pd.DataFrame(co).sort_values("co_spike_count", ascending=False)

    def event_timeline_plot(self, event_df: pd.DataFrame, hashtag: str) -> str:
        """Annotated event timeline → PNG path."""
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#30363d")

        # background mentions
        ax.fill_between(event_df.index, event_df["mentions"], 0,
                        color="#58a6ff", alpha=0.25, label="Mentions")
        ax.plot(event_df.index, event_df["mentions"],
                color="#58a6ff", linewidth=1.2, alpha=0.6)

        # scatter events
        for etype, cfg in EVENT_TYPES.items():
            if etype == "NORMAL":
                continue
            mask = event_df["event_type"] == etype
            if mask.any():
                ax.scatter(
                    event_df.index[mask],
                    event_df.loc[mask, "mentions"],
                    color=cfg["color"], s=80, zorder=5,
                    label=etype.replace("_", " ").title(),
                    edgecolors="white", linewidth=0.5,
                )

        ax.set_title(f"Event Classification Timeline — #{hashtag}",
                     fontsize=14, color="white", pad=12)
        ax.set_ylabel("Daily Mentions", color="white")
        ax.set_xlabel("Date", color="white")
        ax.legend(fontsize=9, labelcolor="white", facecolor="#21262d", loc="upper left")

        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30, color="white")

        plt.tight_layout()
        safe  = hashtag.replace("/", "_").replace("\\", "_")
        fname = f"{PLOTS_DIR}/events_{safe}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"[Module 7] Event timeline saved → {fname}")
        return fname


# ── standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from module1_data_loader import SocialMediaLoader
    from module3_anomaly_detection import AnomalyDetector
    from module5_viral_score import ViralScorer

    loader   = SocialMediaLoader()
    df_all   = loader.load()
    tag      = loader.top_hashtags(1)[0]
    series   = loader.get_hashtag_series(tag)

    anom_df  = AnomalyDetector().detect(series, tag)
    viral_df = ViralScorer().score(series, tag)

    ed = EventDetector()
    ev = ed.classify_events(anom_df, viral_df)
    print(ev["event_type"].value_counts())
    ed.event_timeline_plot(ev, tag)
