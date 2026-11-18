"""
module8_early_detection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 8 – Early Viral Detection
  Predicts whether a hashtag is *about to go viral* BEFORE the
  main spike using leading indicators:

  • Acceleration breakout – 2nd-derivative inflection
  • Short-window z-score (3-day) vs long-window (7-day)
  • Momentum divergence index (MDI)
  • Early Warning Score (EWS, 0-100)
  • Alerts: 🟡 Watch / 🔴 Warning / 🚨 Pre-Viral
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

warnings.filterwarnings("ignore")

from config import PLOTS_DIR, EARLY_DETECT_WINDOW

Path(PLOTS_DIR).mkdir(exist_ok=True)


ALERT_LEVELS = [
    (75, "🚨 PRE-VIRAL",  "#ff0000"),
    (50, "🔴 WARNING",    "#ff6600"),
    (25, "🟡 WATCH",      "#ffcc00"),
    ( 0, "✅ NORMAL",      "#3fb950"),
]


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window, min_periods=1).mean()
    sig = series.rolling(window, min_periods=1).std().fillna(1).replace(0, 1)
    return (series - mu) / sig


class EarlyViralDetector:
    """
    Generates early-warning signals 1–3 days before a viral spike.
    """

    def __init__(self, short_w: int = 3, long_w: int = 7,
                 detect_window: int = EARLY_DETECT_WINDOW):
        self.short_w       = short_w
        self.long_w        = long_w
        self.detect_window = detect_window

    # ── public ──────────────────────────────────────────────────────────────

    def compute_ews(self, series: pd.Series, hashtag: str = "") -> pd.DataFrame:
        """
        Compute Early Warning Score time-series.

        Returns
        -------
        DataFrame with EWS, alert level, and leading indicators.
        """
        df = pd.DataFrame({"mentions": series}).copy()
        df.index = pd.to_datetime(df.index)
        df = df.resample("D").sum().fillna(0)

        sig = df["mentions"].values.astype(float)
        s   = pd.Series(sig, index=df.index)

        # ── leading indicators ───────────────────────────────────────────────
        # 1) short-window z-score (sensitive, reacts fast)
        df["z_short"] = _rolling_z(s, self.short_w).clip(-5, 5)

        # 2) long-window z-score (stable baseline)
        df["z_long"]  = _rolling_z(s, self.long_w).clip(-5, 5)

        # 3) Momentum Divergence Index: short z − long z
        #    Positive → short window rising faster than long → early sign
        df["mdi"] = df["z_short"] - df["z_long"]

        # 4) Acceleration (2nd derivative of rolling mean)
        rm = s.rolling(self.long_w, min_periods=1).mean()
        df["accel"] = rm.diff().diff().fillna(0)
        df["accel_norm"] = df["accel"] / (df["accel"].abs().max() + 1e-9)

        # 5) Velocity ratio: today / yesterday
        df["vel_ratio"] = s / (s.shift(1).replace(0, np.nan).fillna(1))
        df["vel_ratio"] = df["vel_ratio"].clip(0, 20)

        # ── EWS composition (0-100) ──────────────────────────────────────────
        def _norm(x: pd.Series, lo=0, hi=5) -> pd.Series:
            """Clip to [lo, hi] then normalise to [0, 1]."""
            return ((x.clip(lo, hi) - lo) / (hi - lo + 1e-9)).fillna(0)

        ews = (
            35 * _norm(df["mdi"],       -2,  5)   +  # divergence
            25 * _norm(df["z_short"],    0,  5)   +  # short-term z
            20 * _norm(df["accel_norm"], 0,  1)   +  # acceleration
            20 * _norm(df["vel_ratio"],  1, 10)      # velocity ratio
        )

        df["ews"] = ews.clip(0, 100)
        df["alert_level"] = df["ews"].apply(self._alert)
        df["hashtag"] = hashtag

        # ── look-ahead accuracy (did the EWS correctly warn?) ────────────────
        df["actual_viral"] = _rolling_z(s, self.long_w) > 2.5
        df["warned"] = df["ews"].shift(self.detect_window).fillna(0) > 50
        return df

    def prediction_accuracy(self, df: pd.DataFrame) -> dict:
        """Precision / Recall of EWS vs actual viral days."""
        tp = int((df["warned"] & df["actual_viral"]).sum())
        fp = int((df["warned"] & ~df["actual_viral"]).sum())
        fn = int((~df["warned"] & df["actual_viral"]).sum())
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        return {
            "true_positives":  tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision":       round(precision, 3),
            "recall":          round(recall, 3),
            "f1_score":        round(f1, 3),
        }

    def plot_ews(self, df: pd.DataFrame, hashtag: str) -> str:
        """3-panel EWS plot → PNG path."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1.5, 1]})
        fig.suptitle(f"Early Viral Detection — #{hashtag}", fontsize=16,
                     fontweight="bold", color="white")
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#30363d")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        dates = df.index

        # ① Mentions + EWS alert zones
        ax0 = axes[0]
        ax0.plot(dates, df["mentions"], color="#58a6ff", linewidth=1.5,
                 label="Daily Mentions", alpha=0.85)
        # colour background by alert
        for _, row in df.iterrows():
            color = self._alert_color(row["ews"])
            ax0.axvspan(row.name, row.name + pd.Timedelta(days=1),
                        alpha=0.07, color=color, linewidth=0)
        # highlight pre-viral warnings
        warned_mask = df["ews"] > 50
        if warned_mask.any():
            ax0.scatter(dates[warned_mask], df.loc[warned_mask, "mentions"],
                        color="#ffcc00", s=40, zorder=6,
                        label="Early Warning Signal", marker="v")
        ax0.set_title("Daily Mentions + Warning Signals")
        ax0.set_ylabel("Mentions", color="white")
        ax0.legend(fontsize=9, labelcolor="white", facecolor="#21262d")

        # ② EWS gauge
        ax1 = axes[1]
        alert_colors = [self._alert_color(v) for v in df["ews"]]
        ax1.bar(dates, df["ews"], color=alert_colors, width=0.8, alpha=0.9)
        for threshold, label, color in ALERT_LEVELS[:-1]:
            ax1.axhline(threshold, color=color, linestyle="--",
                        linewidth=1.1, alpha=0.8, label=f"{label} (≥{threshold})")
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("EWS Score (0-100)", color="white")
        ax1.set_title("Early Warning Score")
        ax1.legend(fontsize=9, labelcolor="white", facecolor="#21262d", ncol=2)

        # ③ Momentum Divergence Index
        ax2 = axes[2]
        ax2.fill_between(dates, df["mdi"], 0,
                         where=df["mdi"] >= 0, color="#3fb950", alpha=0.7,
                         label="MDI Positive")
        ax2.fill_between(dates, df["mdi"], 0,
                         where=df["mdi"] < 0,  color="#f85149", alpha=0.7,
                         label="MDI Negative")
        ax2.axhline(0, color="white", linewidth=0.7)
        ax2.set_title("Momentum Divergence Index (Short-Z − Long-Z)")
        ax2.set_ylabel("MDI", color="white")
        ax2.legend(fontsize=9, labelcolor="white", facecolor="#21262d")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        plt.tight_layout()
        safe  = hashtag.replace("/", "_").replace("\\", "_")
        fname = f"{PLOTS_DIR}/early_detection_{safe}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"[Module 8] Early detection plot saved → {fname}")
        return fname

    # ── private ─────────────────────────────────────────────────────────────

    def _alert(self, score: float) -> str:
        for threshold, label, _ in ALERT_LEVELS:
            if score >= threshold:
                return label
        return "✅ NORMAL"

    def _alert_color(self, score: float) -> str:
        for threshold, _, color in ALERT_LEVELS:
            if score >= threshold:
                return color
        return "#3fb950"


# ── standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from module1_data_loader import SocialMediaLoader
    loader = SocialMediaLoader()
    df_all = loader.load()
    tag    = loader.top_hashtags(1)[0]
    series = loader.get_hashtag_series(tag)

    detector = EarlyViralDetector()
    ews_df   = detector.compute_ews(series, tag)
    acc      = detector.prediction_accuracy(ews_df)
    path     = detector.plot_ews(ews_df, tag)

    print(f"\n📊 EWS Accuracy for #{tag}:")
    for k, v in acc.items():
        print(f"   {k:20s}: {v}")
    print(f"\nPlot → {path}")
