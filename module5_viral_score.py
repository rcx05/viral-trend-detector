"""
module5_viral_score.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 5 – Viral Score System  (0 – 100 scale)
  Combines:
    • Magnitude score   – how large the spike is vs baseline
    • Velocity score    – how fast it rose
    • Duration score    – how long it sustained
    • Acceleration score– sharpness of the leading edge
    • Reach score       – estimated unique user exposure
  → Weighted Viral Score  + "Virality Tier" (Cold/Warm/Hot/🔥Mega)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import MIN_VIRAL_SCORE, PLOTS_DIR

Path(PLOTS_DIR).mkdir(exist_ok=True)


# ── score helpers ────────────────────────────────────────────────────────────

def _sigmoid_scale(x: float, steepness: float = 1.0, midpoint: float = 0.0) -> float:
    """Map any real to (0, 100) via logistic function."""
    return 100 / (1 + np.exp(-steepness * (x - midpoint)))


def _minmax(series: pd.Series, clip_percentile: float = 99) -> pd.Series:
    """Min-max scale to [0, 1], clipped at the given upper percentile."""
    clean = series.dropna()
    if len(clean) == 0:
        return pd.Series(0.0, index=series.index)
    upper = np.percentile(clean, clip_percentile)
    lower = float(series.min())
    span  = upper - lower + 1e-9
    return ((series.clip(lower, upper) - lower) / span).fillna(0)


# ════════════════════════════════════════════════════════════════════════════
# Main class
# ════════════════════════════════════════════════════════════════════════════

class ViralScorer:
    """
    Computes a composite 0-100 Viral Score for each day of a hashtag's timeline.
    """

    WEIGHTS = {
        "magnitude":    0.30,
        "velocity":     0.25,
        "acceleration": 0.20,
        "duration":     0.15,
        "reach":        0.10,
    }

    TIERS = [
        (80, "🔥 MEGA VIRAL",  "#ff4444"),
        (60, "🔴 HOT",         "#ff8800"),
        (40, "🟡 WARM",        "#ffcc00"),
        (20, "🔵 RISING",      "#00aaff"),
        ( 0, "⚪ COLD",         "#888888"),
    ]

    def __init__(self):
        pass

    # ── public API ──────────────────────────────────────────────────────────

    def score(self, series: pd.Series, hashtag: str = "") -> pd.DataFrame:
        """
        Parameters
        ----------
        series  : DatetimeIndex Series of daily mention counts
        hashtag : label

        Returns
        -------
        DataFrame with daily viral scores and tier labels.
        """
        df = pd.DataFrame({"mentions": series}).copy()
        df.index = pd.to_datetime(df.index)
        df = df.resample("D").sum().fillna(0)

        sig = df["mentions"].values.astype(float)

        # ── magnitude: mentions / rolling_baseline ───────────────────────────
        baseline = pd.Series(sig).rolling(7, min_periods=1).mean().values
        baseline = np.where(baseline == 0, 1, baseline)
        df["magnitude_raw"] = sig / baseline
        df["magnitude"]     = _minmax(df["magnitude_raw"]) * 100

        # ── velocity: day-over-day absolute change ───────────────────────────
        df["velocity_raw"] = df["mentions"].diff().fillna(0)
        vel_clipped = df["velocity_raw"].clip(lower=0)
        if vel_clipped.sum() == 0:
            df["velocity"] = 0.0
        else:
            df["velocity"] = _minmax(vel_clipped) * 100

        # ── acceleration: change of change ──────────────────────────────────
        df["acceleration_raw"] = df["velocity_raw"].diff().fillna(0)
        accel_clipped = df["acceleration_raw"].clip(lower=0)
        if accel_clipped.sum() == 0:
            df["acceleration"] = 0.0
        else:
            df["acceleration"] = _minmax(accel_clipped) * 100

        # ── duration: rolling count of consecutive above-mean days ───────────
        above_mean = sig > baseline
        consec     = np.zeros(len(sig))
        cnt        = 0
        for i, v in enumerate(above_mean):
            cnt = cnt + 1 if v else 0
            consec[i] = cnt
        df["duration"] = _minmax(
            pd.Series(consec, index=df.index)
            ) * 100

        # ── reach proxy: total cumulative momentum in spike window ───────────
        df["reach"] = _minmax(df["mentions"].rolling(7, min_periods=1).sum()) * 100

        # ── composite score ──────────────────────────────────────────────────
        df["viral_score"] = (
            self.WEIGHTS["magnitude"]    * df["magnitude"]    +
            self.WEIGHTS["velocity"]     * df["velocity"]     +
            self.WEIGHTS["acceleration"] * df["acceleration"] +
            self.WEIGHTS["duration"]     * df["duration"]     +
            self.WEIGHTS["reach"]        * df["reach"]
        )

        df["viral_tier"] = df["viral_score"].apply(self._classify)
        df["is_viral"]   = df["viral_score"] >= MIN_VIRAL_SCORE
        df["hashtag"]    = hashtag
        return df

    def summary(self, df: pd.DataFrame) -> dict:
        """Return a summary dict of the scoring results."""
        vs_clean = df["viral_score"].dropna()
        if vs_clean.empty:
            return {
                "hashtag": df["hashtag"].iloc[0] if "hashtag" in df.columns else "unknown",
                "peak_viral_date": "N/A", "peak_viral_score": 0.0,
                "peak_tier": "⚪ COLD", "viral_days": 0,
                "avg_score": 0.0, "max_mentions": int(df["mentions"].max()),
            }
        peak_idx  = vs_clean.idxmax()
        peak_row  = df.loc[peak_idx]
        return {
            "hashtag":          df["hashtag"].iloc[0],
            "peak_viral_date":  str(peak_row.name.date()),
            "peak_viral_score": round(float(peak_row["viral_score"]), 1),
            "peak_tier":        peak_row["viral_tier"],
            "viral_days":       int(df["is_viral"].sum()),
            "avg_score":        round(float(df["viral_score"].mean()), 1),
            "max_mentions":     int(df["mentions"].max()),
        }

    def plot_viral_score(self, df: pd.DataFrame, hashtag: str) -> str:
        """Dual-axis plot: mentions + viral score gauge → PNG path."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(f"Viral Score System — #{hashtag}", fontsize=16,
                     fontweight="bold", color="white")
        fig.patch.set_facecolor("#0d1117")
        for ax in (ax1, ax2):
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#30363d")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        # ① mentions bars coloured by tier
        colors = [self._tier_color(s) for s in df["viral_score"]]
        ax1.bar(df.index, df["mentions"], color=colors, width=0.8, alpha=0.85)
        ax1.set_title("Daily Mentions (coloured by Virality Tier)")
        ax1.set_ylabel("Mentions", color="white")

        # tier legend
        for threshold, label, color in self.TIERS:
            ax1.plot([], [], color=color, linewidth=6,
                     label=f"{label} (≥{threshold})")
        ax1.legend(fontsize=8, labelcolor="white", facecolor="#21262d",
                   loc="upper left", ncol=2)

        # ② viral score line
        ax2.plot(df.index, df["viral_score"], color="#f78166",
                 linewidth=2, label="Viral Score")
        ax2.fill_between(df.index, df["viral_score"], 0,
                         alpha=0.25, color="#f78166")
        for threshold, label, color in self.TIERS:
            ax2.axhline(threshold, color=color, linestyle="--",
                        linewidth=0.9, alpha=0.7)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Viral Score (0-100)", color="white")
        ax2.set_title("Composite Viral Score")
        ax2.legend(fontsize=9, labelcolor="white", facecolor="#21262d")

        plt.tight_layout()
        safe  = hashtag.replace("/", "_").replace("\\", "_")
        fname = f"{PLOTS_DIR}/viral_score_{safe}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"[Module 5] Viral score plot saved → {fname}")
        return fname

    # ── private ─────────────────────────────────────────────────────────────

    def _classify(self, score: float) -> str:
        for threshold, label, _ in self.TIERS:
            if score >= threshold:
                return label
        return "⚪ COLD"

    def _tier_color(self, score: float) -> str:
        for threshold, _, color in self.TIERS:
            if score >= threshold:
                return color
        return "#888888"


# ── standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from module1_data_loader import SocialMediaLoader
    loader  = SocialMediaLoader()
    df_all  = loader.load()
    tag     = loader.top_hashtags(1)[0]
    series  = loader.get_hashtag_series(tag)

    scorer = ViralScorer()
    result = scorer.score(series, tag)
    print(scorer.summary(result))
    scorer.plot_viral_score(result, tag)
