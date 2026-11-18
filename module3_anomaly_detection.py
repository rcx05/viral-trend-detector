"""
module3_anomaly_detection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 3 – Viral Spike / Anomaly Detection
  • Z-score anomaly flagging
  • IQR-based outlier detection
  • PELT change-point detection (pure Python via scipy)
  • Spike characterisation:
      – start date, peak date, peak magnitude
      – rise speed (days to peak), duration above threshold
      – whether anomaly or seasonal
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
from scipy import stats
from scipy.signal import find_peaks
from pathlib import Path

warnings.filterwarnings("ignore")

from config import Z_SCORE_THRESHOLD, ROLLING_WINDOW, SPIKE_MULTIPLIER, PLOTS_DIR

Path(PLOTS_DIR).mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Pure-Python PELT change-point detection (no ruptures dependency)
# ════════════════════════════════════════════════════════════════════════════

def _pelt_changepoints(signal: np.ndarray, penalty: float = 3.0) -> list[int]:
    """
    Simplified PELT-style change-point detection using cumulative sum (CUSUM).
    Returns list of indices where significant structural breaks occur.
    """
    n = len(signal)
    if n < 10:
        return []

    # Normalise
    mu, sigma = signal.mean(), signal.std() + 1e-9
    z = (signal - mu) / sigma

    # Cumulative sum chart
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    k = penalty  # slack

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + z[i] - k)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - z[i] - k)

    # Find peaks in CUSUM as change-points
    threshold = penalty * 2
    cp_pos, _ = find_peaks(cusum_pos, height=threshold, distance=7)
    cp_neg, _ = find_peaks(cusum_neg, height=threshold, distance=7)

    changepoints = sorted(set(cp_pos.tolist() + cp_neg.tolist()))
    return changepoints


# ════════════════════════════════════════════════════════════════════════════
# Main class
# ════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Detects viral spikes and anomalies in a daily mention time-series.
    """

    def __init__(
        self,
        z_threshold: float = Z_SCORE_THRESHOLD,
        spike_multiplier: float = SPIKE_MULTIPLIER,
        window: int = ROLLING_WINDOW,
        cpd_penalty: float = 3.0,
    ):
        self.z_threshold      = z_threshold
        self.spike_multiplier = spike_multiplier
        self.window           = window
        self.cpd_penalty      = cpd_penalty

    # ── public API ──────────────────────────────────────────────────────────

    def detect(self, series: pd.Series, hashtag: str = "") -> pd.DataFrame:
        """
        Parameters
        ----------
        series  : DatetimeIndex Series of daily mention counts
        hashtag : label

        Returns
        -------
        DataFrame enriched with anomaly flags and spike metadata.
        """
        df = pd.DataFrame({"mentions": series}).copy()
        df.index = pd.to_datetime(df.index)
        df = df.resample("D").sum().fillna(0)

        sig = df["mentions"].values.astype(float)
        n   = len(sig)

        # ── Z-score ─────────────────────────────────────────────────────────
        rolling_mean = pd.Series(sig).rolling(self.window, min_periods=1).mean().values
        rolling_std  = pd.Series(sig).rolling(self.window, min_periods=1).std().fillna(1).values
        rolling_std  = np.where(rolling_std == 0, 1, rolling_std)
        z_scores     = (sig - rolling_mean) / rolling_std

        df["z_score"]        = z_scores
        df["rolling_mean"]   = rolling_mean
        df["rolling_std"]    = rolling_std
        df["z_anomaly"]      = z_scores > self.z_threshold

        # ── IQR outlier ──────────────────────────────────────────────────────
        q1, q3  = np.percentile(sig, 25), np.percentile(sig, 75)
        iqr     = q3 - q1
        iqr_hi  = q3 + 1.5 * iqr
        df["iqr_anomaly"] = sig > iqr_hi

        # ── Spike multiplier ─────────────────────────────────────────────────
        df["spike_flag"] = sig > (self.spike_multiplier * rolling_mean)

        # ── Combined anomaly flag ────────────────────────────────────────────
        df["is_anomaly"] = df["z_anomaly"] | df["iqr_anomaly"] | df["spike_flag"]

        # ── Change-point detection ───────────────────────────────────────────
        cps = _pelt_changepoints(sig, self.cpd_penalty)
        df["is_changepoint"] = False
        df.iloc[cps, df.columns.get_loc("is_changepoint")] = True

        # ── Spike event characterisation ─────────────────────────────────────
        df = self._characterise_spikes(df, hashtag)

        df["hashtag"] = hashtag
        return df

    def get_spike_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows where a spike started (first day of each spike event)."""
        return df[df.get("spike_event_start", pd.Series(False, index=df.index))].copy()

    def plot_anomalies(self, df: pd.DataFrame, hashtag: str) -> str:
        """Save annotated anomaly plot → returns PNG path."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f"Anomaly Detection — #{hashtag}", fontsize=16,
                     fontweight="bold", color="white")
        fig.patch.set_facecolor("#0d1117")

        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#30363d")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        dates = df.index

        # ① mentions + anomalies + change-points
        axes[0].plot(dates, df["mentions"], color="#58a6ff", linewidth=1.4,
                     label="Mentions", alpha=0.85)
        axes[0].plot(dates, df["rolling_mean"], color="#e3b341",
                     linewidth=2, label="Rolling Mean")
        axes[0].fill_between(dates,
                             df["rolling_mean"] - 2 * df["rolling_std"],
                             df["rolling_mean"] + 2 * df["rolling_std"],
                             alpha=0.12, color="#e3b341", label="±2σ Band")

        anomaly_dates = df.index[df["is_anomaly"]]
        anomaly_vals  = df.loc[anomaly_dates, "mentions"]
        axes[0].scatter(anomaly_dates, anomaly_vals, color="#f85149",
                        zorder=5, s=50, label="🚨 Anomaly", marker="^")

        cp_dates = df.index[df["is_changepoint"]]
        for cp in cp_dates:
            axes[0].axvline(cp, color="#bc8cff", linestyle="--",
                            linewidth=1.2, alpha=0.75)
        # legend entry
        axes[0].plot([], [], color="#bc8cff", linestyle="--",
                     linewidth=1.2, label="Change Point")
        axes[0].set_title("Mentions + Anomalies + Change Points")
        axes[0].legend(fontsize=9, labelcolor="white", facecolor="#21262d",
                       loc="upper left")

        # ② Z-score
        axes[1].plot(dates, df["z_score"], color="#79c0ff", linewidth=1.2)
        axes[1].axhline(self.z_threshold,  color="#f85149", linestyle="--",
                        linewidth=1.5, label=f"Z = {self.z_threshold}")
        axes[1].axhline(-self.z_threshold, color="#f85149", linestyle="--",
                        linewidth=1.5)
        axes[1].axhline(0, color="white", linewidth=0.7)
        axes[1].fill_between(dates, df["z_score"], 0,
                             where=df["z_score"] > self.z_threshold,
                             color="#f85149", alpha=0.3, label="Spike Zone")
        axes[1].set_title("Z-Score (Rolling)")
        axes[1].legend(fontsize=9, labelcolor="white", facecolor="#21262d")

        # ③ spike flag timeline
        spike_binary = df["is_anomaly"].astype(int)
        axes[2].fill_between(dates, spike_binary, 0,
                             where=spike_binary == 1, step="post",
                             color="#f78166", alpha=0.8, label="Viral Event")
        axes[2].set_ylim(-0.1, 1.5)
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(["Normal", "Viral"], color="white")
        axes[2].set_title("Viral Event Timeline")
        axes[2].legend(fontsize=9, labelcolor="white", facecolor="#21262d")
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[2].xaxis.set_major_locator(mdates.MonthLocator())

        plt.tight_layout()
        safe  = hashtag.replace("/", "_").replace("\\", "_")
        fname = f"{PLOTS_DIR}/anomaly_{safe}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"[Module 3] Anomaly plot saved → {fname}")
        return fname

    # ── private ─────────────────────────────────────────────────────────────

    def _characterise_spikes(self, df: pd.DataFrame, hashtag: str) -> pd.DataFrame:
        """
        For each contiguous anomaly cluster, compute:
          spike_event_id, spike_event_start, peak_date,
          peak_magnitude, rise_days, duration_days, growth_factor
        """
        df = df.copy()
        for col in ["spike_event_id", "spike_event_start", "peak_date",
                    "peak_magnitude", "rise_days", "duration_days", "growth_factor"]:
            df[col] = np.nan

        is_anom = df["is_anomaly"].values
        event_id = 0
        i = 0
        while i < len(is_anom):
            if is_anom[i]:
                # find end of contiguous block
                j = i
                while j < len(is_anom) and is_anom[j]:
                    j += 1
                # slice
                block = df.iloc[i:j]
                peak_idx  = block["mentions"].idxmax()
                peak_val  = block["mentions"].max()
                base_val  = df["rolling_mean"].iloc[i]
                rise_days = (peak_idx - block.index[0]).days + 1
                duration  = j - i
                g_factor  = (peak_val / base_val) if base_val > 0 else np.nan

                df.iloc[i, df.columns.get_loc("spike_event_id")]    = event_id
                df.iloc[i, df.columns.get_loc("spike_event_start")] = True
                df.loc[df.index[i]:df.index[j - 1], "spike_event_id"] = event_id
                df.loc[df.index[i]:df.index[j - 1], "peak_date"]      = str(peak_idx.date())
                df.loc[df.index[i]:df.index[j - 1], "peak_magnitude"] = peak_val
                df.loc[df.index[i]:df.index[j - 1], "rise_days"]      = rise_days
                df.loc[df.index[i]:df.index[j - 1], "duration_days"]  = duration
                df.loc[df.index[i]:df.index[j - 1], "growth_factor"]  = g_factor

                event_id += 1
                i = j
            else:
                i += 1

        df["spike_event_start"] = df["spike_event_start"].fillna(False).astype(bool)
        return df


# ── standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from module1_data_loader import SocialMediaLoader
    loader = SocialMediaLoader()
    df_all = loader.load()
    tag    = loader.top_hashtags(1)[0]
    series = loader.get_hashtag_series(tag)

    detector = AnomalyDetector()
    result   = detector.detect(series, tag)
    path     = detector.plot_anomalies(result, tag)

    spikes = detector.get_spike_events(result)
    print(f"\n🚨 Detected {len(spikes)} spike start(s) for #{tag}:")
    if not spikes.empty:
        print(spikes[["mentions", "z_score", "peak_magnitude",
                       "rise_days", "duration_days", "growth_factor"]].to_string())
    print(f"\nPlot → {path}")
