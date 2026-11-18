"""
module2_trend_analysis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 2 – Trend Analysis
  • Rolling mean / std
  • Momentum & acceleration
  • Seasonal decomposition (STL)
  • Growth rate & velocity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from pathlib import Path

warnings.filterwarnings("ignore")

from config import ROLLING_WINDOW, PLOTS_DIR

Path(PLOTS_DIR).mkdir(exist_ok=True)


class TrendAnalyzer:
    """Computes rolling statistics, momentum, and seasonal decomposition."""

    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window

    # ── public ──────────────────────────────────────────────────────────────

    def analyze(self, series: pd.Series, hashtag: str = "") -> pd.DataFrame:
        """
        Parameters
        ----------
        series : pd.Series  DatetimeIndex, daily mention counts
        hashtag: str        Label for plots

        Returns
        -------
        pd.DataFrame with rich trend columns.
        """
        df = pd.DataFrame({"mentions": series}).copy()
        df.index = pd.to_datetime(df.index)
        df = df.resample("D").sum().fillna(0)

        # rolling stats
        df["rm"]   = df["mentions"].rolling(self.window, min_periods=1).mean()
        df["rstd"] = df["mentions"].rolling(self.window, min_periods=1).std().fillna(0)

        # momentum (1st derivative approximation)
        df["momentum"]     = df["rm"].diff().fillna(0)
        # acceleration (2nd derivative)
        df["acceleration"] = df["momentum"].diff().fillna(0)

        # growth rate %
        df["growth_rate"]  = df["mentions"].pct_change().replace([np.inf, -np.inf], 0).fillna(0) * 100

        # cumulative sum (virality footprint)
        df["cumulative"]   = df["mentions"].cumsum()

        # STL decomposition (requires ≥ 14 obs)
        df["trend_stl"]    = np.nan
        df["seasonal_stl"] = np.nan
        df["residual_stl"] = np.nan
        if len(df) >= 14:
            try:
                stl = STL(df["mentions"], period=7, robust=True)
                res = stl.fit()
                df["trend_stl"]    = res.trend
                df["seasonal_stl"] = res.seasonal
                df["residual_stl"] = res.resid
            except Exception as exc:
                print(f"[Module 2] STL failed for {hashtag}: {exc}")

        df["hashtag"] = hashtag
        return df

    def plot_trend(self, df: pd.DataFrame, hashtag: str) -> str:
        """4-panel trend analysis plot → saved PNG path."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        fig.suptitle(f"Trend Analysis — #{hashtag}", fontsize=16, fontweight="bold", y=1.01)
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#30363d")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")

        # ① raw + rolling mean
        axes[0].plot(df.index, df["mentions"], color="#58a6ff", alpha=0.6,
                     linewidth=1.2, label="Daily Mentions")
        axes[0].plot(df.index, df["rm"],       color="#f78166", linewidth=2,
                     label=f"{self.window}-day Rolling Mean")
        axes[0].fill_between(df.index,
                             df["rm"] - df["rstd"],
                             df["rm"] + df["rstd"],
                             alpha=0.15, color="#f78166", label="±1 STD Band")
        axes[0].set_title("Daily Mentions + Rolling Statistics")
        axes[0].legend(fontsize=9, labelcolor="white", facecolor="#21262d")

        # ② momentum
        axes[1].fill_between(df.index, df["momentum"], 0,
                             where=df["momentum"] >= 0, color="#3fb950", alpha=0.7, label="Positive")
        axes[1].fill_between(df.index, df["momentum"], 0,
                             where=df["momentum"] < 0,  color="#f85149", alpha=0.7, label="Negative")
        axes[1].axhline(0, color="white", linewidth=0.7, linestyle="--")
        axes[1].set_title("Momentum (Δ Rolling Mean)")
        axes[1].legend(fontsize=9, labelcolor="white", facecolor="#21262d")

        # ③ STL decomposition
        if df["trend_stl"].notna().any():
            axes[2].plot(df.index, df["trend_stl"],    color="#e3b341", linewidth=2, label="Trend")
            axes[2].plot(df.index, df["seasonal_stl"], color="#79c0ff", linewidth=1.2, alpha=0.8, label="Seasonal")
            axes[2].set_title("STL Decomposition")
            axes[2].legend(fontsize=9, labelcolor="white", facecolor="#21262d")
        else:
            axes[2].text(0.5, 0.5, "Insufficient data for STL",
                         transform=axes[2].transAxes, ha="center", va="center",
                         color="gray", fontsize=13)
            axes[2].set_title("STL Decomposition")

        # ④ growth rate
        axes[3].bar(df.index, df["growth_rate"], color=np.where(df["growth_rate"] >= 0,
                                                                  "#3fb950", "#f85149"),
                    width=0.8, alpha=0.85)
        axes[3].axhline(0, color="white", linewidth=0.7)
        axes[3].set_title("Daily Growth Rate (%)")
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[3].xaxis.set_major_locator(mdates.MonthLocator())

        plt.tight_layout()
        safe  = hashtag.replace("/", "_").replace("\\", "_")
        fname = f"{PLOTS_DIR}/trend_{safe}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"[Module 2] Trend plot saved → {fname}")
        return fname


# ── standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from module1_data_loader import SocialMediaLoader
    loader = SocialMediaLoader()
    df     = loader.load()
    tag    = loader.top_hashtags(1)[0]
    series = loader.get_hashtag_series(tag)

    analyzer = TrendAnalyzer()
    result   = analyzer.analyze(series, tag)
    path     = analyzer.plot_trend(result, tag)
    print(result.tail(10))
    print(f"\nPlot → {path}")
