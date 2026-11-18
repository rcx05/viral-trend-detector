"""
module4_hashtag_trends.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 4 – Hashtag-Level Trend Detection
  • Ranks every hashtag by various virality dimensions
  • Heatmap of top-N hashtags × week
  • Cross-hashtag correlation matrix
  • Burst detection per hashtag (sudden mention explosions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

from config import PLOTS_DIR, ROLLING_WINDOW

Path(PLOTS_DIR).mkdir(exist_ok=True)


class HashtagTrendDetector:
    """Multi-hashtag trend ranking and burst detection."""

    def __init__(self, top_n: int = 20, window: int = ROLLING_WINDOW):
        self.top_n  = top_n
        self.window = window

    # ── public ──────────────────────────────────────────────────────────────

    def rank_hashtags(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-hashtag ranking metrics.

        Parameters
        ----------
        daily_df : output of SocialMediaLoader.load()

        Returns
        -------
        pd.DataFrame  sorted by viral_rank (ascending → #1 is most viral)
        """
        grp = daily_df.groupby("hashtag")

        metrics = pd.DataFrame({
            "total_mentions":  grp["mentions"].sum(),
            "peak_mentions":   grp["mentions"].max(),
            "avg_mentions":    grp["mentions"].mean(),
            "active_days":     grp["mentions"].count(),
            "max_growth_rate": grp["pct_change"].max() if "pct_change" in daily_df.columns
                               else grp["mentions"].apply(lambda x: x.pct_change().max()),
            "std_mentions":    grp["mentions"].std().fillna(0),
        })

        # coefficient of variation: high = volatile / bursty
        metrics["cv"] = metrics["std_mentions"] / (metrics["avg_mentions"] + 1e-9)

        # burst score: how quickly it reached its peak
        def _burst(grp_df: pd.DataFrame) -> float:
            grp_df = grp_df.sort_values("date")
            peak_idx = grp_df["mentions"].idxmax()
            peak_day = grp_df.loc[peak_idx, "date"]
            start    = grp_df["date"].min()
            days_to_peak = (peak_day - start).days + 1
            return grp_df.loc[peak_idx, "mentions"] / (days_to_peak + 1)

        metrics["burst_score"] = grp.apply(_burst, include_groups=False)

        # composite rank (higher = more viral)
        for col in ["total_mentions", "peak_mentions", "max_growth_rate",
                    "cv", "burst_score"]:
            metrics[f"{col}_rank"] = metrics[col].rank(pct=True)

        metrics["viral_composite"] = (
            0.25 * metrics["total_mentions_rank"] +
            0.25 * metrics["peak_mentions_rank"]  +
            0.20 * metrics["max_growth_rate_rank"] +
            0.15 * metrics["cv_rank"]              +
            0.15 * metrics["burst_score_rank"]
        ) * 100

        metrics["viral_rank"] = metrics["viral_composite"].rank(ascending=False).astype(int)
        metrics.sort_values("viral_rank", inplace=True)
        return metrics

    def weekly_heatmap(self, daily_df: pd.DataFrame, hashtag_list: list[str] | None = None) -> str:
        """
        Heatmap: top-N hashtags (rows) × ISO week (columns).
        Returns saved PNG path.
        """
        if hashtag_list is None:
            ranked  = self.rank_hashtags(daily_df)
            hashtag_list = ranked.index[:self.top_n].tolist()

        sub = daily_df[daily_df["hashtag"].isin(hashtag_list)].copy()
        sub["week"] = pd.to_datetime(sub["date"]).dt.to_period("W").astype(str)

        pivot = (
            sub.groupby(["hashtag", "week"])["mentions"]
            .sum()
            .unstack(fill_value=0)
        )
        # keep only top_n hashtags
        pivot = pivot.loc[pivot.index.isin(hashtag_list)]
        # limit columns to last 26 weeks
        pivot = pivot.iloc[:, -26:]

        fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) // 2), max(6, len(pivot) // 2)))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            linewidths=0.3,
            linecolor="#30363d",
            cbar_kws={"label": "Weekly Mentions"},
            fmt=".0f",
            annot=len(pivot) <= 15,   # annotate only if fits
        )
        ax.set_title("Weekly Mentions Heatmap – Top Hashtags",
                     fontsize=14, color="white", pad=12)
        ax.set_xlabel("Week", color="white")
        ax.set_ylabel("Hashtag", color="white")
        ax.tick_params(colors="white", labelsize=8)
        plt.xticks(rotation=45, ha="right", color="white", fontsize=7)
        plt.yticks(color="white", fontsize=8)

        fname = f"{PLOTS_DIR}/hashtag_heatmap.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"[Module 4] Heatmap saved → {fname}")
        return fname

    def correlation_matrix(self, daily_df: pd.DataFrame, hashtag_list: list[str] | None = None) -> str:
        """Pearson correlation between top hashtags → saved PNG."""
        if hashtag_list is None:
            ranked = self.rank_hashtags(daily_df)
            hashtag_list = ranked.index[:min(self.top_n, 15)].tolist()

        sub   = daily_df[daily_df["hashtag"].isin(hashtag_list)].copy()
        pivot = sub.pivot_table(index="date", columns="hashtag",
                                values="mentions", aggfunc="sum").fillna(0)
        corr  = pivot.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, ax=ax,
            cmap="coolwarm", center=0,
            vmin=-1, vmax=1,
            annot=True, fmt=".2f",
            linewidths=0.5, linecolor="#30363d",
            cbar_kws={"label": "Pearson r"},
        )
        ax.set_title("Hashtag Mention Correlation Matrix",
                     fontsize=14, color="white", pad=12)
        ax.tick_params(colors="white", labelsize=8)

        fname = f"{PLOTS_DIR}/hashtag_correlation.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"[Module 4] Correlation matrix saved → {fname}")
        return fname

    def burst_detection(self, daily_df: pd.DataFrame, hashtag: str,
                        threshold_pct: float = 200.0) -> pd.DataFrame:
        """
        Detect days where mentions exceeded threshold_pct % of the rolling mean.
        Returns burst-event rows.
        """
        sub = daily_df[daily_df["hashtag"] == hashtag].copy().sort_values("date")
        sub["rm"]    = sub["mentions"].rolling(self.window, min_periods=1).mean()
        sub["burst"] = sub["mentions"] > (sub["rm"] * (1 + threshold_pct / 100))
        return sub[sub["burst"]]


# ── standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from module1_data_loader import SocialMediaLoader
    loader  = SocialMediaLoader()
    df      = loader.load()
    detector = HashtagTrendDetector(top_n=15)

    ranked = detector.rank_hashtags(df)
    print("── Top 10 hashtags by viral composite ──")
    print(ranked.head(10)[["total_mentions", "peak_mentions",
                            "viral_composite", "viral_rank"]].to_string())

    detector.weekly_heatmap(df)
    detector.correlation_matrix(df)
