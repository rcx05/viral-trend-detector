"""
main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAIN PIPELINE — Viral Trend Detector
  Runs all 9 modules end-to-end then launches the dashboard.

  Usage:
    python main.py                   # full pipeline + dashboard
    python main.py --no-dashboard    # analysis only (saves plots)
    python main.py --hashtag chatgpt # analyse specific hashtag
    python main.py --top 5           # analyse top-5 hashtags
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ── colour helpers ────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    MAGENTA= "\033[95m"

def banner():
    b = f"""
{C.CYAN}{C.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    🔥  VIRAL TREND DETECTOR  🔥                               ║
║    Social Media Time-Series Analysis & Spike Detection        ║
║                                                               ║
║    MODULE 1  │  Data Loading & Time Series                    ║
║    MODULE 2  │  Trend Analysis (Rolling + STL)                ║
║    MODULE 3  │  Anomaly / Spike Detection                     ║
║    MODULE 4  │  Hashtag-Level Trend Ranking                   ║
║    MODULE 5  │  Viral Score System (0-100)                    ║
║    MODULE 7  │  Event Classification                          ║
║    MODULE 8  │  Early Viral Detection (EWS)                   ║
║    MODULE 9  │  Interactive Dashboard                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
{C.RESET}"""
    print(b)


def section(title: str):
    print(f"\n{C.YELLOW}{C.BOLD}{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}{C.RESET}\n")


def ok(msg: str):
    print(f"  {C.GREEN}✔{C.RESET}  {msg}")


def info(msg: str):
    print(f"  {C.CYAN}ℹ{C.RESET}  {msg}")


def warn(msg: str):
    print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")


# ════════════════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(hashtags_to_analyse: list[str], daily_df, loader):
    from module2_trend_analysis    import TrendAnalyzer
    from module3_anomaly_detection import AnomalyDetector
    from module4_hashtag_trends    import HashtagTrendDetector
    from module5_viral_score       import ViralScorer
    from module7_event_detection   import EventDetector
    from module8_early_detection   import EarlyViralDetector

    all_summaries = []
    spike_reports = []

    trend_ana  = TrendAnalyzer()
    anom_det   = AnomalyDetector()
    ht_det     = HashtagTrendDetector(top_n=20)
    scorer     = ViralScorer()
    event_det  = EventDetector()
    early_det  = EarlyViralDetector()

    # ── MODULE 4: hashtag rankings (whole dataset) ─────────────────────────
    section("MODULE 4 — Hashtag Rankings")
    ranked = ht_det.rank_hashtags(daily_df)
    print(f"  Top 10 hashtags by viral composite:\n")
    print(ranked.head(10)[["total_mentions", "peak_mentions",
                             "viral_composite", "viral_rank"]]
          .to_string(index=True))
    ht_det.weekly_heatmap(daily_df)
    ht_det.correlation_matrix(daily_df)
    ok("Heatmap + correlation matrix saved")

    # ── per-hashtag analysis ───────────────────────────────────────────────
    for hashtag in hashtags_to_analyse:
        section(f"ANALYSING  #{hashtag.upper()}")
        series = loader.get_hashtag_series(hashtag)

        if series.empty or series.sum() == 0:
            warn(f"No data for #{hashtag} — skipping.")
            continue

        info(f"  {len(series)} days  |  "
             f"mean={series.mean():.0f}  max={series.max():.0f}")

        # MODULE 2 – trend
        print(f"\n  {C.BLUE}[Module 2]{C.RESET} Trend analysis...")
        t2 = trend_ana.analyze(series, hashtag)
        trend_ana.plot_trend(t2, hashtag)
        ok("Trend plot saved")

        # MODULE 3 – anomaly detection
        print(f"\n  {C.BLUE}[Module 3]{C.RESET} Anomaly/spike detection...")
        t3 = anom_det.detect(series, hashtag)
        anom_det.plot_anomalies(t3, hashtag)
        spikes = anom_det.get_spike_events(t3)
        n_spikes = len(spikes)
        ok(f"Detected {n_spikes} spike event(s)")
        if n_spikes > 0:
            s = spikes.iloc[0]
            info(f"  First spike → peak {s.get('peak_magnitude', '?'):.0f} mentions  "
                 f"| growth ×{s.get('growth_factor', 0):.1f}  "
                 f"| duration {s.get('duration_days', '?'):.0f} days")
            spike_reports.append({
                "hashtag":       hashtag,
                "spike_count":   n_spikes,
                "peak_mentions": spikes["peak_magnitude"].max(),
                "max_growth":    spikes["growth_factor"].max(),
                "max_duration":  spikes["duration_days"].max(),
            })

        # MODULE 5 – viral score
        print(f"\n  {C.BLUE}[Module 5]{C.RESET} Viral scoring...")
        t5 = scorer.score(series, hashtag)
        summ = scorer.summary(t5)
        scorer.plot_viral_score(t5, hashtag)
        ok(f"Peak viral score: {summ['peak_viral_score']} ({summ['peak_tier']}) "
           f"on {summ['peak_viral_date']}")
        all_summaries.append(summ)

        # MODULE 7 – event classification
        print(f"\n  {C.BLUE}[Module 7]{C.RESET} Event detection...")
        t7 = event_det.classify_events(t3, t5)
        event_det.event_timeline_plot(t7, hashtag)
        counts = t7["event_type"].value_counts()
        ok("Event types: " + ", ".join(f"{k}:{v}" for k, v in counts.items()))

        # MODULE 8 – early detection
        print(f"\n  {C.BLUE}[Module 8]{C.RESET} Early warning system...")
        t8  = early_det.compute_ews(series, hashtag)
        acc = early_det.prediction_accuracy(t8)
        early_det.plot_ews(t8, hashtag)
        ok(f"EWS F1={acc['f1_score']:.3f}  "
           f"Precision={acc['precision']:.3f}  "
           f"Recall={acc['recall']:.3f}")

    # ── final summary ──────────────────────────────────────────────────────
    section("PIPELINE COMPLETE — SUMMARY")
    if all_summaries:
        summ_df = __import__("pandas").DataFrame(all_summaries)
        print(summ_df.to_string(index=False))

    if spike_reports:
        print(f"\n  {C.RED}🚨 Spike Summary:{C.RESET}")
        sr_df = __import__("pandas").DataFrame(spike_reports)
        print(sr_df.to_string(index=False))

    plots_dir = Path("plots")
    plot_files = list(plots_dir.glob("*.png")) if plots_dir.exists() else []
    ok(f"{len(plot_files)} plots saved in  plots/")
    return all_summaries, spike_reports


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🔥 Viral Trend Detector – Social Media Analysis Pipeline"
    )
    parser.add_argument("--hashtag",       type=str,  default=None,
                        help="Analyse a specific hashtag (e.g. chatgpt)")
    parser.add_argument("--top",           type=int,  default=3,
                        help="Number of top hashtags to analyse (default: 3)")
    parser.add_argument("--no-dashboard",  action="store_true",
                        help="Skip launching the interactive dashboard")
    parser.add_argument("--generate-data", action="store_true",
                        help="Force regenerate / re-download data")
    args = parser.parse_args()

    banner()

    # ── ensure data exists ─────────────────────────────────────────────────
    import os
    data_csv = os.path.join("data", "social_media_trends.csv")
    if not os.path.exists(data_csv) or args.generate_data:
        section("DATA SETUP")
        import generate_data as gd
        if not gd.try_kaggle():
            gd.generate_dataset()
    else:
        info(f"Using existing data: {data_csv}")

    # ── MODULE 1 – load data ───────────────────────────────────────────────
    section("MODULE 1 — Loading Social Media Time Series")
    from module1_data_loader import SocialMediaLoader
    loader   = SocialMediaLoader()
    daily_df = loader.load()

    # ── select hashtags to analyse ─────────────────────────────────────────
    if args.hashtag:
        targets = [args.hashtag.lower().lstrip("#")]
    else:
        targets = loader.top_hashtags(args.top)
        info(f"Analysing top-{args.top} hashtags: {targets}")

    # ── run pipeline ───────────────────────────────────────────────────────
    Path("plots").mkdir(exist_ok=True)
    t_start = time.time()
    run_pipeline(targets, daily_df, loader)
    elapsed = time.time() - t_start
    ok(f"Full pipeline completed in {elapsed:.1f}s")

    # ── MODULE 9 – dashboard ───────────────────────────────────────────────
    if not args.no_dashboard:
        section("MODULE 9 — Launching Interactive Dashboard")
        try:
            from module9_dashboard import DataCache, build_app
            from config import DASH_HOST, DASH_PORT
            cache = DataCache()
            app   = build_app(cache)
            print(f"\n  {C.GREEN}🌐 Open your browser →  "
                  f"http://{DASH_HOST}:{DASH_PORT}/{C.RESET}\n")
            app.run(host=DASH_HOST, port=DASH_PORT, debug=False)
        except ImportError as exc:
            warn(f"Dashboard unavailable: {exc}")
            warn("Install:  pip install dash dash-bootstrap-components")
    else:
        info("Dashboard skipped (--no-dashboard flag).")


if __name__ == "__main__":
    main()
