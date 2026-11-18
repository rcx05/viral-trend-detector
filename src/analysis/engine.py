"""
Viral Trend Detector — Core Statistical Analysis Engine
Techniques:
  1. Rolling Average (MA)
  2. Z-Score based anomaly detection
  3. Change-Point Detection (CUSUM binary segmentation)
  4. Seasonal Decomposition (STL via statsmodels)
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from typing import Dict, List, Any

from .changepoint import detect_changepoints


# ─────────────────────────── Rolling Statistics ────────────────────────────

def compute_rolling_stats(series: pd.Series, window: int = 7) -> pd.DataFrame:
    """
    Compute rolling mean, std, and momentum for a time series.

    Args:
        series: Pandas Series with DatetimeIndex
        window: Rolling window size (default = 7 days)

    Returns:
        DataFrame with original values + rolling stats
    """
    df = series.to_frame(name="value")
    df["rolling_mean"]     = df["value"].rolling(window=window, min_periods=1).mean()
    df["rolling_std"]      = df["value"].rolling(window=window, min_periods=1).std().fillna(0)
    df["rolling_upper"]    = df["rolling_mean"] + 2 * df["rolling_std"]
    df["rolling_lower"]    = (df["rolling_mean"] - 2 * df["rolling_std"]).clip(lower=0)
    # Momentum: % change in rolling mean
    df["momentum"]         = df["rolling_mean"].pct_change(periods=1).fillna(0) * 100
    return df


# ──────────────────────────── Z-Score Detection ───────────────────────────

def zscore_anomalies(series: pd.Series, threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag anomalies using a rolling Z-score.

    Args:
        series: Pandas Series with DatetimeIndex
        threshold: Z-score threshold to flag a spike (default 2.5)

    Returns:
        DataFrame with 'zscore' and boolean 'is_anomaly' columns
    """
    window = max(7, len(series) // 5)
    roll_mean = series.rolling(window=window, min_periods=3).mean()
    roll_std  = series.rolling(window=window, min_periods=3).std().replace(0, np.nan)

    zscore = ((series - roll_mean) / roll_std).fillna(0)
    df = pd.DataFrame({
        "value":      series,
        "zscore":     zscore,
        "is_anomaly": zscore.abs() > threshold,
        "is_spike":   zscore > threshold,       # positive direction only
        "is_drop":    zscore < -threshold,
    })
    return df


# ──────────────────────────── Viral Score ─────────────────────────────────

def compute_viral_score(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Proprietary Viral Score (0–100) combining:
      - Momentum (rate of change)
      - Z-score magnitude
      - Acceleration (2nd derivative)

    Returns a Series of viral scores.
    """
    if len(series) < 3:
        return pd.Series([0.0] * len(series), index=series.index)

    rolling = compute_rolling_stats(series, window)
    z_df    = zscore_anomalies(series)

    momentum     = rolling["momentum"].fillna(0)
    zscore_mag   = z_df["zscore"].abs().fillna(0)
    accel        = series.diff().diff().fillna(0)
    accel_norm   = accel / (accel.abs().max() + 1e-9) * 100

    # Weighted composite score (0–100)
    raw_score = (
        0.40 * momentum.clip(-100, 100) / 100 * 100  +
        0.40 * zscore_mag.clip(0, 5) / 5 * 100       +
        0.20 * accel_norm.clip(-100, 100)
    )
    return raw_score.clip(0, 100).rename("viral_score")


# ──────────────────────────── Seasonal Decomposition ─────────────────────

def seasonal_decompose(series: pd.Series, period: int = 7) -> Dict[str, pd.Series]:
    """
    STL decomposition into Trend + Seasonal + Residual.

    Args:
        series: Pandas Series (needs at least 2 * period observations)
        period: Seasonal period (default 7 = weekly)

    Returns:
        Dict with 'trend', 'seasonal', 'residual' Series
    """
    if len(series) < 2 * period:
        # Not enough data — return raw series as trend
        return {
            "trend":    series.copy(),
            "seasonal": pd.Series(0, index=series.index),
            "residual": pd.Series(0, index=series.index),
        }
    try:
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        return {
            "trend":    result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        }
    except Exception:
        return {
            "trend":    series.copy(),
            "seasonal": pd.Series(0, index=series.index),
            "residual": pd.Series(0, index=series.index),
        }


# ──────────────────────────── Change-Point Detection ─────────────────────

def find_changepoints(series: pd.Series, penalty: float = 3.0) -> List[pd.Timestamp]:
    """
    Returns timestamps where a significant structural break was detected.
    """
    indices = detect_changepoints(series.tolist(), penalty=penalty)
    return [series.index[i] for i in indices if i < len(series)]


# ──────────────────────────── Full Analysis Pipeline ─────────────────────

def full_analysis(
    series: pd.Series,
    topic: str,
    window: int = 7,
    zscore_thresh: float = 2.5,
    seasonal_period: int = 7,
    cp_penalty: float = 3.0,
) -> Dict[str, Any]:
    """
    Run the complete viral trend analysis pipeline.

    Returns a dict with all analysis results, ready for the dashboard.
    """
    series = series.dropna().sort_index()
    if series.empty or len(series) < 2:
        return {"error": "Not enough data to analyse."}

    rolling     = compute_rolling_stats(series, window)
    anomaly_df  = zscore_anomalies(series, zscore_thresh)
    viral_score = compute_viral_score(series, window)
    decomp      = seasonal_decompose(series, seasonal_period)
    changepoints= find_changepoints(series, cp_penalty)

    latest_z    = float(anomaly_df["zscore"].iloc[-1])
    latest_vs   = float(viral_score.iloc[-1]) if len(viral_score) > 0 else 0.0
    is_viral    = latest_vs > 60 or latest_z > zscore_thresh

    # Classify trend status
    if latest_vs >= 75:
        status = "🔥 VIRAL"
    elif latest_vs >= 50:
        status = "📈 RISING"
    elif latest_vs >= 25:
        status = "📊 ACTIVE"
    else:
        status = "😴 QUIET"

    return {
        "topic":        topic,
        "series":       series,
        "rolling":      rolling,
        "anomaly":      anomaly_df,
        "viral_score":  viral_score,
        "decomp":       decomp,
        "changepoints": changepoints,
        "is_viral":     is_viral,
        "status":       status,
        "latest_zscore": latest_z,
        "latest_viral_score": latest_vs,
        "spike_dates":  anomaly_df[anomaly_df["is_spike"]].index.tolist(),
        "peak_value":   float(series.max()),
        "peak_date":    series.idxmax(),
        "total_points": len(series),
    }
