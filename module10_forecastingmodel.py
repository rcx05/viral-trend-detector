import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error

from module1_data_loader import SocialMediaLoader

warnings.filterwarnings("ignore")

PLOT_DIR = Path("plots")
RESULT_DIR = Path("results")

PLOT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


# -----------------------------------------
# Evaluation metrics
# -----------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[valid] - predicted[valid])
                / actual[valid]
            )
        ) * 100
        if valid.any()
        else np.nan
    )

    return {
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted),
        "MAPE (%)": mape
    }


# -----------------------------------------
# Forecasting models
# -----------------------------------------

def run_forecasting(hashtag=None):

    loader = SocialMediaLoader()
    loader.load()

    if hashtag:
        target = hashtag.lower().lstrip("#")
    else:
        target = loader.top_hashtags(1)[0]

    series = loader.get_hashtag_series(target)

    if series.empty:
        raise ValueError(f"No data found for #{target}")

    # Ensure a continuous daily time series
    series.index = pd.to_datetime(series.index)
    series = series.groupby(level=0).sum()
    series = series.resample("D").sum().asfreq("D", fill_value=0)
    series = series.astype(float)

    if len(series) < 100:
        raise ValueError(
            "At least 100 daily observations are recommended."
        )

    # Chronological split: no random shuffling
    split = int(len(series) * 0.80)

    train = series.iloc[:split]
    test = series.iloc[split:]

    forecasts = {}
    rows = []

    print(f"\nForecasting hashtag: #{target}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")

    def evaluate(name, prediction):
        prediction = np.asarray(prediction, dtype=float).reshape(-1)

        if len(prediction) != len(test):
            raise ValueError("Forecast length does not match test data.")

        # Mention counts cannot be negative
        prediction = np.maximum(prediction, 0)

        forecasts[name] = prediction

        metrics = calculate_metrics(test.values, prediction)

        rows.append({
            "Model": name,
            **metrics
        })

        print(f"{name}: RMSE={metrics['RMSE']:.2f}")

    # AR(p)
    try:
        model = ARIMA(train, order=(2, 0, 0)).fit()
        evaluate("AR(2)", model.forecast(len(test)))
    except Exception as e:
        print("AR failed:", e)

    # MA(q)
    try:
        model = ARIMA(train, order=(0, 0, 2)).fit()
        evaluate("MA(2)", model.forecast(len(test)))
    except Exception as e:
        print("MA failed:", e)

    # ARMA(p,q)
    try:
        model = ARIMA(train, order=(2, 0, 2)).fit()
        evaluate("ARMA(2,2)", model.forecast(len(test)))
    except Exception as e:
        print("ARMA failed:", e)

    # ARIMA(p,d,q)
    try:
        model = ARIMA(train, order=(2, 1, 2)).fit()
        evaluate("ARIMA(2,1,2)", model.forecast(len(test)))
    except Exception as e:
        print("ARIMA failed:", e)

    # SARIMA(p,d,q)(P,D,Q,s)
    try:
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        evaluate("SARIMA", model.forecast(len(test)))
    except Exception as e:
        print("SARIMA failed:", e)

    # Holt-Winters
    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated"
        ).fit()

        evaluate("Holt-Winters", model.forecast(len(test)))
    except Exception as e:
        print("Holt-Winters failed:", e)

    # SARIMAX with another hashtag as an external variable
    try:
        other_tags = [
            tag for tag in loader.top_hashtags(5)
            if tag.lower() != target.lower()
        ]

        if not other_tags:
            raise ValueError("No second hashtag available for exogenous data.")

        exog_series = loader.get_hashtag_series(other_tags[0])
        exog_series.index = pd.to_datetime(exog_series.index)

        exog = (
            exog_series.groupby(level=0).sum()
            .reindex(series.index)
            .fillna(0)
        )

        model = SARIMAX(
            train,
            exog=exog.iloc[:split],
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        prediction = model.forecast(
            len(test),
            exog=exog.iloc[split:]
        )

        evaluate("SARIMAX", prediction)

    except Exception as e:
        print("SARIMAX failed:", e)

    # VAR: multiple hashtags forecast together
    try:
        tags = [target] + [
            tag for tag in loader.top_hashtags(5)
            if tag.lower() != target.lower()
        ][:2]

        multivariate = pd.DataFrame({
            tag: (
                loader.get_hashtag_series(tag)
                .groupby(level=0).sum()
                .reindex(series.index)
                .fillna(0)
            )
            for tag in tags
        })

        if len(tags) >= 2:
            model = VAR(multivariate.iloc[:split]).fit(
                maxlags=3,
                ic="aic"
            )

            prediction = model.forecast(
                multivariate.iloc[:split].values[-model.k_ar:],
                steps=len(test)
            )

            # Evaluate the target hashtag, not the other columns
            evaluate("VAR", prediction[:, 0])

    except Exception as e:
        print("VAR failed:", e)

    # Optional: AutoTS
    try:
        from autots import AutoTS

        model = AutoTS(
            forecast_length=len(test),
            frequency="infer",
            ensemble="simple",
            max_generations=3,
            num_validations=2,
            verbose=0
        )

        model = model.fit(
            train.to_frame(name=target)
        )

        prediction = model.predict().forecast[target]

        evaluate("AutoTS", prediction.iloc[:len(test)])

    except ImportError:
        print("AutoTS skipped. Install with: pip install autots")
    except Exception as e:
        print("AutoTS failed:", e)

    # Optional: Prophet
    try:
        from prophet import Prophet

        prophet_train = pd.DataFrame({
            "ds": train.index,
            "y": train.values
        })

        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False
        )

        model.fit(prophet_train)

        future = pd.DataFrame({"ds": test.index})
        prediction = model.predict(future)["yhat"]

        evaluate("Prophet", prediction)

    except ImportError:
        print("Prophet skipped. Install with: pip install prophet")
    except Exception as e:
        print("Prophet failed:", e)

    # -------------------------------------
    # Compare results
    # -------------------------------------

    if not rows:
        raise RuntimeError("All forecasting models failed.")

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("RMSE")

    comparison.to_csv(
        RESULT_DIR / f"{target}_model_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print(comparison.round(3).to_string(index=False))

    # -------------------------------------
    # Plot forecasts
    # -------------------------------------

    plt.figure(figsize=(15, 6))

    plt.plot(
        test.index,
        test.values,
        label="Actual",
        linewidth=2
    )

    for name, prediction in forecasts.items():
        plt.plot(
            test.index,
            prediction,
            label=name,
            alpha=0.8
        )

    plt.title(f"Forecast Comparison: #{target}")
    plt.xlabel("Date")
    plt.ylabel("Daily Mentions")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"forecast_comparison_{target}.png",
        dpi=150
    )

    plt.close()

    # -------------------------------------
    # Plot metric comparison
    # -------------------------------------

    comparison.set_index("Model")[["RMSE", "MAE"]].plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(f"Forecast Error Comparison: #{target}")
    plt.ylabel("Error")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"forecast_metrics_{target}.png",
        dpi=150
    )

    plt.close()

    best_model = comparison.iloc[0]["Model"]

    print(f"\nBest model by RMSE: {best_model}")
    print(f"Results saved in: {RESULT_DIR}")
    print(f"Plots saved in: {PLOT_DIR}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hashtag",
        type=str,
        default=None
    )

    args = parser.parse_args()

    run_forecasting(args.hashtag)