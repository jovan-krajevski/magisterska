import os

os.environ["TQDM_DISABLE"] = "1"

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)

set_log_level("ERROR")  # Suppress NeuralProphet logs

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Download and process data
dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])


def metrics(y_true, yhat, label="y"):
    """Calculate forecasting metrics."""
    y = y_true["y"]
    return pd.DataFrame(
        {
            "mse": {f"{label}": mean_squared_error(y, yhat)},
            "rmse": {f"{label}": root_mean_squared_error(y, yhat)},
            "mae": {f"{label}": mean_absolute_error(y, yhat)},
            "mape": {f"{label}": mean_absolute_percentage_error(y, yhat)},
        }
    )


# Directory to save results
csv_path = Path("./out/neural_prophet_ar")

# Benchmarking NeuralProphet
date_range = pd.date_range("2015-01-01", "2017-01-01")
all_averages = []

for point in date_range:
    points = f"{point.year}-{'0' if point.month < 10 else ''}{point.month}-{'0' if point.day < 10 else ''}{point.day}"

    # Check if already processed
    metrics_file_path = csv_path / f"{points}.csv"
    if metrics_file_path.exists():
        # Read existing metrics and calculate mean MAPE
        existing_metrics = pd.read_csv(metrics_file_path, index_col=0)
        mean_mape = existing_metrics["mape"].mean()
        all_averages.append(mean_mape)

for point in date_range:
    points = f"{point.year}-{'0' if point.month < 10 else ''}{point.month}-{'0' if point.day < 10 else ''}{point.day}"

    # Check if already processed
    metrics_file_path = csv_path / f"{points}.csv"
    if metrics_file_path.exists():
        # Read existing metrics and calculate mean MAPE
        existing_metrics = pd.read_csv(metrics_file_path, index_col=0)
        mean_mape = existing_metrics["mape"].mean()
        print(f"{points} (existing): {mean_mape}")
        print(f"Avg: {sum(all_averages) / len(all_averages)}")
        continue

    model_metrics = []
    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
    )
    train_df_tickers, test_df_tickers, scales_tickers = (
        generate_train_test_df_around_point(
            window=91, horizon=365, dfs=gspc_tickers, point=points
        )
    )

    train_df_tickers = train_df_tickers.rename(columns={"series": "ID"})
    train_df_smp = train_df_smp.rename(columns={"series": "ID"})

    min_smp_y = train_df_smp["y"].iloc[-91:].min()
    max_smp_y = train_df_smp["y"].iloc[-91:].max()

    min_y = train_df_tickers.groupby("ID")["y"].transform("min")
    max_y = train_df_tickers.groupby("ID")["y"].transform("max")
    denom = max_y - min_y
    train_df_tickers["y"] = np.where(
        denom == 0,
        train_df_tickers["y"],
        (train_df_tickers["y"] - min_y) / denom * (max_smp_y - min_smp_y) + min_smp_y,
    )

    train_df = pd.concat([train_df_smp, train_df_tickers], ignore_index=True)
    test_df = pd.concat([test_df_smp, test_df_tickers], ignore_index=True)

    N_FORECASTS = 10
    HORIZON = 365
    forecaster = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_changepoints=25,
        seasonality_mode="multiplicative",
        n_lags=10,
        n_forecasts=N_FORECASTS,
        # epochs=2,
        # accelerator="auto", # Enable automatic accelerator selection (GPU if available)
    )
    forecaster.fit(train_df, freq="D", progress=None)  # Disable progress bar

    history = train_df.copy()
    final_forecast = []
    for i in range(0, HORIZON, N_FORECASTS):
        print(f"Processing {points} - Day {i + 1}/{HORIZON}", end="\r")
        future = forecaster.make_future_dataframe(history)
        forecast = forecaster.predict(
            future, decompose=False
        )  # Disable decomposition progress
        forecast["yhat"] = forecast.filter(like="yhat").bfill(axis=1).iloc[:, 0]
        forecast = (
            forecast.dropna(subset=["yhat"])[["ds", "ID", "yhat"]]
            .rename(columns={"yhat": "y"})
            .reset_index(drop=True)
        )
        # set negative forecasts to 0
        forecast["y"] = forecast["y"].apply(lambda x: max(x, 0))
        final_forecast.append(forecast)
        history = pd.concat([history, forecast], ignore_index=True)

    final_forecast = pd.concat(final_forecast, ignore_index=True)
    for ticker in test_df["series"].unique():
        true_y = test_df[test_df["series"] == ticker].iloc[:HORIZON]
        final_ds = true_y["ds"].iloc[-1]
        y = final_forecast[final_forecast["ID"] == ticker]
        y = y[y["ds"] <= final_ds]["y"]
        if ticker != "^GSPC":
            ticker_min_y = min_y[train_df_tickers["ID"] == ticker].iloc[0]
            ticker_max_y = max_y[train_df_tickers["ID"] == ticker].iloc[0]
            if ticker_max_y != ticker_min_y:
                y = (y - min_smp_y) / (max_smp_y - min_smp_y) * (
                    ticker_max_y - ticker_min_y
                ) + ticker_min_y

        model_metrics.append(metrics(true_y, y, label=ticker))

    # Save results
    final_metrics = pd.concat(model_metrics)
    final_metrics = final_metrics.sort_index()
    save_path = csv_path / f"{points}.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    final_metrics.to_csv(save_path)
    mean_mape = final_metrics["mape"].mean()
    all_averages.append(mean_mape)
    print(f"{points}: {mean_mape}")
    print(f"Avg: {sum(all_averages) / len(all_averages)}")
