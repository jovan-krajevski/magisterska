import os

os.environ["TQDM_DISABLE"] = "1"

import warnings
from itertools import product
from pathlib import Path

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


points = "2015-01-01"

N_LAGS = [1, 3, 5, 10, 20]
LOSS = ["SmoothL1Loss", "MSE", "MAE"]
LR = [1e-3, 1e-2, 1e-1, 1, 10]
TREND_REG = [0.1, 10]
SEASONALITY_REG = [0.5, 10]
AR_REG = [0.1, 10]

N_FORECASTS = 10
HORIZON = 365

hparams_metrics_file = Path("./out/neural_prophet_full_hp.csv")
if hparams_metrics_file.exists():
    hparams_metrics_file.unlink()

with open(hparams_metrics_file, "w") as f:
    f.write("n_lags,loss,lr,trend_reg,seasonality_reg,ar_reg,mse,rmse,mae,mape\n")

params = list(product(N_LAGS, LOSS, LR, TREND_REG, SEASONALITY_REG, AR_REG))
for n_lags, loss, lr, trend_reg, seasonality_reg, ar_reg in params:
    forecaster = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_changepoints=25,
        seasonality_mode="multiplicative",
        n_forecasts=N_FORECASTS,
        trainer_config={"enable_checkpointing": False, "logger": False},
        # params
        n_lags=n_lags,
        loss_func=loss,
        learning_rate=lr,
        trend_reg=trend_reg,
        seasonality_reg=seasonality_reg,
        ar_reg=ar_reg,
        # epochs=2,
        # accelerator="auto", # Enable automatic accelerator selection (GPU if available)
    )

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

    # min_smp_y = train_df_smp["y"].iloc[-91:].min()
    # max_smp_y = train_df_smp["y"].iloc[-91:].max()

    # min_y = train_df_tickers.groupby("ID")["y"].transform("min")
    # max_y = train_df_tickers.groupby("ID")["y"].transform("max")
    # denom = max_y - min_y
    # train_df_tickers["y"] = np.where(
    #     denom == 0,
    #     train_df_tickers["y"],
    #     (train_df_tickers["y"] - min_y) / denom * (max_smp_y - min_smp_y) + min_smp_y,
    # )

    train_df = pd.concat([train_df_smp, train_df_tickers], ignore_index=True)
    test_df = pd.concat([test_df_smp, test_df_tickers], ignore_index=True)

    forecaster.fit(
        train_df, freq="D", progress=None, minimal=True
    )  # Disable progress bar

    history = train_df.copy()
    final_forecast = []
    for i in range(0, HORIZON, N_FORECASTS):
        print(f"Processing {points} - Day {i + 1}/{HORIZON}", end="\r")
        future = forecaster.make_future_dataframe(history)
        forecast = forecaster.predict(
            future, decompose=True
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
        # if ticker != "^GSPC":
        #     ticker_min_y = min_y[train_df_tickers["ID"] == ticker].iloc[0]
        #     ticker_max_y = max_y[train_df_tickers["ID"] == ticker].iloc[0]
        #     if ticker_max_y != ticker_min_y:
        #         y = (y - min_smp_y) / (max_smp_y - min_smp_y) * (
        #             ticker_max_y - ticker_min_y
        #         ) + ticker_min_y

        model_metrics.append(metrics(true_y, y, label=ticker))

    # Save results
    final_metrics = pd.concat(model_metrics)
    final_metrics = final_metrics.sort_index()
    print(final_metrics)

    # append to csv
    with open(hparams_metrics_file, "a") as f:
        f.write(
            f"{n_lags},{loss},{lr},{trend_reg},{seasonality_reg},{ar_reg},"
            f"{final_metrics['mse'].mean()},{final_metrics['rmse'].mean()},"
            f"{final_metrics['mae'].mean()},{final_metrics['mape'].mean()}\n"
        )
