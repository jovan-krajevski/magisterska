from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.structural import UnobservedComponents
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)

print("START")
dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])
print("DATA READY")


def plot(train, test, pred, prefix, point, ticker_name):
    train_dates = train["ds"].dt.date
    train_values = train["y"]
    test_dates = test["ds"].dt.date
    test_values = test["y"]
    pred_dates = pred.index
    pred_values = pred.values
    plt.figure(figsize=(12, 6))
    plt.plot(train_dates, train_values, label="Тренирачко множество", color="blue")
    plt.plot(test_dates, test_values, label="Вистински вредности", color="green")
    plt.plot(pred_dates, pred_values, label="Прогнозирани вредности", color="orange")
    plt.grid()
    plt.legend()

    # save plot to file
    plt.savefig(f"prediction_plots/{prefix}/{point}_{ticker_name}.png")


# Iterate over all files in predictions folder
predictions_folder = Path("./predictions")
for file in predictions_folder.iterdir():
    if file.is_file():
        point, ticker_name = file.stem.split("_")
        # Read the CSV file
        vangja_preds = pd.read_csv(file, index_col=0)

        gspc_ticker = [
            ticker for ticker in gspc_tickers if ticker["series"].iloc[0] == ticker_name
        ][0]
        train_df_tickers, test_df_tickers, scales_tickers = (
            generate_train_test_df_around_point(
                window=91,
                horizon=365,
                dfs=[gspc_ticker],
                for_prophet=False,
                point=point,
            )
        )
        y_train = train_df_tickers.set_index("ds")["y"]
        fh = ForecastingHorizon(
            test_df_tickers.set_index("ds").index, is_relative=False
        )
        es_forecaster = ExponentialSmoothing(seasonal="additive", sp=7)
        es_forecaster.fit(y=y_train)
        y_pred_es = es_forecaster.predict(fh=fh)
        y_pred_es = y_pred_es.rename("y")

        plot(
            train_df_tickers,
            test_df_tickers,
            y_pred_es,
            "es",
            point,
            ticker_name,
        )

        plot(
            train_df_tickers,
            test_df_tickers,
            pd.DataFrame(
                vangja_preds[vangja_preds.columns[0]].to_list(), index=y_pred_es.index
            )[0],
            "vangja",
            point,
            ticker_name,
        )
