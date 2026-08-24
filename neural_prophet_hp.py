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

points = "2015-01-01"
train_df_smp, _, _ = generate_train_test_df_around_point(
    window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
)
train_df_smp.drop("series", axis=1, inplace=True)

N_LAGS = list(range(0, 21))
N_FORECASTS = 10
for n_lags in N_LAGS:
    forecaster = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_changepoints=25,
        seasonality_mode="multiplicative",
        n_lags=n_lags,
        n_forecasts=N_FORECASTS,
        # epochs=15,
    )
    df_train, df_test = forecaster.split_df(df=train_df_smp, freq="D", valid_p=0.2)

    metrics_train = forecaster.fit(df_train, freq="D", progress="bar")
    metrics_test = forecaster.test(df=df_test)
    # open a file to append results
    with open("./out/neural_prophet_hp.txt", "a") as f:
        f.write(
            f"""
n_lags: {n_lags}
train_loss: {metrics_train.iloc[-1]["Loss"]}
train_mae: {metrics_train.iloc[-1]["MAE"]}
test_loss: {metrics_test.iloc[-1]["Loss_test"]}
test_mae: {metrics_test.iloc[-1]["MAE_val"]}
        """
        )

    print(
        f"""
n_lags: {n_lags}
train_loss: {metrics_train.iloc[-1]["Loss"]}
train_mae: {metrics_train.iloc[-1]["MAE"]}
test_loss: {metrics_test.iloc[-1]["Loss_test"]}
test_mae: {metrics_test.iloc[-1]["MAE_val"]}
        """
    )
