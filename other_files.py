import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
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

import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="sktime")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels")
# sys.stderr = open("/dev/null", "w")


dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])


def metrics(y_true, yhat, label="y"):
    y = y_true["y"]
    return pd.DataFrame(
        {
            "mse": {f"{label}": mean_squared_error(y, yhat)},
            "rmse": {f"{label}": root_mean_squared_error(y, yhat)},
            "mae": {f"{label}": mean_absolute_error(y, yhat)},
            "mape": {f"{label}": mean_absolute_percentage_error(y, yhat)},
        }
    )


csv_path = Path("./out")

for point in pd.date_range(f"2016-01-01", f"2017-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    model_metrics = {"arima": [], "es": [], "uc": []}
    arima_model_metrics = []
    es_model_metrics = []
    for gspc_ticker in tqdm(gspc_tickers):
        check = generate_train_test_df_around_point(
            window=91,
            horizon=365,
            dfs=[gspc_ticker],
            for_prophet=False,
            point=point,
        )
        if check is None:
            continue

        train_df_tickers, test_df_tickers, scales_tickers = check
        y_train = train_df_tickers.set_index("ds")["y"]
        fh = ForecastingHorizon(
            test_df_tickers.set_index("ds").index, is_relative=False
        )

        arima_forecaster = AutoARIMA(suppress_warnings=True)
        arima_forecaster.fit(y=y_train)
        y_pred_arima = arima_forecaster.predict(fh=fh)
        model_metrics["arima"].append(
            metrics(
                test_df_tickers, y_pred_arima, label=train_df_tickers["series"].iloc[0]
            )
        )

        fh = ForecastingHorizon(
            test_df_tickers.set_index("ds").index, is_relative=False
        )

        es_forecaster = ExponentialSmoothing(seasonal="additive", sp=7)
        es_forecaster.fit(y=y_train)
        y_pred_es = es_forecaster.predict(fh=fh)
        model_metrics["es"].append(
            metrics(
                test_df_tickers, y_pred_es, label=train_df_tickers["series"].iloc[0]
            )
        )

        fh = ForecastingHorizon(
            test_df_tickers.set_index("ds").index, is_relative=False
        )

        uc_forecaster = UnobservedComponents(
            level="local level",
            freq_seasonal=[
                {"period": 7, "harmonics": 6},
            ],
        )
        uc_forecaster.fit(y=y_train)
        y_pred_uc = uc_forecaster.predict(fh=fh)
        model_metrics["uc"].append(
            metrics(
                test_df_tickers, y_pred_uc, label=train_df_tickers["series"].iloc[0]
            )
        )

    for key, one_metrics in model_metrics.items():
        final_metrics = pd.concat(one_metrics)
        final_metrics = final_metrics.sort_index()
        save_path = csv_path / f"{key}" / f"{points}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_metrics.to_csv(save_path)
        print(f"{key}: {final_metrics['mape'].mean()}")
