from pathlib import Path

import numpy as np
import pandas as pd
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.rnn._rnn import RecurrentNetwork
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)


def convert_to_dl_format(series, train_window_length, test_window_length):
    X = []
    y_out = []
    for i in range(len(series) - train_window_length - test_window_length + 1):
        X.append(series.iloc[i : i + train_window_length].values)
        y_out.append(
            series.iloc[
                i + train_window_length : i + train_window_length + test_window_length
            ]
        )
    return (np.array(X).reshape(-1, train_window_length, 1), np.array(y_out))


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

train_window_length = 60
test_window_length = 1

max_encoder_length = 60
max_prediction_length = 10

for point in pd.date_range(f"2016-01-01", f"2017-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    model_metrics = {"lstm": []}
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

        y_train = pd.DataFrame(
            {
                "time_idx": np.arange(train_df_tickers.shape[0]),
                "date": train_df_tickers["ds"],
                "value": train_df_tickers["y"],
                "series_id": train_df_tickers["series"],
            }
        )

        y_test = pd.DataFrame(
            {
                "time_idx": np.arange(test_df_tickers.shape[0]),
                "date": test_df_tickers["ds"],
                "value": test_df_tickers["y"],
                "series_id": test_df_tickers["series"].iloc[0],
            }
        )

        # forecast_horizon = 365
        # max_time = df["time_idx"].max()

        # future_df = pd.DataFrame(
        #     {
        #         "time_idx": np.arange(max_time + 1, max_time + forecast_horizon + 1),
        #         "date": pd.date_range(
        #             df["date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon
        #         ),
        #         "value": [np.nan] * forecast_horizon,
        #         "series_id": [train_df_tickers["series"].iloc[0]] * forecast_horizon,
        #     }
        # )
        # breakpoint()
        # df_full = pd.concat([df, future_df]).reset_index(drop=True)

        training = TimeSeriesDataSet(
            y_train,
            time_idx="time_idx",
            target="value",
            group_ids=["series_id"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["series_id"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["value"],
        )
        validation = TimeSeriesDataSet(
            y_test,
            time_idx="time_idx",
            target="value",
            group_ids=["series_id"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["series_id"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["value"],
            predict_mode=True,
        )

        train_loader = training.to_dataloader(train=True, batch_size=8)
        val_loader = validation.to_dataloader(train=False, batch_size=1)

        model = RecurrentNetwork.from_dataset(
            training,
            cell_type="LSTM",
            hidden_size=64,
            learning_rate=1e-3,
        )

        trainer = Trainer(max_epochs=100, gradient_clip_val=0.1)
        trainer.fit(model, train_dataloaders=train_loader)

        predictions = model.predict(val_loader, mode="prediction", return_index=True)

        breakpoint()

        # y_train = train_df_tickers.set_index("ds")["y"]
        # model = NeuralForecastLSTM(local_scaler_type="standard", verbose_fit=True)
        # model.fit(y_train, fh=list(range(1, 366)))

        # breakpoint()

        # fh = ForecastingHorizon(
        #     train_df_tickers.set_index("ds")[train_window_length:].index,
        #     is_relative=False,
        # )
        # fh = ForecastingHorizon(
        #     test_df_tickers.set_index("ds").index, is_relative=False
        # )

        # X_train, y_train_dl = convert_to_dl_format(
        #     train_df_tickers["y"], train_window_length, test_window_length
        # )
        # model = NeuralForecastLSTM(local_scaler_type="standard", verbose_fit=True)
        # breakpoint()
        # model.fit(X_train, y_train_dl, fh=fh)

        # preds = []
        # current_input = train_df_tickers[-train_window_length:].values.reshape(
        #     1, train_window_length, 1
        # )

        # for _ in range(365):
        #     X_input = from_2d_array_to_nested(current_input)
        #     y_pred = model.predict(X_input)
        #     preds.append(y_pred[0])
        #     current_input = np.append(current_input[:, 1:, :], [[[y_pred[0]]]], axis=1)

        # # Convert predictions to pandas Series
        # breakpoint()
        # y_pred_series = pd.Series(preds, index=test_df_tickers.index)

        # model_metrics["uc"].append(
        #     metrics(
        #         test_df_tickers, y_pred_uc, label=train_df_tickers["series"].iloc[0]
        #     )
        # )

    for key, one_metrics in model_metrics.items():
        final_metrics = pd.concat(one_metrics)
        final_metrics = final_metrics.sort_index()
        save_path = csv_path / f"{key}" / f"{points}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_metrics.to_csv(save_path)
        print(f"{key}: {final_metrics['mape'].mean()}")
