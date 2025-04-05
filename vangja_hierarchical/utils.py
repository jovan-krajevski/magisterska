import numpy as np
import pandas as pd
from typing import Literal
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)


def get_group_definition(
    data: pd.DataFrame, pool_type: Literal["partial", "complete", "indivdual"]
) -> tuple[np.ndarray, int, dict[int, str]]:
    """Assign group codes to different series.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe that must at least have columns ds (predictor), y
        (target) and series (name of time series).
    pool_type : Literal["partial", "complete", "indivdual"]
        Type of pooling performed when sampling.
    """
    pool_cols = "series"
    if pool_type == "complete":
        group = np.zeros(len(data), dtype="int")
        group_mapping = {0: "all"}
        n_groups = 1
    else:
        data[pool_cols] = pd.Categorical(data[pool_cols])
        group = data[pool_cols].cat.codes.values
        group_mapping = dict(enumerate(data[pool_cols].cat.categories))
        n_groups = data[pool_cols].nunique()

    return group, n_groups, group_mapping


def metrics(y_true, future):
    metrics = {"mse": {}, "rmse": {}, "mae": {}, "mape": {}}
    test_group, _, test_groups_ = get_group_definition(y_true, "series", "partial")
    for group_code, group_name in test_groups_.items():
        group_idx = test_group == group_code
        y = y_true["y"][group_idx]
        yhat = future[f"yhat_{group_code}"][-len(y) :]
        metrics["mse"][group_name] = mean_squared_error(y, yhat)
        metrics["rmse"][group_name] = root_mean_squared_error(y, yhat)
        metrics["mae"][group_name] = mean_absolute_error(y, yhat)
        metrics["mape"][group_name] = mean_absolute_percentage_error(y, yhat)

    return pd.DataFrame(metrics)
