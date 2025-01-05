import numpy as np
import pandas as pd


def get_group_definition(X, pool_cols, pool_type):
    if pool_type == "complete":
        group = np.zeros(len(X), dtype="int")
        group_mapping = {0: "all"}
        n_groups = 1
    else:
        X[pool_cols] = pd.Categorical(X[pool_cols])
        group = X[pool_cols].cat.codes.values
        group_mapping = dict(enumerate(X[pool_cols].cat.categories))
        n_groups = X[pool_cols].nunique()
    return group, n_groups, group_mapping


def get_changepoints_params(
    short_window, large_window, n_changepoints, changepoint_range=0.8
):
    used = short_window * changepoint_range
    leftover = short_window * (1 - changepoint_range)
    n_changepoints_multiplier = (large_window - leftover) / used
    return (
        n_changepoints,
        changepoint_range,
        n_changepoints * n_changepoints_multiplier,
        1 - leftover / large_window,
    )
