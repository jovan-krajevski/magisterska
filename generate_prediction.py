import argparse
import warnings
from pathlib import Path

import arviz as az
import pandas as pd

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_hierarchical.components import FourierSeasonality, LinearTrend
from vangja_hierarchical.utils import metrics

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

parser = argparse.ArgumentParser(
    prog="Generate prediction",
    description="Generate prediction from a given model on a given data point",
    epilog="---",
)

parser.add_argument("-p", "--point")
parser.add_argument("-m", "--model")

args = parser.parse_args()
points = args.point
model_idx = int(args.model)

print("START")

dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])

print("DATA READY")


model_params: dict[str, list] = {
    "loss_factor_trend": [0, 1],
    "loss_factor_seasonality": [0, 1],
    "tune_method": ["parametric", "prior_from_idata"],
    "shrinkage_strength": [1, 10, 100, 1000, 10000],
}

model_params_combined = []

for loss_factor_trend in model_params["loss_factor_trend"]:
    for loss_factor_seasonality in model_params["loss_factor_seasonality"]:
        for tune_method in model_params["tune_method"]:
            for shrinkage_strength in model_params["shrinkage_strength"]:
                model_params_combined.append(
                    {
                        "tune_method": tune_method,
                        "shrinkage_strength": shrinkage_strength,
                        "loss_factor_trend": loss_factor_trend,
                        "loss_factor_seasonality": loss_factor_seasonality,
                    }
                )

train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
    window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
)
large_trend = LinearTrend(
    n_changepoints=25,
    changepoint_range=0.99375,
    delta_side="right",
    pool_type="complete",
)
large_yearly = FourierSeasonality(365.25, 10, pool_type="complete")
large_weekly = FourierSeasonality(7, 3, pool_type="complete")
large_model = large_trend * (1 + large_weekly + large_yearly)
large_model.fit(train_df_smp, progressbar=True)
slope_mean = large_model.map_approx[f"lt_{large_trend.model_idx} - slope"]
delta_loc = large_model.map_approx[f"lt_{large_trend.model_idx} - delta"]
weekly_mean = large_model.map_approx[
    f"fs_{large_weekly.model_idx} - beta(p={large_weekly.period},n={large_weekly.series_order})"
]
yearly_mean = large_model.map_approx[
    f"fs_{large_yearly.model_idx} - beta(p={large_yearly.period},n={large_yearly.series_order})"
]

params = model_params_combined[model_idx]
trend = LinearTrend(
    n_changepoints=25,
    tune_method=params["tune_method"],
    delta_tune_method=params["tune_method"],
    loss_factor_for_tune=params["loss_factor_trend"],
    pool_type="partial",
    delta_pool_type="complete",
    delta_side="right",
    override_slope_mean_for_tune=slope_mean,
    override_delta_loc_for_tune=delta_loc,
    shrinkage_strength=params["shrinkage_strength"],
)
yearly = FourierSeasonality(
    365.25,
    10,
    tune_method=params["tune_method"],
    loss_factor_for_tune=params["loss_factor_seasonality"],
    pool_type="partial",
    override_beta_mean_for_tune=yearly_mean,
    shrinkage_strength=params["shrinkage_strength"],
)
weekly = FourierSeasonality(
    7,
    3,
    tune_method=params["tune_method"],
    loss_factor_for_tune=params["loss_factor_seasonality"],
    pool_type="partial",
    override_beta_mean_for_tune=weekly_mean,
    shrinkage_strength=params["shrinkage_strength"],
)
model = trend * (1 + weekly + yearly)

train_df_tickers, test_df_tickers, scales_tickers = generate_train_test_df_around_point(
    window=91,
    horizon=365,
    dfs=gspc_tickers,
    for_prophet=False,
    point=points,
)

train_data = pd.concat([train_df_smp, train_df_tickers])
test_data = pd.concat([test_df_smp, test_df_tickers])

trace_path = Path("./") / "models" / "simple_advi" / f"{points}" / "trace.nc"
trace = az.from_netcdf(trace_path)

model.fit(train_data, idata=trace, progressbar=False)
yhat = model.predict(365)
model_metrics = metrics(test_data, yhat, "partial").sort_index()

yhat.to_csv("predictions.csv")
model_metrics.to_csv("metrics.csv")
