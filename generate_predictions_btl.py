import warnings
from pathlib import Path
from random import randint

import arviz as az
import pandas as pd

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_hierarchical.components import FourierSeasonality, LinearTrend
from vangja_hierarchical.utils import get_group_definition, metrics

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# parser = argparse.ArgumentParser(
#     prog="Generate prediction",
#     description="Generate prediction from a given model on a given data point",
#     epilog="---",
# )

# parser.add_argument("-p", "--point")
# parser.add_argument("-m", "--model")
# parser.add_argument("-t", "--ticker")

# args = parser.parse_args()
# points = args.point
# model_idx = int(args.model)
# ticker = args.ticker


model_idx = 0

print("START")

dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])

print("DATA READY")


model_params: dict[str, list] = {
    "trend_loss_factor": [0],
    "seasonality_loss_factor": [0],
    "trend_tune_method": ["parametric"],
    "seasonality_tune_method": ["parametric"],
}

model_params_combined = []

for trend_tune_method in model_params["trend_tune_method"]:
    for seasonality_tune_method in model_params["seasonality_tune_method"]:
        for seasonality_loss_factor in model_params["seasonality_loss_factor"]:
            for trend_loss_factor in model_params["trend_loss_factor"]:
                model_params_combined.append(
                    {
                        "seasonality_loss_factor": seasonality_loss_factor,
                        "trend_loss_factor": trend_loss_factor,
                        "trend_tune_method": trend_tune_method,
                        "seasonality_tune_method": seasonality_tune_method,
                    }
                )

for _ in range(20):
    day = randint(1, 28)
    month = randint(1, 12)
    year = randint(2015, 2016)
    points = f"{year}-{month:02d}-{day:02d}"

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
        n_changepoints=0,
        tune_method=params["trend_tune_method"],
        loss_factor_for_tune=params["trend_loss_factor"],
        pool_type="individual",
        delta_side="right",
        override_slope_mean_for_tune=slope_mean,
    )
    yearly = FourierSeasonality(
        365.25,
        10,
        tune_method=params["seasonality_tune_method"],
        loss_factor_for_tune=params["seasonality_loss_factor"],
        pool_type="individual",
        override_beta_mean_for_tune=yearly_mean,
    )
    weekly = FourierSeasonality(
        7,
        3,
        tune_method=params["seasonality_tune_method"],
        loss_factor_for_tune=params["seasonality_loss_factor"],
        pool_type="individual",
        override_beta_mean_for_tune=weekly_mean,
    )
    model = trend * (1 + weekly + yearly)

    train_df_tickers, test_df_tickers, scales_tickers = (
        generate_train_test_df_around_point(
            window=91,
            horizon=365,
            dfs=gspc_tickers,
            for_prophet=False,
            point=points,
        )
    )

    train_data = pd.concat([train_df_smp, train_df_tickers])
    test_data = pd.concat([test_df_smp, test_df_tickers])

    t_scale_params = {
        "ds_min": train_df_smp["ds"].min(),
        "ds_max": train_df_smp["ds"].max(),
    }

    trace_path = Path("./") / "models" / "simple_advi" / f"{points}" / "trace.nc"
    trace = az.from_netcdf(trace_path)

    model.fit(train_data, idata=trace, t_scale_params=t_scale_params)
    yhat = model.predict(365)
    model_metrics = metrics(test_data, yhat, "partial").sort_index()

    test_group, _, test_groups_ = get_group_definition(test_data, "partial")
    for group_code, group_name in test_groups_.items():
        group_idx = test_group == group_code
        if model_metrics.loc[group_name]["mape"] < 0.02:
            yhat[f"yhat_{group_code}"][-365:].to_csv(
                f"predictions/{points}_{group_name}.csv"
            )

    # model_metrics.to_csv("metrics.csv")
