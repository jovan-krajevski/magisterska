import argparse
import gc
import warnings
from pathlib import Path

import arviz as az
import jax
import pandas as pd
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_hierarchical.components import FourierSeasonality, LinearTrend
from vangja_hierarchical.utils import get_group_definition, metrics

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

parser = argparse.ArgumentParser(
    prog="Vangja Test", description="Run Vangja on test set", epilog="---"
)

parser.add_argument("-ys", "--ystart")
parser.add_argument("-ye", "--yend")

args = parser.parse_args()
year_start = args.ystart
year_end = args.yend

print("START")

dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])

print("DATA READY")

scores = {}

model_params: dict[str, list] = {
    "trend_loss_factor": [-1, 0, 1],
    "seasonality_loss_factor": [-1, 0, 1],
    "trend_tune_method": ["parametric", "prior_from_idata"],
    "seasonality_tune_method": ["parametric", "prior_from_idata"],
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

parent_path = Path("./") / "out" / "vangja5"
parent_path.mkdir(parents=True, exist_ok=True)
pd.DataFrame.from_records(model_params_combined).to_csv(
    parent_path / "model_params.csv"
)

for point in pd.date_range(f"{year_start}", f"{year_end}"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    if (parent_path / "model_0" / f"{points}.csv").is_file():
        for idx, _ in enumerate(model_params_combined):
            scores[idx] = scores.get(idx, [])
            scores[idx].append(
                pd.read_csv(
                    parent_path / f"model_{idx}" / f"{points}.csv", index_col=0
                )["mape"].mean()
            )
            print(f"so far {idx}: {sum(scores[idx]) / len(scores[idx])}")

        continue

    model_metrics = {}
    model_maps = {}
    models = []

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
    weekly_mean = large_model.map_approx[
        f"fs_{large_weekly.model_idx} - beta(p={large_weekly.period},n={large_weekly.series_order})"
    ]
    yearly_mean = large_model.map_approx[
        f"fs_{large_yearly.model_idx} - beta(p={large_yearly.period},n={large_yearly.series_order})"
    ]

    for params in model_params_combined:
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
        models.append(model)

    train_df_tickers, test_df_tickers, scales_tickers = (
        generate_train_test_df_around_point(
            window=91,
            horizon=365,
            dfs=gspc_tickers,
            for_prophet=False,
            point=points,
        )
    )

    t_scale_params = {
        "ds_min": train_df_smp["ds"].min(),
        "ds_max": train_df_smp["ds"].max(),
    }

    trace_path = Path("./") / "models" / "simple_advi" / f"{points}" / "trace.nc"
    trace = az.from_netcdf(trace_path)

    for idx, model in enumerate(tqdm(models)):
        model.fit(
            train_df_tickers,
            idata=trace,
            t_scale_params=t_scale_params,
            progressbar=False,
        )
        yhat = model.predict(365)
        model_metrics[idx] = metrics(test_df_tickers, yhat, "partial")
        model_maps[idx] = [model.map_approx]

    print(points)

    for idx, _ in enumerate(models):
        csv_path = parent_path / f"model_{idx}"
        csv_path.mkdir(parents=True, exist_ok=True)
        final_maps = pd.DataFrame.from_records([model_maps[idx]])
        final_metrics = model_metrics[idx].sort_index()
        final_metrics.to_csv(csv_path / f"{points}.csv")
        final_maps.to_csv(csv_path / f"{points}_maps.csv")

        scores[idx] = scores.get(idx, [])
        scores[idx].append(final_metrics["mape"].mean())

        print(f"{final_metrics['mape'].mean()}")
        print(f"so far: {sum(scores[idx]) / len(scores[idx])}")

    for model in models:
        del model
    gc.collect()
    jax.clear_caches()
