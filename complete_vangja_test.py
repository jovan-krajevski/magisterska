import jax

jax.config.update("jax_platform_name", "cpu")
print(jax.numpy.ones(3).device)

import argparse
import gc
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_simple.components import (
    FourierSeasonality,
    LinearTrend,
)

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
    "trend_tune_method": ["simple", "prior_from_idata"],
    "seasonality_tune_method": ["simple", "prior_from_idata"],
    "trend_loss_factor": [-1, 0, 1],
}

model_params_combined = []


for trend_tune_method in model_params["trend_tune_method"]:
    for seasonality_tune_method in model_params["seasonality_tune_method"]:
        for trend_loss_factor in model_params["trend_loss_factor"]:
            model_params_combined.append(
                {
                    "trend_tune_method": trend_tune_method,
                    "seasonality_tune_method": seasonality_tune_method,
                    "trend_loss_factor": trend_loss_factor,
                }
            )

parent_path = Path("./") / "out" / "vangja2"
parent_path.mkdir(parents=True, exist_ok=True)
pd.DataFrame.from_records(model_params_combined).to_csv(
    parent_path / "model_params.csv"
)

for point in pd.date_range(f"{year_start}", f"{year_end}"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    if (parent_path / "model_0" / f"{point}.csv").is_file():
        for idx, _ in enumerate(model_params_combined):
            scores[idx] = scores.get(idx, [])
            scores[idx].append(
                pd.read_csv(parent_path / f"model_{idx}" / f"{point}.csv", index_col=0)[
                    "mape"
                ].mean()
            )
            print(f"so far {idx}: {sum(scores[idx]) / len(scores[idx])}")

        continue

    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
    )
    trend = LinearTrend(changepoint_range=1)
    yearly = FourierSeasonality(365.25, 10)
    weekly = FourierSeasonality(7, 3)
    model = trend ** (weekly + yearly)
    model.fit(train_df_smp)

    slope_mean = model.map_approx[f"lt_{trend.model_idx} - slope"]
    weekly_mean = model.map_approx[
        f"fs_{weekly.model_idx} - beta(p={weekly.period},n={weekly.series_order})"
    ]
    yearly_mean = model.map_approx[
        f"fs_{yearly.model_idx} - beta(p={yearly.period},n={yearly.series_order})"
    ]

    model_metrics = {}
    model_maps = {}
    models = []

    for params in model_params_combined:
        trend = LinearTrend(
            n_changepoints=0,
            allow_tune=True,
            override_slope_mean_for_tune=slope_mean,
            tune_method=params["trend_tune_method"],
            loss_factor_for_tune=params["trend_loss_factor"],
        )
        yearly = FourierSeasonality(
            365.25,
            10,
            allow_tune=True,
            tune_method=params["seasonality_tune_method"],
            override_beta_mean_for_tune=yearly_mean,
        )
        weekly = FourierSeasonality(7, 3)
        model = trend * (1 + weekly + yearly)
        model.load_model(Path("./") / "models" / "simple_advi" / f"{points}")
        models.append(model)

    min_smp_y = train_df_smp["y"].iloc[-91:].min()
    max_smp_y = train_df_smp["y"].iloc[-91:].max()

    for gspc_ticker in tqdm(gspc_tickers):
        check = generate_train_test_df_around_point(
            window=91,
            horizon=365,
            dfs=[gspc_ticker],
            for_prophet=False,
            point=points,
        )
        if check is None:
            continue

        train_df_tickers, test_df_tickers, scales_tickers = check

        min_y = train_df_tickers["y"].min()
        max_y = train_df_tickers["y"].max()
        if max_y != min_y:
            train_df_tickers["y"] = (train_df_tickers["y"] - min_y) / (
                max_y - min_y
            ) * (max_smp_y - min_smp_y) + min_smp_y

        for idx, model in enumerate(models):
            model.tune(train_df_tickers, progressbar=False)
            yhat = model.predict(365)

            if max_y != min_y:
                yhat["yhat"] = (yhat["yhat"] - min_smp_y) / (max_smp_y - min_smp_y) * (
                    max_y - min_y
                ) + min_y

            model_test_metrics = model.metrics(
                test_df_tickers, yhat, label=train_df_tickers["series"].iloc[0]
            )
            model_metrics[idx] = model_metrics.get(idx, [])
            model_maps[idx] = model_maps.get(idx, [])
            model_metrics[idx].append(model_test_metrics)
            model_maps[idx].append(model.map_approx)

    for idx, _ in enumerate(models):
        csv_path = parent_path / f"model_{idx}"
        csv_path.mkdir(parents=True, exist_ok=True)
        final_metrics = pd.concat(model_metrics[idx])
        final_maps = pd.DataFrame.from_records(
            model_maps[idx], index=final_metrics.index
        )
        final_metrics = final_metrics.sort_index()
        final_maps = final_maps.sort_index()
        final_metrics.to_csv(csv_path / f"{points}.csv")
        final_maps.to_csv(csv_path / f"{points}_maps.csv")

        scores[idx] = scores.get(idx, [])
        scores[idx].append(final_metrics["mape"].mean())

        print(f"{final_metrics['mape'].mean()}")
        print(f"so far: {sum(scores[idx]) / len(scores[idx])}")

    for model in models:
        del model
    gc.collect()
    # jax.clear_backends()
    jax.clear_caches()
