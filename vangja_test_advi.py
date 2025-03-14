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

scores = []

for point in pd.date_range(f"{year_start}-01-01", f"{year_end}-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    parent_path = Path("./") / "out" / "vangja" / "test204"
    csv_path = parent_path / f"{points}.csv"
    maps_path = parent_path / f"{points}_maps.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.is_file():
        scores.append(pd.read_csv(csv_path, index_col=0)["mape"].mean())
        print(f"so far: {sum(scores) / len(scores)}")
        continue

    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
    )
    trend = LinearTrend(changepoint_range=1)
    yearly = FourierSeasonality(365.25, 10, allow_tune=True, tune_method="simple")
    weekly = FourierSeasonality(7, 3, allow_tune=True, tune_method="simple")
    model = trend ** (weekly + yearly)
    model.fit(train_df_smp)

    slope_mean = model.map_approx[f"lt_{trend.model_idx} - slope"]
    weekly_mean = model.map_approx[
        f"fs_{weekly.model_idx} - beta(p={weekly.period},n={weekly.series_order})"
    ]
    yearly_mean = model.map_approx[
        f"fs_{yearly.model_idx} - beta(p={yearly.period},n={yearly.series_order})"
    ]

    model_metrics = []
    model_maps = []
    trend = LinearTrend(
        n_changepoints=0, allow_tune=True, override_slope_mean_for_tune=slope_mean
    )
    yearly = FourierSeasonality(
        365.25,
        10,
        allow_tune=True,
        tune_method="simple",
        override_beta_mean_for_tune=yearly_mean,
        shift_for_tune=False,
        shrinkage_strength=10,
    )
    weekly = FourierSeasonality(
        7,
        3,
        allow_tune=False,
        tune_method="simple",
        override_beta_mean_for_tune=weekly_mean,
        shift_for_tune=False,
        shrinkage_strength=1,
    )
    model = trend ** (weekly + yearly)
    model.load_model(Path("./") / "models" / "simple_advi" / f"{points}")

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

        model.tune(train_df_tickers, progressbar=False)
        yhat = model.predict(365)

        if max_y != min_y:
            yhat["yhat"] = (yhat["yhat"] - min_smp_y) / (max_smp_y - min_smp_y) * (
                max_y - min_y
            ) + min_y

        model_test_metrics = model.metrics(
            test_df_tickers, yhat, label=train_df_tickers["series"].iloc[0]
        )
        model_metrics.append(model_test_metrics)
        model_maps.append(model.map_approx)

    final_metrics = pd.concat(model_metrics)
    final_maps = pd.DataFrame.from_records(model_maps, index=final_metrics.index)
    final_metrics = final_metrics.sort_index()
    final_maps = final_maps.sort_index()
    final_metrics.to_csv(csv_path)
    final_maps.to_csv(maps_path)

    scores.append(final_metrics["mape"].mean())

    print(f"{final_metrics['mape'].mean()}")
    print(f"so far: {sum(scores) / len(scores)}")

    del model
    gc.collect()
    # jax.clear_backends()
    jax.clear_caches()
