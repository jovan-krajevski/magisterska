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
    BetaConstant,
    Constant,
    FourierSeasonality,
    LinearTrend,
)
from vangja_simple.components.normal_constant import NormalConstant

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


# for key in model.fit_params["trace"]["posterior"]:
#     if key.startswith("fs_"):
#         model.fit_params["trace"]["posterior"][key] = model.fit_params["trace"][
#             "posterior"
#         ][key][:, :, 0, :]


for point in pd.date_range(f"{year_start}-01-01", f"{year_end}-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    parent_path = Path("./") / "out" / "vangja" / "test64"
    csv_path = parent_path / f"{points}.csv"
    maps_path = parent_path / f"{points}_maps.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.is_file():
        continue

    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=point
    )
    trend = LinearTrend(changepoint_range=1)
    # presidential = FourierSeasonality(
    #     365.25 * 4, 9, allow_tune=True, tune_method="simple"
    # )
    yearly = FourierSeasonality(365.25, 10, allow_tune=True, tune_method="simple")
    # quarterly = FourierSeasonality(365.25 / 4, 5, allow_tune=True, tune_method="simple")
    weekly = FourierSeasonality(7, 3, allow_tune=True, tune_method="simple")
    model = trend ** (weekly + yearly)
    model.fit(train_df_smp)

    slope_mean = model.map_approx[f"lt_{trend.model_idx} - slope"]
    weekly_mean = model.map_approx[
        f"fs_{weekly.model_idx} - beta(p={weekly.period},n={weekly.series_order})"
    ]
    # quarterly_mean = model.map_approx[
    #     f"fs_{quarterly.model_idx} - beta(p={quarterly.period},n={quarterly.series_order})"
    # ]
    yearly_mean = model.map_approx[
        f"fs_{yearly.model_idx} - beta(p={yearly.period},n={yearly.series_order})"
    ]
    # presidential_mean = model.map_approx[
    #     f"fs_{presidential.model_idx} - beta(p={presidential.period},n={presidential.series_order})"
    # ]

    model_metrics = []
    model_maps = []
    trend = LinearTrend(
        n_changepoints=0, allow_tune=True, override_slope_mean_for_tune=slope_mean
    )
    # presidential = FourierSeasonality(
    #     365.25 * 4,
    #     9,
    #     allow_tune=True,
    #     tune_method="simple",
    #     override_beta_mean_for_tune=presidential_mean,
    #     shift_for_tune=False,
    #     shrinkage_strength=1,
    # )
    yearly = FourierSeasonality(
        365.25,
        10,
        allow_tune=True,
        tune_method="simple",
        override_beta_mean_for_tune=yearly_mean,
        shift_for_tune=False,
        shrinkage_strength=1,
    )
    # quarterly = FourierSeasonality(
    #     365.25 / 4,
    #     5,
    #     allow_tune=True,
    #     tune_method="simple",
    #     override_beta_mean_for_tune=quarterly_mean,
    #     shift_for_tune=False,
    #     shrinkage_strength=100,
    # )
    weekly = FourierSeasonality(
        7,
        3,
        allow_tune=True,
        tune_method="simple",
        override_beta_mean_for_tune=weekly_mean,
        shift_for_tune=False,
        shrinkage_strength=1,
    )
    constant = NormalConstant(1, 0.1)
    model = trend ** (weekly + yearly)
    model.load_model(Path("./") / "models" / "test30" / f"{points}")
    # model.scale_params = {
    #     **model.scale_params,
    #     "ds_min": scale_params["ds_min"],
    #     "ds_max": scale_params["ds_max"],
    # }

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
        model.tune(train_df_tickers, progressbar=False)
        yhat = model.predict(365)
        model_metrics.append(
            model.metrics(
                test_df_tickers, yhat, label=train_df_tickers["series"].iloc[0]
            )
        )
        model_maps.append(model.map_approx)
        # print(model_metrics[-1]["mape"].iloc[0])

    final_metrics = pd.concat(model_metrics)
    final_maps = pd.DataFrame.from_records(model_maps, index=final_metrics.index)
    final_metrics = final_metrics.sort_index()
    final_maps = final_maps.sort_index()
    final_metrics.to_csv(csv_path)
    final_maps.to_csv(maps_path)

    print(f"{final_metrics['mape'].mean()}")

    del model
    gc.collect()
    # jax.clear_backends()
    jax.clear_caches()
