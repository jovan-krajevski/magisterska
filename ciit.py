import argparse
import gc
import shutil
from pathlib import Path

import jax
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


for point in pd.date_range(f"{year_start}", f"{year_end}"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    parent_path = Path("./") / "out" / "vangja" / "test_ciit_1"
    csv_path = parent_path / f"{points}.csv"
    maps_path = parent_path / f"{points}_maps.csv"
    if csv_path.is_file():
        continue

    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=point
    )
    trend = LinearTrend(changepoint_range=1)
    yearly = FourierSeasonality(365.25, 10)
    weekly = FourierSeasonality(7, 3)
    constant = NormalConstant(0, 0.3, deterministic=1)
    constant.freeze()
    model = trend ** (weekly + constant * yearly)
    model.fit(train_df_smp)

    first_objs = model.save_model(parent_path / "model", True)

    model_metrics = []
    model_maps = []

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

        trend = LinearTrend(n_changepoints=0)
        yearly = FourierSeasonality(365.25, 10)
        weekly = FourierSeasonality(7, 3)
        constant = NormalConstant(0, 0.3, deterministic=1)
        model = trend ** (weekly + constant * yearly)
        model.load_model(parent_path / "model", first_objs)

        yearly.freeze()
        constant.freeze()
        model.tune(train_df_tickers, progressbar=False)
        second_objs = model.save_model(parent_path / "model1", True)

        trend.freeze()
        weekly.freeze()
        constant.unfreeze()
        model.load_model(parent_path / "model1", second_objs)
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
