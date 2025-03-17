import jax

# jax.config.update("jax_platform_name", "cpu")
# print(jax.numpy.ones(3).device)

jax.random.key(42)

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_simple.components import FourierSeasonality, LinearTrend
from vangja_simple.components.normal_constant import NormalConstant

print("START")

dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])

print("DATA READY")

points = "2015-01-01"
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
constant = NormalConstant(1, 0.1)
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
    if train_df_tickers["series"].iloc[0] != "LOW":
        continue
    min_y = train_df_tickers["y"].min()
    max_y = train_df_tickers["y"].max()
    if max_y != min_y:
        train_df_tickers["y"] = (train_df_tickers["y"] - min_y) / (max_y - min_y) * (
            max_smp_y - min_smp_y
        ) + min_smp_y

    model.tune(train_df_tickers, progressbar=False)
    yhat = model.predict(365)

    if max_y != min_y:
        yhat["yhat"] = (yhat["yhat"] - min_smp_y) / (max_smp_y - min_smp_y) * (
            max_y - min_y
        ) + min_y
    print(
        model.metrics(test_df_tickers, yhat, label=train_df_tickers["series"].iloc[0])
    )
