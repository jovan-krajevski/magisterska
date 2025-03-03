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
import itertools


print("START")

dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])

print("DATA READY")

ct_length = [years * 365 for years in [10, 20, 30, 40]]
trend_params = {
    "n_changepoints": [25, 30, 35],
    "changepoint_range": [0.8, 0.9, 1],
    "delta_sd": [0.005, 0.05, 0.5],
}
beta_sds = [0.01, 0.1, 1.0, 10.0]
point = "2015-01-01"


def run_tune(small_tickers, slope_mean, yearly_mean, weekly_mean):
    trend = LinearTrend(
        n_changepoints=0, allow_tune=False, override_slope_mean_for_tune=slope_mean
    )
    yearly = FourierSeasonality(
        365.25,
        10,
        allow_tune=True,
        tune_method="simple",
        override_beta_mean_for_tune=yearly_mean,
        shift_for_tune=False,
        shrinkage_strength=1,
    )
    weekly = FourierSeasonality(
        7,
        3,
        allow_tune=True,
        tune_method="simple",
        override_beta_mean_for_tune=weekly_mean,
        shift_for_tune=False,
        shrinkage_strength=1,
    )
    model = trend ** (weekly + yearly)
    model.load_model(Path("./") / "models" / "test30" / f"{point}")
    model_metrics = []
    for gspc_ticker in tqdm(small_tickers):
        train_df_tickers, test_df_tickers, scales_tickers = gspc_ticker
        model.tune(train_df_tickers, progressbar=False)
        yhat = model.predict(365)
        model_metrics.append(
            model.metrics(
                test_df_tickers, yhat, label=train_df_tickers["series"].iloc[0]
            )
        )

    del model
    gc.collect()
    # jax.clear_backends()
    jax.clear_caches()

    return pd.concat(model_metrics).sort_index()


done = 0

small_tickers = []
for gspc_ticker in tqdm(gspc_tickers):
    check = generate_train_test_df_around_point(
        window=91,
        horizon=365,
        dfs=[gspc_ticker],
        for_prophet=False,
        point=point,
    )
    if check is None:
        continue

    small_tickers.append(check)

for ct in ct_length:
    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=ct, horizon=365, dfs=smp, for_prophet=False, point=point
    )
    all_trend_params = [
        dict(zip(trend_params.keys(), v))
        for v in itertools.product(*trend_params.values())
    ]
    for tp in all_trend_params:
        for beta_sd in beta_sds:
            done += 1
            if done == 1:
                continue

            trend = LinearTrend(**tp, allow_tune=True)
            yearly = FourierSeasonality(365.25, 10, beta_sd=beta_sd, allow_tune=True)
            weekly = FourierSeasonality(7, 3, beta_sd=beta_sd, allow_tune=True)
            model = trend ** (weekly + yearly)

            model.fit(train_df_smp, progressbar=True)
            smp_yhat = model.predict(365)
            smp_metrics = model.metrics(test_df_smp, smp_yhat, label="smp")

            slope_mean = model.map_approx[f"lt_{trend.model_idx} - slope"]
            weekly_mean = model.map_approx[
                f"fs_{weekly.model_idx} - beta(p={weekly.period},n={weekly.series_order})"
            ]
            yearly_mean = model.map_approx[
                f"fs_{yearly.model_idx} - beta(p={yearly.period},n={yearly.series_order})"
            ]

            model_metrics_1 = run_tune(
                small_tickers, slope_mean, yearly_mean, weekly_mean
            )
            # model_metrics_2 = run_tune(61, 30, slope_mean, yearly_mean, weekly_mean)

            print(f"-- {ct} -- {tp} -- {beta_sd} --")
            print(f"smp mape: {smp_metrics['mape'].iloc[0]}")
            print(f"1y mape: {model_metrics_1['mape'].mean()}")
            # print(f"30d mape: {model_metrics_2['mape'].mean()}")
            print(
                f"done: {done} / {len(ct_length) * len(all_trend_params) * len(beta_sds)}"
            )

            with open("cv_results.txt", "a") as f:
                f.write(f"-- {ct} -- {tp} -- {beta_sd} --\n")
                f.write(f"smp mape: {smp_metrics['mape'].iloc[0]}\n")
                f.write(f"1y mape: {model_metrics_1['mape'].mean()}\n")
                # f.write(f"30d mape: {model_metrics_2['mape'].mean()}\n")

            del model
            gc.collect()
            # jax.clear_backends()
            jax.clear_caches()
