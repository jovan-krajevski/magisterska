from pathlib import Path
from tqdm import tqdm
import pandas as pd

from vangja_simple.components import LinearTrend, FourierSeasonality, BetaConstant
from vangja_simple.components.normal_constant import NormalConstant
from vangja.data_utils import (
    generate_train_test_df_around_point,
    download_data,
    process_data,
)

print("Downloading data...")
dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])
print("Data downloaded!")

for yc in [10, 20, 30, 40]:
    trend = LinearTrend(changepoint_range=1)
    yearly = FourierSeasonality(365.25, yc, allow_tune=True, tune_method="simple")
    weekly = FourierSeasonality(7, 3, allow_tune=False, tune_method="simple")
    model = trend ** (yearly + weekly)

    point = "2014-01-01"

    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=point
    )

    model.fit(train_df_smp, mcmc_samples=1000, nuts_sampler="numpyro")
    model.fit_params["trace"].to_netcdf(Path("./") / "models" / f"40_y{yc}_w.nc")
    yhat = model.predict(365)
    print(model.metrics(test_df_smp, yhat)["mape"].iloc[0])

    model_metrics = []
    point = "2014-01-01"

    trend.changepoint_range = 0.8
    # yearly.beta_sd = 0.001
    fit_params = model.fit_params

    model_positive = trend ** (NormalConstant(mu=1, sd=0.1) * yearly + weekly)
    model_positive.fit_params = fit_params
    model_positive.tuned_model = None

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

        train_df_tickers, test_df_tickers, scales_tickers = check
        model.tune(train_df_tickers, progressbar=False)
        yhat = model.predict(365)
        model_metrics.append(model.metrics(test_df_tickers, yhat))

    final_metrics = pd.concat(model_metrics)
    print(f"mape: {final_metrics['mape'].mean()}")
