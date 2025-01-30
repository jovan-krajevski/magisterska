from pathlib import Path

from vangja_simple.components import LinearTrend, FourierSeasonality, Constant
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

model = (
    LinearTrend(changepoint_range=1)
    + FourierSeasonality(365.25, 10, allow_tune=True, tune_method="simple")
    + Constant(-1, 1) * FourierSeasonality(7, 3, allow_tune=True, tune_method="simple")
)

point = "2014-01-01"

train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
    window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=point
)

model.fit(train_df_smp, mcmc_samples=1000, nuts_sampler="numpyro")
model.fit_params["trace"].to_netcdf(Path("./") / "models" / "108.nc")
yhat = model.predict(365)
print(model.metrics(test_df_smp, yhat)["mape"].iloc[0])
model.plot(yhat, test_df_smp)
