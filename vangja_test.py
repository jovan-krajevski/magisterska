from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_simple.components import BetaConstant, FourierSeasonality, LinearTrend

print("START")

dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])

print("DATA READY")

model = LinearTrend() * (
    FourierSeasonality(365.25, 10, allow_tune=True, tune_method="simple")
    + BetaConstant(-1, 1)
    * FourierSeasonality(7, 3, allow_tune=True, tune_method="simple")
)
model.load_trace(Path("./") / "models" / "trace_40_y_bc_w.nc")
for key in model.fit_params["trace"]["posterior"]:
    if key.startswith("fs_"):
        model.fit_params["trace"]["posterior"][key] = model.fit_params["trace"][
            "posterior"
        ][key][:, :, 0, :]
# model.fit_params["map_approx"] = None

# model.left = LinearTrend(pool_cols="series", pool_type="individual")
# model.right.left = FourierSeasonality(
#     365.25, 10, tune_method="simple", pool_cols="series", pool_type="individual"
# )
# model.right.right.right = FourierSeasonality(
#     7,
#     3,
#     allow_tune=True,
#     tune_method="simple",
#     pool_cols="series",
#     pool_type="individual",
# )
for point in pd.date_range("2015-01-01", "2019-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    model_metrics = []
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
        model_metrics.append(model.metrics(test_df_tickers, yhat))

    final_metrics = pd.concat(model_metrics)
    final_metrics.to_csv(Path("./") / "out" / "vangja" / "test" / f"{points}.csv")
    print(f"{final_metrics['mape'].mean()}")
