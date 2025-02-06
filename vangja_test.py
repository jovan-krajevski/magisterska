from pathlib import Path

import argparse
import pandas as pd
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_simple.components import BetaConstant, FourierSeasonality, LinearTrend, Constant
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

trend = LinearTrend()
yearly = FourierSeasonality(365.25, 10, allow_tune=True, tune_method="simple")
weekly = FourierSeasonality(7, 3, allow_tune=True, tune_method="simple")
model = trend ** (BetaConstant(-1, 1) * yearly + weekly)

model.load_trace(Path("./") / "models" / "40_y10_w.nc")
# for key in model.fit_params["trace"]["posterior"]:
#     if key.startswith("fs_"):
#         model.fit_params["trace"]["posterior"][key] = model.fit_params["trace"][
#             "posterior"
#         ][key][:, :, 0, :]

for point in pd.date_range(f"{year_start}-01-01", f"{year_end}-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    csv_path = Path("./") / "out" / "vangja" / "test11" / f"{points}.csv"
    if csv_path.is_file():
        continue
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
        model_metrics.append(
            model.metrics(
                test_df_tickers, yhat, label=train_df_tickers["series"].iloc[0]
            )
        )

    final_metrics = pd.concat(model_metrics)
    final_metrics.to_csv(csv_path)
    print(f"{final_metrics['mape'].mean()}")
