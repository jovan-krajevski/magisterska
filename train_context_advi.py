from pathlib import Path
from tqdm import tqdm
import pandas as pd

from vangja_simple.components import LinearTrend, FourierSeasonality
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

trend = LinearTrend(changepoint_range=1)
decenial = FourierSeasonality(365.25 * 10, 4, allow_tune=True, tune_method="simple")
presidential = FourierSeasonality(365.25 * 4, 9, allow_tune=True, tune_method="simple")
yearly = FourierSeasonality(365.25, 10, allow_tune=True, tune_method="simple")
weekly = FourierSeasonality(7, 3, allow_tune=False, tune_method="simple")
model = trend ** (weekly + yearly)

point = "2014-01-01"

train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
    window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=point
)

model.fit(train_df_smp, samples=1000, method="fullrank_advi")
model.save_model(Path("./") / "models" / "advi_40_y_w")

yhat = model.predict(365)
print(model.metrics(test_df_smp, yhat)["mape"].iloc[0])

model_metrics = []
trend.changepoint_range = 0.8
point = "2015-01-01"
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
