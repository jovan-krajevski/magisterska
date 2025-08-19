from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_simple.components import FourierSeasonality, LinearTrend

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

for point in pd.date_range("2015-01-01", "2017-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    csv_path = Path("./") / "models" / "test30" / f"{points}"

    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
    )

    model.fit(train_df_smp, samples=1000, method="fullrank_advi")
    model.save_model(csv_path)
