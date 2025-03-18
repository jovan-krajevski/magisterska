from pathlib import Path
from tqdm import tqdm
import pandas as pd

from vangja_simple.components import (
    LinearTrend,
    FourierSeasonality,
    ExponentialSmoothing,
)
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
ets = ExponentialSmoothing()
decenial = FourierSeasonality(365.25 * 10, 4, allow_tune=True, tune_method="simple")
presidential = FourierSeasonality(365.25 * 4, 9, allow_tune=True, tune_method="simple")
yearly = FourierSeasonality(365.25, 10, allow_tune=True, tune_method="simple")
weekly = FourierSeasonality(7, 3, allow_tune=False, tune_method="simple")
model = ets ** (yearly)

for point in pd.date_range("2015-01-01", "2017-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    csv_path = Path("./") / "models" / "advi_ets" / f"{points}"

    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
    )

    model.fit(train_df_smp, samples=1000, method="advi")
    model.save_model(csv_path)
