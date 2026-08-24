from pathlib import Path

from tft.utils import prepare_sp500_data
from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)

# Download and process data
dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])


train_df_smp, val_df_smp, scales_smp = generate_train_test_df_around_point(
    window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point="2014-01-01"
)
_, test_df_smp, _ = generate_train_test_df_around_point(
    window=1, horizon=365, dfs=smp, for_prophet=False, point="2015-01-01"
)

# Prepare data for S&P500 pretraining
print("Preparing S&P500 data...")
train_df = prepare_sp500_data(train_df_smp[0].copy())
val_df = prepare_sp500_data(val_df_smp[0].copy())
test_df = prepare_sp500_data(test_df_smp[0].copy())

# Add time index for TFT
train_df["time_idx"] = range(len(train_df))
val_df["time_idx"] = range(len(train_df), len(train_df) + len(val_df))
test_df["time_idx"] = range(
    len(train_df) + len(val_df), len(train_df) + len(val_df) + len(test_df)
)
