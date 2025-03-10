import time
from pathlib import Path

import pandas as pd

from vangja_simple.components import FourierSeasonality, LinearTrend

df = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
)

train_df = df[:730]
test_df = df[730 : 730 + 365]

model = (
    LinearTrend()
    + FourierSeasonality(period=365.25, series_order=10)
    + FourierSeasonality(period=7, series_order=3)
)
# map fit just to load model in memory
model.fit(train_df)

for method in [
    # "mapx",
    # "metropolis",
    # "demetropolisz",
    # "nuts",
    # "fullrank_advi",
    # "advi",
    # "svgd",
    "asvgd",
]:
    start = time.time()
    model.fit(train_df, method=method, samples=2000)
    model.save_model(Path("./") / "models" / "methods" / f"{method}")

    with open(Path("./") / "models" / "methods" / "timing.txt", "a") as f:
        f.write(f"Method: {method}; {time.time() - start:.2f}s\n")
