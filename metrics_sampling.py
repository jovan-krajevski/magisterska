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

for method in [
    "mapx",
    "metropolis",
    "demetropolisz",
    "nuts",
    "fullrank_advi",
    "advi",
    "svgd",
    "asvgd",
]:
    model.load_model(Path("./") / "models" / "methods" / f"{method}")
    yhat = model.predict(365)
    metrics = model.metrics(test_df, yhat)
    metrics.to_csv(Path("./") / "models" / "methods" / f"{method}_metrics.csv")
    print(metrics)
