import time
from pathlib import Path

import pandas as pd

from vangja_simple.components import FourierSeasonality, LinearTrend

df = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
)

train_df = df[:730]
test_df = df[730 : 730 + 365]


# map fit just to load model in memory
# model.fit(train_df)

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
    model = (
        LinearTrend()
        + FourierSeasonality(period=365.25, series_order=10)
        + FourierSeasonality(period=7, series_order=3)
    )
    model.load_model(Path("./") / "methods_1000" / f"{method}")
    yhat = model.predict(365)
    metrics = model.metrics(test_df, yhat)
    metrics.to_csv(Path("./") / "methods_1000" / f"{method}" / "metrics.csv")
    print(
        f"\\textit{{{method}}} & {metrics['mse'].iloc[0]:.4f} & {metrics['rmse'].iloc[0]:.4f} & {metrics['mae'].iloc[0]:.4f} & {metrics['mape'].iloc[0]:.4f} \\\\ \n"
    )
