import argparse
import gc
import warnings
from pathlib import Path

import jax
import pandas as pd
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_hierarchical.components import FourierSeasonality, LinearTrend
from vangja_hierarchical.time_series import TimeSeriesModel
from vangja_hierarchical.utils import metrics

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

scores = {}

parent_path = Path("./") / "out" / "timeseers"
parent_path.mkdir(parents=True, exist_ok=True)
shrinkage_strengths = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

for point in pd.date_range(f"{year_start}", f"{year_end}"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    if (parent_path / "model_0" / f"{points}.csv").is_file():
        for idx, _ in enumerate(shrinkage_strengths):
            scores[idx] = scores.get(idx, [])
            scores[idx].append(
                pd.read_csv(
                    parent_path / f"model_{idx}" / f"{points}.csv",
                    index_col=0,
                )["mape"]
                .iloc[:-1]
                .mean()
            )
            print(f"so far {idx}: {sum(scores[idx]) / len(scores[idx])}")

        continue

    model_metrics = {}
    model_maps = {}
    models: list[TimeSeriesModel] = []

    train_df_smp, test_df_smp, scales_smp = generate_train_test_df_around_point(
        window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
    )

    for shrinkage_strength in shrinkage_strengths:
        trend = LinearTrend(
            n_changepoints=25,
            tune_method=None,
            pool_type="partial",
            delta_pool_type="partial",
            delta_side="left",
            shrinkage_strength=1,
        )
        yearly = FourierSeasonality(
            365.25,
            10,
            tune_method=None,
            pool_type="partial",
            shrinkage_strength=shrinkage_strength,
        )
        weekly = FourierSeasonality(
            7,
            3,
            tune_method=None,
            pool_type="partial",
            shrinkage_strength=shrinkage_strength,
        )
        model = trend * (1 + weekly + yearly)
        models.append(model)

    train_df_tickers, test_df_tickers, scales_tickers = (
        generate_train_test_df_around_point(
            window=91,
            horizon=365,
            dfs=gspc_tickers,
            for_prophet=False,
            point=points,
        )
    )

    train_data = pd.concat([train_df_smp, train_df_tickers])
    test_data = pd.concat([test_df_smp, test_df_tickers])

    for idx, model in enumerate(tqdm(models)):
        model.fit(
            train_data,
            progressbar=False,
            scale_mode="complete",
            sigma_pool_type="complete",
        )
        yhat = model.predict(365)
        model_metrics[idx] = metrics(test_data, yhat, "partial")
        model_maps[idx] = [model.map_approx]

    print(points)

    for idx, _ in enumerate(models):
        csv_path = parent_path / f"model_{idx}"
        csv_path.mkdir(parents=True, exist_ok=True)
        final_maps = pd.DataFrame.from_records([model_maps[idx]])
        final_metrics = model_metrics[idx].sort_index()
        final_metrics.to_csv(csv_path / f"{points}.csv")
        final_maps.to_csv(csv_path / f"{points}_maps.csv")

        scores[idx] = scores.get(idx, [])
        scores[idx].append(final_metrics["mape"].iloc[:-1].mean())

        print(f"{final_metrics['mape'].mean()}")
        print(f"so far: {sum(scores[idx]) / len(scores[idx])}")

    for model in models:
        del model
    gc.collect()
    jax.clear_caches()
