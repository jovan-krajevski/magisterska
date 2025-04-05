import pandas as pd
from pathlib import Path

from vangja.data_utils import (
    download_data,
    process_data,
    generate_train_test_df_around_point,
)
from time import time

start = time()
dfs = download_data(Path("./data"))
print(f"Download data: {time() - start:.2f}s")
start = time()
indexes = process_data(dfs[0])
print(f"Process data: {time() - start:.2f}s")
start = time()
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])
print(f"Process data: {time() - start:.2f}s")

points = "2015-01-01"
train_df, test_df, scales_df = generate_train_test_df_around_point(
    window=91, horizon=365, dfs=gspc_tickers, for_prophet=False, point=points
)
breakpoint()

results = []

parent_path = Path("./") / "out" / "prophet" / "test2"
scores = []
for point in pd.date_range("2015-01-01", "2017-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    path = parent_path / f"prophet_{points}_multiplicative_lt0_w.csv"
    if path.exists():
        scores.append(pd.read_csv(path, index_col=0).iloc[:-1]["mape"].mean())

results.append(
    (
        sum(scores) / len(scores),
        f"{sum(scores) / len(scores) * 100:.5f}% - prophet, done {len(scores) / 7.32:.2f}%",
    )
)

parent_path = Path("./") / "out" / "es"
scores = []
for point in pd.date_range("2015-01-01", "2017-01-01"):
    points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
    path = parent_path / f"{points}.csv"
    if path.exists():
        scores.append(pd.read_csv(path, index_col=0).iloc[:-1]["mape"].mean())

results.append(
    (
        sum(scores) / len(scores),
        f"{sum(scores) / len(scores) * 100:.5f}% - ets, done {len(scores) / 7.32:.2f}%",
    )
)

for x in range(5, 6):
    parent_path = Path("./") / "out" / "vangja2" / f"model_{x}"
    scores = []
    for point in pd.date_range("2015-01-01", "2017-01-01"):
        points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
        path = parent_path / f"{points}.csv"
        if path.exists():
            scores.append(pd.read_csv(path, index_col=0).iloc[:-1]["mape"].mean())

    results.append(
        (
            sum(scores) / len(scores),
            f"{sum(scores) / len(scores) * 100:.5f}% - vangja_2, model {x}, done {len(scores) / 7.32:.2f}%",
        )
    )

for x in range(6, 8):
    parent_path = Path("./") / "out" / f"timeseers{x}"

    for shrinkage_strength in [1, 10, 50, 100, 500, 1000]:
        if not (parent_path / f"shrinkage_{shrinkage_strength}").exists():
            continue

        scores = []
        for point in pd.date_range("2015-01-01", "2017-01-01"):
            points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
            path = parent_path / f"shrinkage_{shrinkage_strength}" / f"{points}.csv"
            if path.exists():
                scores.append(pd.read_csv(path, index_col=0).iloc[:-1]["mape"].mean())

        results.append(
            (
                sum(scores) / len(scores),
                f"{sum(scores) / len(scores) * 100:.5f}% - hierarchical vangja {x}, shrinkage {shrinkage_strength}, done {len(scores) / 7.32:.2f}%",
            )
        )

for result in sorted(results, key=lambda x: x[0]):
    print(result[1])
