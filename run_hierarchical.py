from pathlib import Path
from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
import pandas as pd
from vangja_simple.components import FourierSeasonality, LinearTrend
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])

trend = LinearTrend()
yearly = FourierSeasonality(365.25, 10)
weekly = FourierSeasonality(7, 3)
model = trend ** (weekly + yearly)

for shrinkage_strength in [100, 10, 50, 500, 1000]:
    scores = []
    yearly.shrinkage_strength = shrinkage_strength
    weekly.shrinkage_strength = shrinkage_strength

    for point in pd.date_range("2015-01-01", "2017-01-01"):
        points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
        parent_path = (
            Path("./") / "out" / "timeseers" / f"shrinkage_{shrinkage_strength}"
        )
        csv_path = parent_path / f"{points}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.is_file():
            scores.append(pd.read_csv(csv_path, index_col=0).iloc[:-1]["mape"].mean())
            print(f"so far: {sum(scores) / len(scores)}")
            continue

        train_smp, test_smp, scales_smp = generate_train_test_df_around_point(
            window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
        )
        train_df, test_df, scales_df = generate_train_test_df_around_point(
            window=91, horizon=365, dfs=gspc_tickers, for_prophet=False, point=points
        )
        train_data = pd.concat([train_smp, train_df])
        test_data = pd.concat([test_smp, test_df])

        model.fit(train_data, hierarchical_model=True)
        yhat = model.predict(365, True)
        metrics = model.hierarchical_metrics(test_data, yhat)

        metrics.to_csv(csv_path)
        scores.append(metrics.iloc[:-1]["mape"].mean())
        print(f"{metrics['mape'].mean()}")
        print(f"so far: {sum(scores) / len(scores)}")
