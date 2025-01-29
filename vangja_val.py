import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
from vangja_simple.components import (
    BetaConstant,
    Constant,
    FourierSeasonality,
    LinearTrend,
)

parser = argparse.ArgumentParser(
    prog="Vangja Val", description="Run Vangja on validation set", epilog="---"
)

parser.add_argument("-w", "--workers")  # option that takes a value
parser.add_argument("-i", "--idx")

args = parser.parse_args()
workers = int(args.workers)
worker_idx = int(args.idx)

print("Downloading data...")
dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])
print("Data downloaded!")

model_components: list[list] = [
    [LinearTrend(changepoint_range=1)],
    [
        FourierSeasonality(period=365.25, series_order=10, allow_tune=allow_tune)
        for allow_tune in [True]
    ],
    # [
    #     FourierSeasonality(period=365.25 / 4, series_order=7, allow_tune=allow_tune)
    #     for allow_tune in [True, False]
    # ],
    # [
    #     FourierSeasonality(period=365.25 / 12, series_order=5, allow_tune=allow_tune)
    #     for allow_tune in [True, False]
    # ],
    [
        FourierSeasonality(period=7, series_order=3, allow_tune=allow_tune)
        for allow_tune in [True, False]
    ],
]

q = [(0, [mc]) for mc in model_components[0]]
models = []

while len(q):
    level, model = q.pop(0)
    if level + 1 == len(model_components):
        models.append(model)
        continue

    mcs = model_components[level + 1]
    for mc in mcs:
        if mc.allow_tune:
            q.append((level + 1, model + [Constant(lower=-1, upper=1) * mc]))
            q.append((level + 1, model + [BetaConstant(lower=-1, upper=1) * mc]))

        q.append((level + 1, model + [mc]))
        q.append((level + 1, model))


def sum_models(models):
    s = None
    for model in models:
        if s is None:
            s = model
        else:
            s += model

    return s


models = [
    model[0] * sum_models(model[1:]) if len(model) > 1 else model[0] for model in models
] + [sum_models(model) if len(model) > 1 else model[0] for model in models]

str_models = {""}
final_models = []
for model in models:
    str_model = str(model)
    if str_model in str_models:
        continue

    str_models.add(str_model)
    final_models.append(model)

print(f"Total models: {len(final_models)}")


def set_tune_method(model, tune_method, beta_sd):
    if hasattr(model, "left"):
        set_tune_method(model.left, tune_method, beta_sd)
        set_tune_method(model.right, tune_method, beta_sd)

    if hasattr(model, "beta_sd") and model.allow_tune:
        model.tune_method = tune_method
        model.beta_sd = beta_sd


def is_tunable(model):
    left = False
    if hasattr(model, "left"):
        left = left or is_tunable(model.left)

    right = False
    if hasattr(model, "right"):
        right = right or is_tunable(model.right)

    allow_tune = False
    if hasattr(model, "allow_tune"):
        allow_tune = model.allow_tune

    return left or right or allow_tune


point = "2014-01-01"

val_tickers = []
for gspc_ticker in tqdm(gspc_tickers[:10]):
    check = generate_train_test_df_around_point(
        window=91,
        horizon=365,
        dfs=[gspc_ticker],
        for_prophet=False,
        point=point,
    )

    if check is not None:
        val_tickers.append(check)


context_size = 40
model_idx = -1
val_path = Path("./") / "out" / "vangja" / "val3"

start_model_idx = (len(final_models) * 4 // workers) * worker_idx
end_model_idx = (len(final_models) * 4 // workers) * (worker_idx + 1)

print(f"Running validation from {start_model_idx} to {end_model_idx}...")

for beta_sd in [0.0001, 0.001, 0.01, 0.1]:
    check = generate_train_test_df_around_point(
        window=365 * context_size, horizon=365, dfs=smp, for_prophet=False, point=point
    )
    if check is None:
        continue

    train_df_smp, test_df_smp, scales_smp = check
    for model in final_models:
        model_idx += 1
        if not (model_idx >= start_model_idx and model_idx < end_model_idx):
            continue

        print(f"Context: {context_size} years, {model}")
        set_tune_method(model, "simple", 10)
        if hasattr(model, "left"):
            model.left.changepoint_range = 1
        else:
            model.changepoint_range = 1
        model.fit(train_df_smp, progressbar=False)
        yhat = model.predict(365)
        smp_mape = model.metrics(test_df_smp, yhat)["mape"].iloc[0]
        print(f"context mape: {smp_mape}")

        set_tune_method(model, "simple", beta_sd)
        if hasattr(model, "left"):
            model.left.changepoint_range = 0.8
        else:
            model.changepoint_range = 0.8

        csv_path = val_path / f"{model_idx}.csv"
        if csv_path.is_file():
            continue

        model_metrics = []
        for val_ticker in tqdm(val_tickers):
            train_df_tickers, test_df_tickers, scales_tickers = val_ticker
            model.tune(train_df_tickers, progressbar=False)
            yhat = model.predict(365)
            model_metrics.append(
                model.metrics(
                    test_df_tickers, yhat, label=train_df_tickers["series"].iloc[0]
                )
            )

        final_metrics = pd.concat(model_metrics)
        final_metrics.to_csv(csv_path)
        single_mape = final_metrics["mape"].mean()
        with open(val_path / f"model_idxs_{worker_idx}.txt", "a") as f:
            f.write(f"{model_idx},{single_mape},{beta_sd},{smp_mape},{model}\n")

        print(f"Val mape: {final_metrics['mape'].mean()}")
