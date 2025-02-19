from vangja_simple.components import LinearTrend, FourierSeasonality
import pandas as pd
from pathlib import Path
import shutil

df = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
)
models_path = Path("./") / "models"


def train_and_save_map():
    model = LinearTrend(n_changepoints=0) + FourierSeasonality(period=7, series_order=3)
    model.fit(df[:100])
    yhat_pre_save = model.predict(10)
    model.save_model(models_path / "test")

    model = LinearTrend(n_changepoints=0) + FourierSeasonality(period=7, series_order=3)
    model.load_model(models_path / "test")
    yhat_after_save = model.predict(10)

    shutil.rmtree(models_path / "test")

    print(abs(yhat_pre_save["yhat"].mean() - yhat_after_save["yhat"].mean()))
    assert (abs(yhat_pre_save["yhat"].mean() - yhat_after_save["yhat"].mean())) < 0.001


def train_and_save_mcmc():
    model = LinearTrend(n_changepoints=0) + FourierSeasonality(period=7, series_order=3)
    model.fit(df[:100], method="nuts", samples=1000)
    yhat_pre_save = model.predict(10)
    model.save_model(models_path / "test")

    model = LinearTrend(n_changepoints=0) + FourierSeasonality(period=7, series_order=3)
    model.load_model(models_path / "test")
    yhat_after_save = model.predict(10)

    shutil.rmtree(models_path / "test")

    print(abs(yhat_pre_save["yhat"].mean() - yhat_after_save["yhat"].mean()))
    assert (abs(yhat_pre_save["yhat"].mean() - yhat_after_save["yhat"].mean())) < 0.001


def train_and_save_advi():
    model = LinearTrend(n_changepoints=0) + FourierSeasonality(period=7, series_order=3)
    model.fit(df[:100], method="fullrank_advi", samples=1000)
    yhat_pre_save = model.predict(10)
    model.save_model(models_path / "test")

    model = LinearTrend(n_changepoints=0) + FourierSeasonality(period=7, series_order=3)
    model.load_model(models_path / "test")
    yhat_after_save = model.predict(10)

    shutil.rmtree(models_path / "test")

    print(abs(yhat_pre_save["yhat"].mean() - yhat_after_save["yhat"].mean()))
    assert (abs(yhat_pre_save["yhat"].mean() - yhat_after_save["yhat"].mean())) < 0.001


def train_and_tune():
    model = LinearTrend(n_changepoints=0, allow_tune=True) + FourierSeasonality(
        period=7, series_order=3, allow_tune=True
    )
    model.fit(df[:100], method="fullrank_advi", samples=1000)
    yhat_pre_save = model.predict(10)
    model.save_model(models_path / "test")

    model = LinearTrend(n_changepoints=0) + FourierSeasonality(period=7, series_order=3)
    model.load_model(models_path / "test")
    model.tune(df[:100])
    yhat_after_save = model.predict(10)

    shutil.rmtree(models_path / "test")

    print(abs(yhat_pre_save["yhat"].mean() - yhat_after_save["yhat"].mean()))
    # assert (yhat_pre_save["yhat"].mean() - yhat_after_save["yhat"].mean()) < 0.001


train_and_save_map()
train_and_save_mcmc()
train_and_save_advi()
train_and_tune()
