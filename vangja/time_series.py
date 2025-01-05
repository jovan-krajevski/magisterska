import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)

from vangja.utils import get_group_definition


class TimeSeriesModel:
    def _scale_data(self):
        self.y_min = 0
        self.y_max = self.data["y"].abs().max()
        self.ds_min = self.data["ds"].min()
        self.ds_max = self.data["ds"].max()

        self.data["y"] = self.data["y"] / self.y_max
        self.data["t"] = (self.data["ds"] - self.ds_min) / (self.ds_max - self.ds_min)

    def _process_data(self):
        self.data["ds"] = pd.to_datetime(self.data["ds"])
        self.data.sort_values("ds", inplace=True)
        self._scale_data()

    def _model_init(self):
        i0, i1 = self.data["ds"].idxmin(), self.data["ds"].idxmax()
        T = self.data["t"].iloc[i1] - self.data["t"].iloc[i0]
        slope = (self.data["y"].iloc[i1] - self.data["y"].iloc[i0]) / T
        intercept = self.data["y"].iloc[i0] - slope * self.data["t"].iloc[i0]
        return {
            "slope": slope,
            "intercept": intercept,
            "delta": 0.0,
            "beta": 0.0,
            "sigma": 1.0,
        }

    def fit(
        self,
        data,
        sigma_sd=0.5,
        mcmc_samples=0,
        chains=4,
        cores=4,
        use_prophet_initvals=True,
        progressbar=True,
    ):
        self.mcmc_samples = mcmc_samples

        self.data = data.reset_index(drop=True)
        self._process_data()

        self.initvals = {}
        if use_prophet_initvals:
            self.initvals = self._model_init()

        self.model = pm.Model()
        self.model_idxs = {}
        mu = self.definition(self.model, self.data, self.initvals, self.model_idxs)

        with self.model:
            sigma = pm.HalfNormal(
                "sigma", sigma_sd, initval=self.initvals.get("sigma", 1)
            )
            _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=self.data["y"])

            self.map_approx = None
            self.trace = None
            if self.mcmc_samples == 0:
                self.map_approx = pm.find_MAP(progressbar=progressbar, maxeval=1e4)
            else:
                self.trace = pm.sample(self.mcmc_samples, chains=chains, cores=cores)

    def tune(
        self,
        data,
        sigma_sd=0.5,
        mcmc_samples=0,
        chains=4,
        cores=4,
        use_prophet_initvals=True,
        progressbar=True,
    ):
        self.mcmc_samples = mcmc_samples

        self.data = data.reset_index(drop=True)
        self._process_data()

        self.initvals = {}
        if use_prophet_initvals:
            self.initvals = self._model_init()

        self.model = pm.Model()
        self.model_idxs = {}
        mu = self._tune(
            self.model,
            self.data,
            self.initvals,
            self.model_idxs,
            {"map_approx": self.map_approx, "trace": self.trace},
        )

        with self.model:
            sigma = pm.HalfNormal(
                "sigma", sigma_sd, initval=self.initvals.get("sigma", 1)
            )
            _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=self.data["y"])

            self.map_approx = None
            self.trace = None
            if self.mcmc_samples == 0:
                self.map_approx = pm.find_MAP(progressbar=progressbar, maxeval=1e4)
            else:
                self.trace = pm.sample(self.mcmc_samples, chains=chains, cores=cores)

    def _make_future_df(self, days):
        future = pd.DataFrame(
            {
                "ds": pd.DatetimeIndex(
                    np.hstack(
                        (
                            self.data["ds"].unique().to_numpy(),
                            pd.date_range(
                                self.ds_max,
                                self.ds_max + pd.Timedelta(days, "D"),
                                inclusive="right",
                            ).to_numpy(),
                        )
                    )
                )
            }
        )
        future["t"] = (future["ds"] - self.ds_min) / (self.ds_max - self.ds_min)
        return future

    def predict(self, days):
        future = self._make_future_df(days)
        forecasts = self._predict(
            future, self.mcmc_samples, self.map_approx, self.trace
        )

        for group_code in range(forecasts.shape[0]):
            future[f"yhat_{group_code}"] = forecasts[group_code] * self.y_max
            for model_type, model_cnt in self.model_idxs.items():
                if model_type.startswith("fs"):
                    continue
                for model_idx in range(model_cnt):
                    component = f"{model_type}_{model_idx}_{group_code}"
                    if component in future.columns:
                        future[component] *= self.y_max

        return future

    def _predict(self, future, mcmc_samples, map_approx, trace):
        if mcmc_samples == 0:
            return self._predict_map(future, map_approx)

        return self._predict_mcmc(future, trace)

    def plot(self, future, y_true=None, pool_cols=None):
        plt.figure(figsize=(14, 100 * 6))
        plt.subplot(100, 1, 1)
        plt.title("Predictions")
        plt.grid()

        group, _, groups_ = get_group_definition(self.data, pool_cols, "not_complete")
        for group_code, group_name in groups_.items():
            group_idx = group == group_code
            color = np.random.rand(3)
            plt.scatter(
                self.data["ds"][group_idx],
                self.data["y"][group_idx] * self.y_max,
                s=0.5,
                color=color,
                label=group_name,
            )

        if y_true is not None:
            test_group, _, test_groups_ = get_group_definition(
                y_true, pool_cols, "not_complete"
            )
            for group_code, group_name in test_groups_.items():
                group_idx = test_group == group_code
                color = np.random.rand(3)
                plt.scatter(
                    y_true["ds"][group_idx],
                    y_true["y"][group_idx],
                    s=0.5,
                    color=color,
                    label=f"y - {group_name}",
                )

        for group_code, group_name in groups_.items():
            plt.plot(
                future["ds"],
                future[f"yhat_{group_code}"],
                lw=1,
                label=f"yhat - {group_name}",
            )

        plt.legend()
        plot_params = {"idx": 1}
        self._plot(plot_params, future, self.data, self.y_max, y_true)

    def metrics(self, y_true, future, pool_cols=None, pool_type="individual"):
        metrics = {"mse": {}, "rmse": {}, "mae": {}, "mape": {}}
        test_group, _, test_groups_ = get_group_definition(y_true, pool_cols, pool_type)
        for group_code, group_name in test_groups_.items():
            group_idx = test_group == group_code
            y = y_true["y"][group_idx]
            yhat = future[f"yhat_{group_code}"][-len(y) :]
            metrics["mse"][group_name] = mean_squared_error(y, yhat)
            metrics["rmse"][group_name] = root_mean_squared_error(y, yhat)
            metrics["mae"][group_name] = mean_absolute_error(y, yhat)
            metrics["mape"][group_name] = mean_absolute_percentage_error(y, yhat)

        return pd.DataFrame(metrics)

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __mul__(self, other):
        return MultiplicativeTimeSeries(self, other)


class AdditiveTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) + self.right.definition(
            *args, **kwargs
        )

    def _tune(self, *args, **kwargs):
        return self.left._tune(*args, **kwargs) + self.right._tune(*args, **kwargs)

    def _predict(self, *args, **kwargs):
        return self.left._predict(*args, **kwargs) + self.right._predict(
            *args, **kwargs
        )

    def _plot(self, *args, **kwargs):
        self.left._plot(*args, **kwargs)
        self.right._plot(*args, **kwargs)

    def __str__(self):
        return f"{self.left} + {self.right}"


class MultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) * (
            1 + self.right.definition(*args, **kwargs)
        )

    def _tune(self, *args, **kwargs):
        return self.left._tune(*args, **kwargs) * (
            1 + self.right._tune(*args, **kwargs)
        )

    def _predict(self, *args, **kwargs):
        return self.left._predict(*args, **kwargs) * (
            1 + self.right._predict(*args, **kwargs)
        )

    def _plot(self, *args, **kwargs):
        self.left._plot(*args, **kwargs)
        self.right._plot(*args, **kwargs)

    def __str__(self):
        left = f"{self.left}"
        if type(self.left) is AdditiveTimeSeries:
            left = f"({self.left})"

        right = f"{self.right}"
        if type(self.right) is AdditiveTimeSeries:
            right = f"({self.right})"

        return f"{left} * {right}"
