import pickle
from pathlib import Path
from typing import Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_extras as pmx
from pkg_resources import non_empty_lines
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)

from vangja_simple.types import ScaleParams


class TimeSeriesModel:
    frozen: bool = False
    fit_params: dict | None = None

    def freeze(self):
        """Prevent this component from fitting and tuning."""
        self.frozen = True

    def unfreeze(self):
        """Allow this component to fit and tune."""
        self.frozen = False

    def _process_data(
        self, data: pd.DataFrame, scale_params: ScaleParams | None = None
    ) -> None:
        """Converts dataframe to correct format and scale dates and values.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor) and y
            (target).

        scale_params: ScaleParams | None
            Values that override scale_params calculated on data. Used when tuning the
            model.
        """
        self.data = data.reset_index(drop=True)
        self.data["ds"] = pd.to_datetime(self.data["ds"])
        self.data.sort_values("ds", inplace=True)

        self.scale_params: ScaleParams = {
            "mode": "maxabs",
            "y_min": 0,
            "y_max": self.data["y"].abs().max(),
            "ds_min": self.data["ds"].min(),
            "ds_max": self.data["ds"].max(),
            # overwrite some of the old scale params for tune
            **(scale_params if scale_params is not None else {}),
        }
        self.data["y"] = self.data["y"] / self.scale_params["y_max"]
        self.data["t"] = (self.data["ds"] - self.scale_params["ds_min"]) / (
            self.scale_params["ds_max"] - self.scale_params["ds_min"]
        )

    def _get_model_initvals(self) -> dict:
        i0, i1 = self.data["ds"].idxmin(), self.data["ds"].idxmax()
        T = self.data["t"].iloc[i1] - self.data["t"].iloc[i0]
        slope = (self.data["y"].iloc[i1] - self.data["y"].iloc[i0]) / T
        intercept = self.data["y"].iloc[i0] - slope * self.data["t"].iloc[i0]
        return {"slope": slope, "intercept": intercept, "sigma": 1.0}

    def get_initval(self, initvals, model: pm.Model):
        return {
            model.named_vars["sigma"]: initvals.get("sigma", 1),
            **self._get_initval(initvals, model),
        }

    def _init_model(self, model, mu):
        with model:
            # will be set during fit/tune
            sigma_sd = pm.Data("sigma_sd", 0.5)
            observed = pm.Data("data", [])

            sigma = pm.HalfNormal("sigma", sigma_sd)
            _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)

            self.map_approx = None
            self.trace = None

    def _fit_model(
        self,
        model,
        sigma_sd=0.5,
        method: Literal[
            "mapx",
            "map",
            "fullrank_advi",
            "advi",
            "svgd",
            "asvgd",
            "nuts",
            "metropolis",
            "demetropolisz",
        ] = "mapx",
        samples=0,
        chains=4,
        cores=4,
        use_prophet_initvals=True,
        progressbar=True,
        nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc",
    ):
        self.samples = samples
        self.method = method
        self.initvals = {}
        if use_prophet_initvals:
            self.initvals = self._get_model_initvals()

        initval_dict = self.get_initval(self.initvals, model)

        with model:
            pm.set_data({"sigma_sd": sigma_sd})
            pm.set_data({"data": self.data["y"]})

            if self.method == "mapx":
                self.map_approx = pmx.find_MAP(
                    method="L-BFGS-B",
                    use_grad=True,
                    initvals=initval_dict,
                    progressbar=progressbar,
                    gradient_backend="jax",
                    compile_kwargs={"mode": "JAX"},
                    options={"maxiter": 1e4},
                )
            elif self.method == "map":
                self.map_approx = pm.find_MAP(
                    start=initval_dict,
                    method="L-BFGS-B",
                    progressbar=progressbar,
                    maxeval=1e-4,
                )
            elif self.method in ["fullrank_advi", "advi", "svgd", "asvgd"]:
                approx = pm.fit(
                    50000,
                    method=self.method,
                    start=initval_dict if self.method != "asvgd" else None,
                )
                self.trace = approx.sample(draws=self.samples)
            elif self.method in ["nuts", "metropolis", "demetropolisz"]:
                step = pm.NUTS()
                if self.method == "metropolis":
                    step = pm.Metropolis()

                if self.method == "demetropolisz":
                    step = pm.DEMetropolisZ()

                self.trace = pm.sample(
                    self.samples,
                    chains=chains,
                    cores=cores,
                    nuts_sampler=nuts_sampler,
                    initvals=initval_dict,
                    step=step,
                )
            else:
                raise NotImplementedError(
                    f"Method {self.method} is not supported at the moment!"
                )

    def fit(
        self,
        data: pd.DataFrame,
        sigma_sd: float = 0.5,
        method: Literal[
            "mapx",
            "map",
            "fullrank_advi",
            "advi",
            "svgd",
            "asvgd",
            "nuts",
            "metropolis",
            "demetropolisz",
        ] = "mapx",
        samples: int = 0,
        chains: int = 4,
        cores: int = 4,
        use_prophet_initvals: bool = True,
        progressbar: bool = True,
        nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc",
    ):
        self._process_data(data)

        self.model = pm.Model()
        self.tuned_model = None
        self.model_idxs = {}
        self._init_model(
            model=self.model,
            mu=self.definition(self.model, self.data, self.model_idxs, self.fit_params),
        )

        self._fit_model(
            self.model,
            sigma_sd=sigma_sd,
            method=method,
            samples=samples,
            chains=chains,
            cores=cores,
            use_prophet_initvals=use_prophet_initvals,
            progressbar=progressbar,
            nuts_sampler=nuts_sampler,
        )
        self.fit_params = {"map_approx": self.map_approx, "trace": self.trace}

    def tune(
        self,
        data: pd.DataFrame,
        sigma_sd: float = 0.5,
        method: Literal[
            "mapx",
            "map",
            "fullrank_advi",
            "advi",
            "svgd",
            "asvgd",
            "nuts",
            "metropolis",
            "demetropolisz",
        ] = "mapx",
        samples: int = 0,
        chains: int = 4,
        cores: int = 4,
        use_prophet_initvals: bool = True,
        progressbar: bool = True,
        nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc",
    ):
        self._process_data(
            data,
            {
                "ds_min": self.scale_params["ds_min"],
                "ds_max": self.scale_params["ds_max"],
            },
        )

        # cache model if tuned multiple times with same self.fit_params
        if self.tuned_model is None:
            self.model_idxs = {}
            self.tuned_model = pm.Model()
            # create priors if trace exists and some component has a
            # prior_from_idata tune method
            priors = None
            if self.fit_params["trace"] is not None and self.needs_priors():
                with self.tuned_model:
                    priors = pmx.utils.prior.prior_from_idata(
                        self.fit_params["trace"],
                        name="priors",
                        # add a "prior_" prefix to vars
                        **{
                            f"{var}": f"prior_{var}"
                            for var in self.fit_params["trace"]["posterior"].data_vars
                        },
                    )
            self._init_model(
                model=self.tuned_model,
                mu=self._tune(
                    self.tuned_model,
                    self.data,
                    self.model_idxs,
                    self.fit_params,
                    priors,
                ),
            )

        self.model = self.tuned_model
        self._fit_model(
            self.model,
            sigma_sd=sigma_sd,
            method=method,
            samples=samples,
            chains=chains,
            cores=cores,
            use_prophet_initvals=use_prophet_initvals,
            progressbar=progressbar,
            nuts_sampler=nuts_sampler,
        )

    def save_model(self, filepath: Path, return_objs: bool = False):
        model = {
            "scale_params": self.scale_params,
            "map_approx": self.map_approx,
            "method": self.method,
            "samples": self.samples,
        }

        if return_objs:
            return model, self.data, self.trace

        filepath.mkdir(parents=True)
        with open(filepath / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "scale_params": self.scale_params,
                    "map_approx": self.map_approx,
                    "method": self.method,
                    "samples": self.samples,
                },
                f,
            )

        with open(filepath / "data.pkl", "wb") as f:
            pickle.dump(self.data, f)

        if self.trace is not None:
            self.trace.to_netcdf(filepath / "trace.nc")

    def load_model(self, filepath: Path, objs: tuple | None = None):
        self.fit_params = {}

        if objs is not None:
            pkl = objs[0]
        else:
            with open(filepath / "model.pkl", "rb") as f:
                pkl = pickle.load(f)

        self.scale_params = pkl["scale_params"]
        map_approx = pkl["map_approx"]
        self.method = pkl["method"]
        self.samples = pkl["samples"]

        if objs is not None:
            self.data = objs[1]
        else:
            with open(filepath / "data.pkl", "rb") as f:
                self.data = pickle.load(f)

        trace_path = filepath / "trace.nc"
        if objs is not None:
            trace = objs[2]
        else:
            trace = az.from_netcdf(trace_path) if trace_path.exists() else None

        self.model = pm.Model()
        self.tuned_model = None
        self.model_idxs = {}
        self._init_model(
            model=self.model,
            mu=self.definition(
                self.model,
                self.data,
                self.model_idxs,
                {"map_approx": map_approx, "trace": trace},
            ),
        )

        self.map_approx = map_approx
        self.trace = trace
        self.fit_params = {"map_approx": self.map_approx, "trace": self.trace}

    def _make_future_df(self, days):
        future = pd.DataFrame(
            {
                "ds": pd.DatetimeIndex(
                    np.hstack(
                        (
                            pd.date_range(
                                self.scale_params["ds_min"],
                                self.scale_params["ds_max"],
                                freq="D",
                            ).to_numpy(),
                            pd.date_range(
                                self.scale_params["ds_max"],
                                self.scale_params["ds_max"] + pd.Timedelta(days, "D"),
                                inclusive="right",
                            ).to_numpy(),
                        )
                    )
                )
            }
        )
        future["t"] = (future["ds"] - self.scale_params["ds_min"]) / (
            self.scale_params["ds_max"] - self.scale_params["ds_min"]
        )
        return future

    def predict(self, days):
        future = self._make_future_df(days)
        forecasts = self._predict(future, self.method, self.map_approx, self.trace)

        future["yhat"] = forecasts * self.scale_params["y_max"]
        for model_type, model_cnt in self.model_idxs.items():
            if model_type.startswith("lt") is False:
                continue
            for model_idx in range(model_cnt):
                component = f"{model_type}_{model_idx}"
                if component in future.columns:
                    future[component] *= self.scale_params["y_max"]

        return future

    def _predict(self, future, method, map_approx, trace):
        if method in ["mapx", "map"]:
            return self._predict_map(future, map_approx)

        return self._predict_mcmc(future, trace)

    def plot(self, future, y_true=None):
        plt.figure(figsize=(14, 100 * 6))
        plt.subplot(100, 1, 1)
        plt.title("Predictions")
        plt.grid()

        plt.scatter(
            self.data["ds"],
            self.data["y"] * self.scale_params["y_max"],
            s=0.5,
            color="C0",
            label="train y",
        )

        if y_true is not None:
            plt.scatter(y_true["ds"], y_true["y"], s=0.5, color="C1", label="y_true")

        plt.plot(future["ds"], future["yhat"], lw=1, label=r"$\widehat{y}$")

        plt.legend()
        plot_params = {"idx": 1}
        self._plot(plot_params, future, self.data, self.scale_params, y_true)

    def metrics(self, y_true, future, label="y"):
        y = y_true["y"]
        yhat = future["yhat"][-len(y) :]
        return pd.DataFrame(
            {
                "mse": {f"{label}": mean_squared_error(y, yhat)},
                "rmse": {f"{label}": root_mean_squared_error(y, yhat)},
                "mae": {f"{label}": mean_absolute_error(y, yhat)},
                "mape": {f"{label}": mean_absolute_percentage_error(y, yhat)},
            }
        )

    def needs_priors(self):
        return False

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __radd__(self, other):
        return AdditiveTimeSeries(other, self)

    def __pow__(self, other):
        return MultiplicativeTimeSeries(self, other)

    def __rpow__(self, other):
        return MultiplicativeTimeSeries(other, self)

    def __mul__(self, other):
        return SimpleMultiplicativeTimeSeries(self, other)

    def __rmul__(self, other):
        return SimpleMultiplicativeTimeSeries(other, self)


class CombinedTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def _get_initval(self, *args, **kwargs):
        left = {}
        right = {}
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._get_initval(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._get_initval(*args, **kwargs)

        return {**left, **right}

    def _plot(self, *args, **kwargs):
        if not (type(self.left) is int or type(self.left) is float):
            self.left._plot(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            self.right._plot(*args, **kwargs)

    def needs_priors(self, *args, **kwargs):
        left = False
        right = False
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.needs_priors(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            right = self.left.needs_priors(*args, **kwargs)

        return left or right


class AdditiveTimeSeries(CombinedTimeSeries):
    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left + right

    def _tune(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._tune(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._tune(*args, **kwargs)

        return left + right

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left + right

    def __str__(self):
        return f"{self.left} + {self.right}"


class MultiplicativeTimeSeries(CombinedTimeSeries):
    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left * (1 + right)

    def _tune(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._tune(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._tune(*args, **kwargs)

        return left * (1 + right)

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left * (1 + right)

    def __str__(self):
        left = f"{self.left}"
        if type(self.left) is AdditiveTimeSeries:
            left = f"({self.left})"

        return f"{left} * (1 + {self.right})"


class SimpleMultiplicativeTimeSeries(CombinedTimeSeries):
    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left * right

    def _tune(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._tune(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._tune(*args, **kwargs)

        return left * right

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left * right

    def __str__(self):
        left = f"{self.left}"
        if type(self.left) is AdditiveTimeSeries:
            left = f"({self.left})"

        right = f"{self.right}"
        if type(self.right) is AdditiveTimeSeries:
            right = f"({self.right})"

        return f"{left} * {right}"
