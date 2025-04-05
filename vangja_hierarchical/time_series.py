import pickle
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_extras as pmx


from vangja_hierarchical.types import (
    YScaleParams,
    TScaleParams,
    Method,
    NutsSampler,
    ScaleMode,
)
from vangja.utils import get_group_definition


class TimeSeriesModel:
    data: pd.DataFrame
    y_scale_params: YScaleParams
    t_scale_params: TScaleParams

    group: np.ndarray
    n_groups: int
    groups_: dict[int, str]

    model: pm.Model
    model_idxs: dict[str, int]
    samples: int
    method: Method
    initvals: dict[str, float]
    map_approx: dict[str, np.ndarray] | None
    trace: az.InferenceData | None

    def _process_data(
        self, data: pd.DataFrame, mode: ScaleMode, t_scale_params: TScaleParams | None
    ) -> None:
        """Converts dataframe to correct format and scale dates and values.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        mode: ScaleMode
            Whether to use maxabs or minmax scaling of the y (target).
        t_scale_params: TScaleParams | None
            Whether to override scale parameters for ds (predictor).
        """
        self.data = data.reset_index(drop=True)
        self.data["ds"] = pd.to_datetime(self.data["ds"])
        self.data.sort_values("ds", inplace=True)

        self.group, self.n_groups, self.groups_ = get_group_definition(
            self.data, "series", "partial"
        )

        self.t_scale_params = (
            t_scale_params
            if t_scale_params is not None
            else {
                "ds_min": self.data["ds"].min(),
                "ds_max": self.data["ds"].max(),
            }
        )
        self.data["t"] = (self.data["ds"] - self.t_scale_params["ds_min"]) / (
            self.t_scale_params["ds_max"] - self.t_scale_params["ds_min"]
        )

        self.y_scale_params = {
            "mode": mode,
            "y_min": 0 if mode == "maxabs" else self.data["y"].min(),
            "y_max": self.data["y"].abs().max()
            if mode == "maxabs"
            else self.data["y"].max(),
        }
        self.data["y"] = (self.data["y"] - self.y_scale_params["y_min"]) / (
            self.y_scale_params["y_max"] - self.y_scale_params["y_min"]
        )

    def _get_model_initvals(self) -> dict[str, float]:
        initvals: dict[str, float] = {"sigma": 1.0}
        for key in self.groups_.keys():
            series: pd.DataFrame = self.data[self.group == key]
            i0, i1 = series["ds"].idxmin(), series["ds"].idxmax()
            T = series["t"].loc[i1] - series["t"].loc[i0]
            slope = (series["y"].loc[i1] - series["y"].loc[i0]) / T
            intercept = series["y"].loc[i0] - slope * series["t"].loc[i0]
            initvals[f"slope_{key}"] = slope
            initvals[f"intercept_{key}"] = intercept

        return initvals

    def get_initval(self, initvals, model: pm.Model) -> dict:
        return {
            model.named_vars["sigma"]: initvals.get("sigma", 1),
            **self._get_initval(initvals, model),
        }

    def fit(
        self,
        data: pd.DataFrame,
        scale_mode: ScaleMode = "absmax",
        t_scale_params: TScaleParams | None = None,
        sigma_sd: float = 0.5,
        method: Method = "mapx",
        samples: int = 0,
        chains: int = 4,
        cores: int = 4,
        nuts_sampler: NutsSampler = "pymc",
        progressbar: bool = True,
        idata: az.InferenceData | None = None,
    ):
        self._process_data(data, scale_mode, t_scale_params)

        self.model = pm.Model()
        self.model_idxs = {}
        self.samples = samples
        self.method = method

        with self.model:
            priors = None
            if idata is not None:
                priors = pmx.utils.prior.prior_from_idata(
                    idata,
                    name="priors",
                    # add a "prior_" prefix to vars
                    **{
                        f"{var}": f"prior_{var}" for var in idata["posterior"].data_vars
                    },
                )

            mu = self.definition(self.model, self.data, self.model_idxs, priors)
            sigma = pm.HalfNormal("sigma", sigma_sd)
            _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=self.data["y"])

            self.map_approx = None
            self.trace = None

        self.initvals = self._get_model_initvals()
        initval_dict = self.get_initval(self.initvals, self.model)

        with self.model:
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
                    progressbar=progressbar,
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
                    progressbar=progressbar,
                )
            else:
                raise NotImplementedError(
                    f"Method {self.method} is not supported at the moment!"
                )

    def save_model(self, filepath: Path, return_objs: bool = False):
        model = {
            "scale_params": self.scale_params,
            "map_approx": self.map_approx,
            "method": self.method,
            "samples": self.samples,
            "other_components": self.other_components,
        }

        if return_objs:
            return model, self.data, self.trace

        filepath.mkdir(parents=True)
        with open(filepath / "model.pkl", "wb") as f:
            pickle.dump(model, f)

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
        self.other_components = pkl.get("other_components", {})

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
                self.other_components,
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

    def predict(self, days, hierarchical_model=False):
        future = self._make_future_df(days)
        forecasts = self._predict(
            future,
            self.method,
            self.map_approx,
            self.trace,
            self.other_components,
            hierarchical_model,
        )

        if hierarchical_model:
            for group_code in range(forecasts.shape[0]):
                future[f"yhat_{group_code}"] = (
                    forecasts[group_code] * self.scale_params["y_max"]
                )
                for model_type, model_cnt in self.model_idxs.items():
                    if model_type.startswith("lt") is False:
                        continue
                    for model_idx in range(model_cnt):
                        component = f"{model_type}_{model_idx}_{group_code}"
                        if component in future.columns:
                            future[component] *= self.scale_params["y_max"]

            return future

        future["yhat"] = forecasts * self.scale_params["y_max"]
        for model_type, model_cnt in self.model_idxs.items():
            if model_type.startswith("lt") is False:
                continue
            for model_idx in range(model_cnt):
                component = f"{model_type}_{model_idx}"
                if component in future.columns:
                    future[component] *= self.scale_params["y_max"]

        return future

    def _predict(
        self,
        future,
        method,
        map_approx,
        trace,
        other_components,
        hierarchical_model=False,
    ):
        if method in ["mapx", "map"]:
            return self._predict_map(
                future, map_approx, other_components, hierarchical_model
            )

        return self._predict_mcmc(future, trace, other_components)

    def plot(self, future, y_true=None, series=""):
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

        plt.plot(future["ds"], future[f"yhat{series}"], lw=1, label=r"$\widehat{y}$")

        plt.legend()
        plot_params = {"idx": 1}
        self._plot(plot_params, future, self.data, self.scale_params, y_true, series)

    def needs_priors(self, *args, **kwargs):
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
            right = self.right.needs_priors(*args, **kwargs)

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

    def hierarchical_definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.hierarchical_definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.hierarchical_definition(*args, **kwargs)

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

    def hierarchical_definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.hierarchical_definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.hierarchical_definition(*args, **kwargs)

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

    def hierarchical_definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.hierarchical_definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.hierarchical_definition(*args, **kwargs)

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
