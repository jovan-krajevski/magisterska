import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd

from vangja_simple.time_series import TimeSeriesModel
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing as ets
import pytensor.tensor as pt


class ExponentialSmoothing(TimeSeriesModel):
    model_idx: int | None = None
    forecaster: ets | None = None

    def __init__(self, seasonal="additive", sp=7, allow_tune: bool = False):
        self.seasonal = seasonal
        self.sp = sp

        self.allow_tune = allow_tune

    def definition(
        self,
        model: pm.Model,
        other_components: dict,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        fit_params: dict | None,
    ):
        if self.frozen:
            if fit_params is None:
                raise NotImplementedError(
                    "ExponentialSmoothing cannot be frozen before first fit!"
                )

        model_idxs["ets"] = model_idxs.get("ets", 0)
        self.model_idx = model_idxs["ets"]
        model_idxs["ets"] += 1

        fh = ForecastingHorizon(data.index, is_relative=False)
        forecaster = ets(seasonal=self.seasonal, sp=self.sp)
        forecaster.fit(y=data["y"])
        other_components[f"ets_{self.model_idx}"] = forecaster
        return pt.as_tensor_variable(forecaster.predict(fh=fh))

    def _tune(
        self,
        model: TimeSeriesModel,
        other_components: dict,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        prev: dict,
        priors,
    ):
        return self.definition(model, other_components, data, model_idxs, prev)

    def _get_initval(self, initvals, model: pm.Model):
        return {}

    def _predict_map(self, future, map_approx, other_components):
        fh = ForecastingHorizon(future.index, is_relative=False)
        future[f"ets_{self.model_idx}"] = other_components[
            f"ets_{self.model_idx}"
        ].predict(fh=fh)
        return future[f"ets_{self.model_idx}"]

    def _predict_mcmc(self, future, trace, other_components):
        return self._predict_map(future, trace, other_components)

    def _plot(self, plot_params, future, data, scale_params, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"ExponentialSmoothing({self.model_idx})")
        plt.grid()

        plt.plot(future["ds"], future[f"ets_{self.model_idx}"], lw=1)

        plt.legend()

    def __str__(self):
        return f"ETS(at={self.allow_tune})"
