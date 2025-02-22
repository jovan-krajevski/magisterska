import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pandas as pd
import pytensor.tensor as pt

from vangja_simple.time_series import TimeSeriesModel


class Constant(TimeSeriesModel):
    model_idx: int | None = None

    def __init__(
        self,
        lower: float,
        upper: float,
        allow_tune: bool = False,
        deterministic: float | None = None,
    ):
        self.lower = lower
        self.upper = upper
        self.allow_tune = allow_tune
        self.deterministic = deterministic

    def _add_c(self, fit_params: dict | None, prev_model_idx: int):
        if self.frozen:
            if self.deterministic is not None:
                return pm.Deterministic(
                    f"c_{self.model_idx} - normal(mu={self.mu},sd={self.sd})",
                    self.deterministic,
                )

            return pm.Deterministic(
                f"c_{self.model_idx} - normal(mu={self.mu},sd={self.sd})",
                pt.as_tensor_variable(
                    fit_params["map_approx"][
                        f"c_{prev_model_idx} - normal(mu={self.mu},sd={self.sd})"
                    ]
                    if fit_params["map_approx"] is not None
                    else fit_params["trace"]["posterior"][
                        f"c_{prev_model_idx} - normal(mu={self.mu},sd={self.sd})"
                    ].mean(dim=["chain", "draw"])
                ),
            )

        return pm.Uniform(
            f"c_{self.model_idx} - c(l={self.lower},u={self.upper})",
            lower=self.lower,
            upper=self.upper,
        )

    def definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        fit_params: dict | None,
    ):
        prev_model_idx = self.model_idx
        if self.frozen:
            if fit_params is None and self.deterministic is None:
                raise NotImplementedError(
                    "UniformConstant can be frozen before first fit if and only if deterministic is not None!"
                )

        model_idxs["c"] = model_idxs.get("c", 0)
        self.model_idx = model_idxs["c"]
        model_idxs["c"] += 1

        with model:
            c = self._add_c(fit_params, prev_model_idx)

        return c

    def _tune(self, model, data, model_idxs, prev):
        return self.definition(model, data, model_idxs, prev)

    def _get_initval(self, initvals, model: pm.Model):
        return {}

    def _predict_map(self, future, map_approx):
        future[f"c_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * map_approx[f"c_{self.model_idx} - c(l={self.lower},u={self.upper})"]
        )

        return future[f"c_{self.model_idx}"]

    def _predict_mcmc(self, future, trace):
        future[f"c_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * trace["posterior"][
                f"c_{self.model_idx} - c(l={self.lower},u={self.upper})"
            ]
            .to_numpy()[:, :]
            .mean()
        )

        return future[f"c_{self.model_idx}"]

    def _plot(self, plot_params, future, data, scale_params, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"Constant({self.model_idx},l={self.lower},u={self.upper})")
        plt.bar(0, future[f"c_{self.model_idx}"][0])
        plt.axhline(0, c="k", linewidth=3)

    def __str__(self):
        return f"C(l={self.lower},u={self.upper},at={self.allow_tune})"
