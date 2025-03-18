import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pandas as pd
import pytensor.tensor as pt

from vangja_simple.time_series import TimeSeriesModel


class BetaConstant(TimeSeriesModel):
    model_idx: int | None = None

    def __init__(
        self,
        lower: float,
        upper: float,
        alpha: float = 0.5,
        beta: float = 0.5,
        allow_tune: bool = False,
        deterministic: float | None = None,
    ):
        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.beta = beta
        self.allow_tune = allow_tune
        self.deterministic = deterministic

    def _add_beta(self, fit_params: dict | None, prev_model_idx: int):
        if self.frozen:
            if self.deterministic is not None:
                return self.deterministic

            return pm.Deterministic(
                f"bc_{self.model_idx} - beta(alpha={self.alpha},beta={self.beta})",
                pt.as_tensor_variable(
                    fit_params["map_approx"][
                        f"bc_{prev_model_idx} - beta(alpha={self.alpha},beta={self.beta})"
                    ]
                    if fit_params["map_approx"] is not None
                    else fit_params["trace"]["posterior"][
                        f"bc_{prev_model_idx} - beta(alpha={self.alpha},beta={self.beta})"
                    ].mean(dim=["chain", "draw"])
                ),
            )

        return pm.Beta(
            f"bc_{self.model_idx} - beta(alpha={self.alpha},beta={self.beta})",
            alpha=self.alpha,
            beta=self.beta,
        )

    def definition(
        self,
        model: pm.Model,
        other_components: dict,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        fit_params: dict | None,
    ):
        prev_model_idx = self.model_idx
        if self.frozen:
            if fit_params is None and self.deterministic is None:
                raise NotImplementedError(
                    "BetaConstant can be frozen before first fit if and only if deterministic is not None!"
                )

        model_idxs["c"] = model_idxs.get("c", 0)
        self.model_idx = model_idxs["c"]
        model_idxs["c"] += 1

        with model:
            beta = self._add_beta(fit_params, prev_model_idx)
            c = pm.Deterministic(
                f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})",
                beta * (self.upper - self.lower) + self.lower,
            )

        return c

    def _tune(self, model, other_components, data, model_idxs, prev, priors):
        return self.definition(model, other_components, data, model_idxs, prev)

    def _get_initval(self, initvals, model: pm.Model):
        return {}

    def _predict_map(self, future, map_approx, other_components):
        future[f"bc_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * map_approx[f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"]
        )
        return future[f"bc_{self.model_idx}"]

    def _predict_mcmc(self, future, trace, other_components):
        future[f"bc_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * trace["posterior"][
                f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"
            ]
            .to_numpy()[:, :]
            .mean()
        )

        return future[f"bc_{self.model_idx}"]

    def _plot(self, plot_params, future, data, scale_params, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"BetaConstant({self.model_idx},l={self.lower},u={self.upper})")
        plt.bar(0, future[f"bc_{self.model_idx}"][0])
        plt.axhline(0, c="k", linewidth=3)

    def __str__(self):
        return f"BC(alpha={self.alpha},beta={self.beta},l={self.lower},u={self.upper},at={self.allow_tune})"
