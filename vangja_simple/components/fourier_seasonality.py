import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from vangja_simple.time_series import TimeSeriesModel


class FourierSeasonality(TimeSeriesModel):
    model_idx: int | None = None

    def __init__(
        self,
        period,
        series_order,
        beta_mean=0,
        beta_sd=10,
        shrinkage_strength=1,
        allow_tune=False,
        tune_method: Literal["simple", "prior_from_idata"] = "simple",
        override_beta_mean_for_tune: bool | np.ndarray = False,
        override_beta_sd_for_tune: bool | np.ndarray = False,
        shift_for_tune: bool = False,
        loss_factor_for_tune: float = 1,
    ):
        self.period = period
        self.series_order = series_order
        self.beta_mean = beta_mean
        self.beta_sd = beta_sd
        self.shrinkage_strength = shrinkage_strength

        self.allow_tune = allow_tune
        self.tune_method = tune_method
        self.override_beta_mean_for_tune = override_beta_mean_for_tune
        self.override_beta_sd_for_tune = override_beta_sd_for_tune
        self.shift_for_tune = shift_for_tune
        self.loss_factor_for_tune = loss_factor_for_tune

    def _fourier_series(self, data: pd.DataFrame, shift=None):
        # convert to days since epoch
        NANOSECONDS_TO_SECONDS = 1000 * 1000 * 1000
        t = (
            data["ds"].to_numpy(dtype=np.int64)
            // NANOSECONDS_TO_SECONDS
            / (3600 * 24.0)
        )
        if shift is not None:
            t += shift

        x_T = t * np.pi * 2
        fourier_components = np.empty((data["ds"].shape[0], 2 * self.series_order))
        if shift is not None and type(shift) is not np.ndarray:
            fourier_components = pt.as_tensor_variable(fourier_components)

        for i in range(self.series_order):
            c = x_T * (i + 1) / self.period
            if type(fourier_components) is np.ndarray:
                fourier_components[:, 2 * i] = np.sin(c)
                fourier_components[:, (2 * i) + 1] = np.cos(c)
            else:
                fourier_components = pt.set_subtensor(
                    fourier_components[:, 2 * i], np.sin(c)
                )
                fourier_components = pt.set_subtensor(
                    fourier_components[:, (2 * i) + 1], np.cos(c)
                )

        return fourier_components

    def _add_beta(self, fit_params: dict, prev_model_idx: int):
        if self.frozen:
            return pm.Deterministic(
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})",
                pt.as_tensor_variable(
                    fit_params["map_approx"][
                        f"fs_{prev_model_idx} - beta(p={self.period},n={self.series_order})"
                    ]
                    if fit_params["map_approx"] is not None
                    else fit_params["trace"]["posterior"][
                        f"fs_{prev_model_idx} - beta(p={self.period},n={self.series_order})"
                    ].mean(dim=["chain", "draw"])
                ),
            )

        return pm.Normal(
            f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})",
            mu=self.beta_mean,
            sigma=self.beta_sd,
            shape=2 * self.series_order,
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
            if fit_params is None:
                raise NotImplementedError(
                    "LinearTrend cannot be frozen before first fit!"
                )

        model_idxs["fs"] = model_idxs.get("fs", 0)
        self.model_idx = model_idxs["fs"]
        model_idxs["fs"] += 1

        x = self._fourier_series(data)

        with model:
            beta = self._add_beta(fit_params, prev_model_idx)

        return pm.math.sum(x * beta, axis=1)

    def _tune(self, model, other_components, data, model_idxs, prev, priors):
        if not self.allow_tune or self.frozen:
            return self.definition(model, other_components, data, model_idxs, prev)

        model_idxs["fs"] = model_idxs.get("fs", 0)
        self.model_idx = model_idxs["fs"]
        model_idxs["fs"] += 1

        with model:
            shift = None
            if self.shift_for_tune:
                shift = pm.Normal(f"fs_{self.model_idx} - shift", mu=0, sigma=1)

            reg_ds = pd.DataFrame(
                {
                    "ds": pd.date_range(
                        "2000-01-01", periods=math.ceil(self.period), freq="D"
                    )
                }
            )
            x = self._fourier_series(data, shift)
            reg_x = self._fourier_series(reg_ds, shift)

            beta_key = (
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            )
            beta_mu_key = f"{beta_key} - beta_mu"
            beta_sd_key = f"{beta_key} - beta_sd"

            if self.override_beta_mean_for_tune is not False:
                prev[beta_mu_key] = self.override_beta_mean_for_tune
            else:
                if beta_mu_key not in prev:
                    prev[beta_mu_key] = (
                        prev["map_approx"][beta_key]
                        if prev["trace"] is None
                        else prev["trace"]["posterior"][beta_key]
                        .to_numpy()
                        .mean(axis=(1, 0))
                    )

            if self.override_beta_sd_for_tune is not False:
                prev[beta_sd_key] = self.override_beta_sd_for_tune
            else:
                if beta_sd_key not in prev:
                    prev[beta_sd_key] = (
                        self.beta_sd
                        if prev["trace"] is None
                        else prev["trace"]["posterior"][beta_key]
                        .to_numpy()
                        .std(axis=(1, 0))
                    ) / self.shrinkage_strength

            beta_mu = pt.as_tensor_variable(prev[beta_mu_key])

            if self.tune_method == "simple":
                beta = pm.Normal(
                    beta_key,
                    mu=beta_mu,
                    sigma=pt.as_tensor_variable(prev[beta_sd_key]),
                    shape=2 * self.series_order,
                )
            elif self.tune_method == "prior_from_idata":
                beta = pm.Deterministic(beta_key, priors[f"prior_{beta_key}"])
            else:
                raise NotImplementedError(
                    f"Tune method {self.tune_method} is not implemented!"
                )

            old = pm.math.sum(reg_x * beta_mu, axis=1)
            new = pm.math.sum(reg_x * beta, axis=1)
            lam = (
                2 * self.period / data.shape[0]
                if self.period > 2 * data.shape[0]
                else 0
            )
            pm.Potential(
                f"{beta_key} - loss",
                self.loss_factor_for_tune
                * lam
                * pm.math.minimum(0, pm.math.dot(old, old) - pm.math.dot(new, new)),
            )

        return pm.math.sum(x * beta, axis=1)

        # if self.tune_method == "offset":
        #     context_beta = pm.Normal(
        #         f"fs_{self.model_idx} - context_beta(p={self.period},n={self.series_order})",
        #         mu=pt.as_tensor_variable(prev[beta_mu_key]),
        #         sigma=pt.as_tensor_variable(prev[beta_sd_key]),
        #         shape=2 * self.series_order,
        #     )
        #     offset_beta = pm.HalfNormal(
        #         f"fs_{self.model_idx} - offset_beta(p={self.period},n={self.series_order})",
        #         sigma=pt.as_tensor_variable(prev[beta_sd_key]),
        #         shape=2 * self.series_order,
        #     )
        #     beta = pm.Deterministic(beta_key, context_beta + offset_beta)

        # if self.tune_method == "linear":
        #     context_beta = pm.Normal(
        #         f"fs_{self.model_idx} - context_beta(p={self.period},n={self.series_order})",
        #         mu=pt.as_tensor_variable(prev[beta_mu_key]),
        #         sigma=pt.as_tensor_variable(prev[beta_sd_key]),
        #         shape=2 * self.series_order,
        #     )
        #     scale_beta = pm.Normal(
        #         f"fs_{self.model_idx} - scale_beta(p={self.period},n={self.series_order})",
        #         mu=0,
        #         sigma=1 / 3,
        #         shape=2 * self.series_order,
        #     )
        #     offset_beta = pm.HalfNormal(
        #         f"fs_{self.model_idx} - offset_beta(p={self.period},n={self.series_order})",
        #         sigma=pt.as_tensor_variable(prev[beta_sd_key]),
        #         shape=2 * self.series_order,
        #     )
        #     beta = pm.Deterministic(
        #         beta_key, context_beta * scale_beta + offset_beta
        #     )

        # if self.tune_method == "same":
        #     beta = pm.Deterministic(
        #         beta_key, pt.as_tensor_variable(prev[beta_mu_key])
        #     )

    def _get_initval(self, initvals, model: pm.Model):
        return {}

    def _det_seasonality_posterior(self, beta, x):
        return np.dot(x, beta.T)

    def _predict_map(self, future, map_approx, other_components):
        future[f"fs_{self.model_idx}"] = self._det_seasonality_posterior(
            map_approx[
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            ],
            self._fourier_series(
                future, map_approx.get(f"fs_{self.model_idx} - shift", None)
            ),
        )

        return future[f"fs_{self.model_idx}"]

    def _predict_mcmc(self, future, trace, other_components):
        shift = trace["posterior"].get(f"fs_{self.model_idx} - shift", None)
        if shift is not None:
            shift = shift.mean()

        future[f"fs_{self.model_idx}"] = self._det_seasonality_posterior(
            trace["posterior"][
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            ]
            .to_numpy()[:, :]
            .mean(0),
            self._fourier_series(future, shift),
        ).T.mean(0)

        return future[f"fs_{self.model_idx}"]

    def _plot(self, plot_params, future, data, scale_params, y_true=None):
        date = future["ds"] if self.period > 7 else future["ds"].dt.day_name()
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(
            f"FourierSeasonality({self.model_idx},p={self.period},n={self.series_order})"
        )
        plt.grid()
        plt.plot(
            date[-int(self.period) :],
            future[f"fs_{self.model_idx}"][-int(self.period) :],
            lw=1,
        )

    def needs_priors(self, *args, **kwargs):
        return self.tune_method == "prior_from_idata"

    def __str__(self):
        return f"FS(p={self.period},n={self.series_order},at={self.allow_tune})"
