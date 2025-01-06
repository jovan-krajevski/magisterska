import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from vangja.time_series import TimeSeriesModel
from vangja.utils import get_group_definition


class FourierSeasonality(TimeSeriesModel):
    def __init__(
        self,
        period,
        series_order,
        beta_mean=0,
        beta_sd=10,
        shrinkage_strength=100,
        pool_cols=None,
        pool_type="complete",
        allow_tune=False,
    ):
        self.period = period
        self.series_order = series_order
        self.beta_mean = beta_mean
        self.beta_sd = beta_sd
        self.shrinkage_strength = shrinkage_strength

        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.allow_tune = allow_tune

    def _fourier_series(self, data):
        # convert to days since epoch
        NANOSECONDS_TO_SECONDS = 1000 * 1000 * 1000
        t = (
            data["ds"].to_numpy(dtype=np.int64)
            // NANOSECONDS_TO_SECONDS
            / (3600 * 24.0)
        )

        x_T = t * np.pi * 2
        fourier_components = np.empty((data["ds"].shape[0], 2 * self.series_order))
        for i in range(self.series_order):
            c = x_T * (i + 1) / self.period
            fourier_components[:, 2 * i] = np.sin(c)
            fourier_components[:, (2 * i) + 1] = np.cos(c)

        return fourier_components

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["fs"] = model_idxs.get("fs", 0)
        self.model_idx = model_idxs["fs"]
        model_idxs["fs"] += 1

        group, n_groups, self.groups_ = get_group_definition(
            data, self.pool_cols, self.pool_type
        )

        x = self._fourier_series(data)
        beta_initval = initvals.get("beta", None)
        if beta_initval is not None:
            if self.pool_type == "partial":
                beta_initval = np.array([beta_initval] * 2 * self.series_order)
            else:
                beta_initval = np.array(
                    [[beta_initval] * 2 * self.series_order] * n_groups
                )

        with model:
            if self.pool_type == "partial":
                mu_beta = pm.Normal(
                    f"fs_{self.model_idx} - beta_mu(p={self.period},n={self.series_order})",
                    mu=self.beta_mean,
                    sigma=self.beta_sd,
                    shape=2 * self.series_order,
                    initval=beta_initval,
                )
                sigma_beta = pm.HalfNormal(
                    f"fs_{self.model_idx} - beta_sigma(p={self.period},n={self.series_order})",
                    sigma=self.beta_sd / self.shrinkage_strength,
                    shape=2 * self.series_order,
                )
                offset_beta = pm.Normal(
                    f"fs_{self.model_idx} - offset_beta(p={self.period},n={self.series_order})",
                    mu=0,
                    sigma=1,
                    shape=(n_groups, 2 * self.series_order),
                )

                beta = pm.Deterministic(
                    f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})",
                    mu_beta + offset_beta * sigma_beta,
                )
            else:
                beta = pm.Normal(
                    f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})",
                    mu=self.beta_mean,
                    sigma=self.beta_sd,
                    shape=(n_groups, 2 * self.series_order),
                    initval=beta_initval,
                )

        return pm.math.sum(x * beta[group], axis=1)

    def _tune(self, model, data, initvals, model_idxs, prev):
        if not self.allow_tune:
            return self.definition(model, data, initvals, model_idxs)

        model_idxs["fs"] = model_idxs.get("fs", 0)
        self.model_idx = model_idxs["fs"]
        model_idxs["fs"] += 1

        group, n_groups, self.groups_ = get_group_definition(
            data, self.pool_cols, self.pool_type
        )

        x = self._fourier_series(data)
        beta_initval = initvals.get("beta", None)
        if beta_initval is not None:
            beta_initval = np.array([[beta_initval] * 2 * self.series_order] * n_groups)

        with model:
            beta_key = (
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            )
            beta_mu = (
                prev["map_approx"][beta_key]
                if prev["trace"] is None
                else prev["trace"]["posterior"][beta_key].to_numpy().mean(axis=(1, 0))
            )
            beta_sd = (
                self.beta_sd
                if prev["trace"] is None
                else prev["trace"]["posterior"][beta_key].to_numpy().std(axis=(1, 0))
            )
            beta = pm.Normal(
                beta_key,
                mu=pt.as_tensor_variable(beta_mu),
                sigma=pt.as_tensor_variable(beta_sd),
                shape=(n_groups, 2 * self.series_order),
                initval=beta_initval,
            )
            # sigma_beta = pm.Normal(
            #     f"fs_{self.model_idx} - beta_sigma(p={self.period},n={self.series_order})",
            #     mu=0,
            #     sigma=self.beta_sd,
            #     shape=2 * self.series_order,
            # )
            # beta = pm.Deterministic(
            #     f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})",
            #     pt.as_tensor_variable(
            #         prev[
            #             f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            #         ]
            #     ),
            # )
            # sigma_beta = pm.HalfNormal(
            #     f"fs_{self.model_idx} - beta_sigma(p={self.period},n={self.series_order})",
            #     sigma=self.beta_sd,
            #     shape=2 * self.series_order,
            # )
            # offset_beta = pm.Normal(
            #     f"fs_{self.model_idx} - offset_beta(p={self.period},n={self.series_order})",
            #     mu=0,
            #     sigma=0.01,
            #     shape=(n_groups, 2 * self.series_order),
            # )

            # beta = pm.Deterministic(
            #     f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})",
            #     prev[
            #         f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            #     ]
            #     + offset_beta,
            # )

        # beta = prev[f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"]

        return pm.math.sum(x * beta[group], axis=1)

    def _det_seasonality_posterior(self, beta, x):
        return np.dot(x, beta.T)

    def _predict_map(self, future, map_approx):
        forecasts = []
        for group_code in self.groups_.keys():
            forecasts.append(
                self._det_seasonality_posterior(
                    map_approx[
                        f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
                    ][group_code],
                    self._fourier_series(future),
                )
            )
            future[f"fs_{self.model_idx}_{group_code}"] = forecasts[-1]

        return np.vstack(forecasts)

    def _predict_mcmc(self, future, trace):
        forecasts = []
        for group_code in self.groups_.keys():
            forecasts.append(
                self._det_seasonality_posterior(
                    trace["posterior"][
                        f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
                    ]
                    .to_numpy()[:, :, group_code]
                    .mean(0),
                    self._fourier_series(future),
                ).T.mean(0)
            )
            future[f"fs_{self.model_idx}_{group_code}"] = forecasts[-1]

        return np.vstack(forecasts)

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        date = future["ds"] if self.period > 7 else future["ds"].dt.day_name()
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"fs_{self.model_idx} - p={self.period},n={self.series_order}")
        plt.grid()

        for group_code, group_name in self.groups_.items():
            plt.plot(
                date[-int(self.period) :],
                future[f"fs_{self.model_idx}_{group_code}"][-int(self.period) :],
                lw=1,
                label=group_name,
            )

        plt.legend()

    def __str__(self):
        return f"FS(p={self.period},n={self.series_order},at={self.allow_tune},{self.pool_type})"
