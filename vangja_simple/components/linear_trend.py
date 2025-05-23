from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from vangja_simple.time_series import TimeSeriesModel
from vangja.utils import get_group_definition


class LinearTrend(TimeSeriesModel):
    model_idx: int | None = None

    def __init__(
        self,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        slope_mean: float = 0,
        slope_sd: float = 5,
        intercept_mean: float = 0,
        intercept_sd: float = 5,
        delta_mean: float = 0,
        delta_sd: None | float = 0.05,
        allow_tune: bool = False,
        tune_method: Literal["simple", "prior_from_idata"] = "simple",
        override_slope_mean_for_tune: bool | np.ndarray = False,
        override_slope_sd_for_tune: bool | np.ndarray = False,
        loss_factor_for_tune: float = 0,
    ):
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.slope_mean = slope_mean
        self.slope_sd = slope_sd
        self.intercept_mean = intercept_mean
        self.intercept_sd = intercept_sd
        self.delta_mean = delta_mean
        self.delta_sd = delta_sd

        self.allow_tune = allow_tune
        self.tune_method = tune_method
        self.override_slope_mean_for_tune = override_slope_mean_for_tune
        self.override_slope_sd_for_tune = override_slope_sd_for_tune
        self.loss_factor_for_tune = loss_factor_for_tune

    def _add_slope(self, fit_params: dict, prev_model_idx: int):
        if self.frozen:
            return pm.Deterministic(
                f"lt_{self.model_idx} - slope",
                pt.as_tensor_variable(
                    fit_params["map_approx"][f"lt_{prev_model_idx} - slope"]
                    if fit_params["map_approx"] is not None
                    else fit_params["trace"]["posterior"][
                        f"lt_{prev_model_idx} - slope"
                    ].mean()
                ),
            )

        return pm.Normal(f"lt_{self.model_idx} - slope", self.slope_mean, self.slope_sd)

    def _add_intercept(self, fit_params: dict, prev_model_idx: int):
        if self.frozen:
            return pm.Deterministic(
                f"lt_{self.model_idx} - intercept",
                pt.as_tensor_variable(
                    fit_params["map_approx"][f"lt_{prev_model_idx} - intercept"]
                    if fit_params["map_approx"] is not None
                    else fit_params["trace"]["posterior"][
                        f"lt_{prev_model_idx} - intercept"
                    ].mean()
                ),
            )

        return pm.Normal(
            f"lt_{self.model_idx} - intercept", self.intercept_mean, self.intercept_sd
        )

    def _add_delta(self, fit_params: dict, prev_model_idx: int):
        if self.frozen:
            return pm.Deterministic(
                f"lt_{self.model_idx} - delta",
                pt.as_tensor_variable(
                    fit_params["map_approx"][f"lt_{prev_model_idx} - delta"]
                    if fit_params["map_approx"] is not None
                    else (
                        fit_params["trace"]["posterior"][
                            f"lt_{prev_model_idx} - delta"
                        ].mean(dim=["chain", "draw"])
                    )
                ),
            )

        delta_sd = self.delta_sd
        if self.delta_sd is None:
            delta_sd = pm.Exponential(f"lt_{self.model_idx} - tau", 1.5)

        return pm.Laplace(
            f"lt_{self.model_idx} - delta",
            self.delta_mean,
            delta_sd,
            shape=self.n_changepoints,
        )

    def hierarchical_definition(
        self,
        model: TimeSeriesModel,
        other_components: dict,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        fit_params: dict | None,
    ):
        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        self.group, self.n_groups, self.groups_ = get_group_definition(
            data, "series", "partial"
        )

        with model:
            large_slope = pm.Normal(
                f"lt_{self.model_idx} - large_slope", self.slope_mean, self.slope_sd
            )

            small_slope = pm.Normal(
                f"lt_{self.model_idx} - slope",
                self.slope_mean,
                self.slope_sd,
                shape=self.n_groups - 1,
            )

            delta_sd = self.delta_sd
            if self.delta_sd is None:
                delta_sd = pm.Exponential(f"lt_{self.model_idx} - tau", 1.5)

            delta = pm.Laplace(
                f"lt_{self.model_idx} - delta",
                self.delta_mean,
                delta_sd,
                shape=self.n_changepoints,
            )

            large_intercept = pm.Normal(
                f"lt_{self.model_idx} - large_intercept",
                self.intercept_mean,
                self.intercept_sd,
            )

            small_intercept = pm.Normal(
                f"lt_{self.model_idx} - intercept",
                self.intercept_mean,
                self.intercept_sd,
                shape=self.n_groups - 1,
            )

            # only first series gets changepoints
            large_series = data[data["series"] == data["series"].iloc[0]]
            large_t = np.array(large_series["t"])
            hist_size = int(np.floor(large_series.shape[0] * self.changepoint_range))
            cp_indexes = (
                np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                .round()
                .astype(int)
            )
            self.s = np.array(large_series.iloc[cp_indexes]["t"].tail(-1))
            A = (large_t[:, None] > self.s) * 1

            gamma = -self.s * delta

            # breakpoint()

            large_trend = (large_slope + pm.math.sum(A * delta, axis=1)) * large_t + (
                large_intercept + pm.math.sum(A * gamma, axis=1)
            )

            # other series have simple linear trend
            small_series = data[data["series"] != data["series"].iloc[0]]
            small_t = np.array(small_series["t"])
            small_group = self.group[data["series"] != data["series"].iloc[0]]
            small_trend = (
                small_slope[small_group] * small_t + small_intercept[small_group]
            )

        return pm.math.concatenate([large_trend, small_trend])

    def definition(
        self,
        model: TimeSeriesModel,
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

        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        with model:
            t = np.array(data["t"])
            slope = self._add_slope(fit_params, prev_model_idx)
            intercept = self._add_intercept(fit_params, prev_model_idx)

            if self.n_changepoints > 0:
                delta = self._add_delta(fit_params, prev_model_idx)

                hist_size = int(np.floor(data.shape[0] * self.changepoint_range))
                cp_indexes = (
                    np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                    .round()
                    .astype(int)
                )
                self.s = np.array(data.iloc[cp_indexes]["t"].tail(-1))
                A = (t[:, None] > self.s) * 1

                gamma = -self.s * delta
                trend = (slope + pm.math.sum(A * delta, axis=1)) * t + (
                    intercept + pm.math.sum(A * gamma, axis=1)
                )
            else:
                trend = slope * t + intercept

        return trend

    def _tune(
        self,
        model: pm.Model,
        other_components: dict,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        prev: dict,
        priors,
    ):
        if not self.allow_tune or self.frozen:
            return self.definition(model, other_components, data, model_idxs, prev)

        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        with model:
            slope_key = f"lt_{self.model_idx} - slope"
            delta_key = f"lt_{self.model_idx} - delta"
            slope_mu_key = f"{slope_key} - beta_mu"
            slope_sd_key = f"{slope_key} - beta_sd"
            prev_delta_key = f"lt_{self.model_idx} - prev_delta"

            if prev_delta_key not in prev:
                prev[prev_delta_key] = 0
                if prev["trace"] is None:
                    prev[prev_delta_key] = (
                        prev["map_approx"][delta_key].sum()
                        if delta_key in prev["map_approx"]
                        else 0
                    )
                else:
                    prev[prev_delta_key] = (
                        (prev["trace"]["posterior"][delta_key].to_numpy().sum(axis=2))
                        if delta_key in prev["trace"]["posterior"]
                        else 0
                    )

            if self.override_slope_mean_for_tune is not False:
                prev[slope_mu_key] = self.override_slope_mean_for_tune
            else:
                if slope_mu_key not in prev:
                    prev[slope_mu_key] = (
                        prev["map_approx"][slope_key] + prev[prev_delta_key]
                        if prev["trace"] is None
                        else (
                            prev["trace"]["posterior"][slope_key].to_numpy()
                            + prev[prev_delta_key]
                        ).mean()
                    )

            if self.override_slope_sd_for_tune is not False:
                prev[slope_sd_key] = self.override_slope_sd_for_tune
            else:
                if slope_sd_key not in prev:
                    prev[slope_sd_key] = (
                        self.slope_sd
                        if prev["trace"] is None
                        else (
                            prev["trace"]["posterior"][slope_key].to_numpy()
                            + prev[prev_delta_key]
                        ).std()
                    )

            if self.tune_method == "simple":
                slope = pm.Normal(
                    slope_key,
                    pt.as_tensor_variable(prev[slope_mu_key]),
                    pt.as_tensor_variable(prev[slope_sd_key]),
                )
            elif self.tune_method == "prior_from_idata":
                slope = pm.Deterministic(slope_key, priors[f"prior_{slope_key}"])
            else:
                raise NotImplementedError(
                    f"Tune method {self.tune_method} is not implemented!"
                )

            intercept = pm.Normal(
                f"lt_{self.model_idx} - intercept",
                self.intercept_mean,
                self.intercept_sd,
            )

            pm.Potential(
                f"{slope_key} - loss",
                self.loss_factor_for_tune * pm.math.abs(slope - prev[slope_mu_key]),
            )

            t = np.array(data["t"])

        return slope * t + intercept

    def _get_initval(self, initvals, model: pm.Model):
        return {
            model.named_vars[f"lt_{self.model_idx} - slope"]: initvals.get(
                "slope", None
            ),
            model.named_vars[f"lt_{self.model_idx} - intercept"]: initvals.get(
                "intercept", None
            ),
        }

    def _predict_map(
        self, future, map_approx, other_components, hierarchical_model=False
    ):
        if hierarchical_model:
            forecasts = []
            for group_code in self.groups_.keys():
                if group_code == self.group[0]:
                    new_A = (np.array(future["t"])[:, None] > self.s) * 1
                    forecasts.append(
                        np.array(
                            (
                                map_approx[f"lt_{self.model_idx} - large_slope"]
                                + np.dot(
                                    new_A, map_approx[f"lt_{self.model_idx} - delta"]
                                )
                            )
                            * future["t"]
                            + (
                                map_approx[f"lt_{self.model_idx} - large_intercept"]
                                + np.dot(
                                    new_A,
                                    (
                                        -self.s
                                        * map_approx[f"lt_{self.model_idx} - delta"]
                                    ),
                                )
                            )
                        )
                    )
                else:
                    forecasts.append(
                        map_approx[f"lt_{self.model_idx} - slope"][group_code]
                        * future["t"]
                        + map_approx[f"lt_{self.model_idx} - intercept"][group_code]
                    )

                future[f"lt_{self.model_idx}_{group_code}"] = forecasts[-1]

            return np.vstack(forecasts)

        if f"lt_{self.model_idx} - delta" not in map_approx:
            future[f"lt_{self.model_idx}"] = np.array(
                map_approx[f"lt_{self.model_idx} - slope"] * future["t"]
                + map_approx[f"lt_{self.model_idx} - intercept"]
            )
        else:
            new_A = (np.array(future["t"])[:, None] > self.s) * 1

            future[f"lt_{self.model_idx}"] = np.array(
                (
                    map_approx[f"lt_{self.model_idx} - slope"]
                    + np.dot(new_A, map_approx[f"lt_{self.model_idx} - delta"])
                )
                * future["t"]
                + (
                    map_approx[f"lt_{self.model_idx} - intercept"]
                    + np.dot(
                        new_A, (-self.s * map_approx[f"lt_{self.model_idx} - delta"])
                    )
                )
            )

        return future[f"lt_{self.model_idx}"]

    def _predict_mcmc(self, future, trace, other_components):
        slope = (
            trace["posterior"][f"lt_{self.model_idx} - slope"].to_numpy()[:, :].mean(0)
        )
        intercept = (
            trace["posterior"][f"lt_{self.model_idx} - intercept"]
            .to_numpy()[:, :]
            .mean(0)
        )

        if f"lt_{self.model_idx} - delta" not in trace["posterior"]:
            future[f"lt_{self.model_idx}"] = (
                slope.mean() * future["t"].to_numpy() + intercept.mean()
            )
        else:
            new_A = (np.array(future["t"])[:, None] > self.s) * 1
            delta = (
                trace["posterior"][f"lt_{self.model_idx} - delta"]
                .to_numpy()[:, :]
                .mean(0)
            )

            future[f"lt_{self.model_idx}"] = (
                (slope + np.dot(new_A, delta.T)).T * future["t"].to_numpy()
                + (intercept + np.dot(new_A, (-self.s * delta).T)).T
            ).mean(0)

        return future[f"lt_{self.model_idx}"]

    def _plot(self, plot_params, future, data, scale_params, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"LinearTrend({self.model_idx})")
        plt.grid()

        plt.plot(future["ds"], future[f"lt_{self.model_idx}"], lw=1)

        plt.legend()

    def needs_priors(self, *args, **kwargs):
        return self.tune_method == "prior_from_idata"

    def __str__(self):
        return f"LT(n={self.n_changepoints},r={self.changepoint_range},at={self.allow_tune})"
