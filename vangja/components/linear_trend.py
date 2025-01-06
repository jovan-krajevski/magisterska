import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from vangja.time_series import TimeSeriesModel
from vangja.utils import get_group_definition


class LinearTrend(TimeSeriesModel):
    def __init__(
        self,
        n_changepoints=25,
        changepoint_range=0.8,
        slope_mean=0,
        slope_sd=5,
        intercept_mean=0,
        intercept_sd=5,
        delta_mean=0,
        delta_sd=0.05,
        pool_cols=None,
        pool_type="complete",
        allow_tune=False,
    ):
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.slope_mean = slope_mean
        self.slope_sd = slope_sd
        self.intercept_mean = intercept_mean
        self.intercept_sd = intercept_sd
        self.delta_mean = delta_mean
        self.delta_sd = delta_sd

        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.allow_tune = allow_tune

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        self.group, self.n_groups, self.groups_ = get_group_definition(
            data, self.pool_cols, self.pool_type
        )

        slope_initval = initvals.get("slope", None)
        if slope_initval is not None:
            slope_initval = np.array([slope_initval] * self.n_groups)

        delta_initval = initvals.get("delta", None)
        if delta_initval is not None:
            delta_initval = np.array(
                [[delta_initval] * self.n_changepoints] * self.n_groups
            )

        intercept_initval = initvals.get("intercept", None)
        if intercept_initval is not None:
            intercept_initval = np.array([intercept_initval] * self.n_groups)

        with model:
            if self.pool_type == "partial":
                sigma_slope = pm.HalfCauchy(
                    f"lt_{self.model_idx} - sigma_slope", beta=self.slope_sd
                )
                offset_slope = pm.Normal(
                    f"lt_{self.model_idx} - offset_slope",
                    mu=0,
                    sigma=1,
                    shape=self.n_groups,
                )
                slope = pm.Deterministic(
                    f"lt_{self.model_idx} - slope", offset_slope * sigma_slope
                )

                delta_sd = self.delta_sd
                if self.delta_sd is None:
                    delta_sd = pm.Exponential(f"lt_{self.model_idx} - tau", 1.5)

                sigma_delta = pm.HalfCauchy(
                    f"lt_{self.model_idx} - sigma_delta", beta=delta_sd
                )
                offset_delta = pm.Laplace(
                    f"lt_{self.model_idx} - offset_delta",
                    0,
                    1,
                    shape=(self.n_groups, self.n_changepoints),
                )
                delta = pm.Deterministic(
                    f"lt_{self.model_idx} - delta", offset_delta * sigma_delta
                )
            else:
                slope = pm.Normal(
                    f"lt_{self.model_idx} - slope",
                    self.slope_mean,
                    self.slope_sd,
                    initval=slope_initval,
                    shape=self.n_groups,
                )

                delta_sd = self.delta_sd
                if self.delta_sd is None:
                    delta_sd = pm.Exponential(f"lt_{self.model_idx} - tau", 1.5)

                delta = pm.Laplace(
                    f"lt_{self.model_idx} - delta",
                    self.delta_mean,
                    delta_sd,
                    initval=delta_initval,
                    shape=(self.n_groups, self.n_changepoints),
                )

            intercept = pm.Normal(
                f"lt_{self.model_idx} - intercept",
                self.intercept_mean,
                self.intercept_sd,
                initval=intercept_initval,
                shape=self.n_groups,
            )

            if self.pool_type == "individual":
                ss = []
                t = np.array(data["t"])
                for group_code in range(self.n_groups):
                    series_data = data[self.group == group_code]
                    hist_size = int(
                        np.floor(series_data.shape[0] * self.changepoint_range)
                    )
                    cp_indexes = (
                        np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                        .round()
                        .astype(int)
                    )
                    ss.append(np.array(series_data.iloc[cp_indexes]["t"].tail(-1)))

                self.s = np.stack(ss, axis=0)
                A = (t[:, None] > self.s[self.group]) * 1

                gamma = -self.s[self.group, :] * delta[self.group, :]
                trend = pm.Deterministic(
                    f"lt_{self.model_idx} - trend",
                    (slope[self.group] + pm.math.sum(A * delta[self.group], axis=1)) * t
                    + (intercept[self.group] + pm.math.sum(A * gamma, axis=1)),
                )
            else:
                t = np.array(data["t"])
                hist_size = int(np.floor(data.shape[0] * self.changepoint_range))
                cp_indexes = (
                    np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                    .round()
                    .astype(int)
                )
                self.s = np.array(data.iloc[cp_indexes]["t"].tail(-1))
                A = (t[:, None] > self.s) * 1

                gamma = -self.s * delta[self.group, :]
                trend = pm.Deterministic(
                    f"lt_{self.model_idx} - trend",
                    (slope[self.group] + pm.math.sum(A * delta[self.group], axis=1)) * t
                    + (intercept[self.group] + pm.math.sum(A * gamma, axis=1)),
                )

        return trend

    def _tune(self, model, data, initvals, model_idxs, prev):
        if not self.allow_tune:
            return self.definition(model, data, initvals, model_idxs)

        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        self.group, self.n_groups, self.groups_ = get_group_definition(
            data, self.pool_cols, self.pool_type
        )

        slope_mu, intercept_mu = self._get_slope_mus_and_intercept_mus(data, prev)

        slope_initval = initvals.get("slope", None)
        if slope_initval is not None:
            slope_initval = np.array([slope_initval] * self.n_groups)

        delta_initval = initvals.get("delta", None)
        if delta_initval is not None:
            delta_initval = np.array(
                [[delta_initval] * self.n_changepoints] * self.n_groups
            )

        intercept_initval = initvals.get("intercept", None)
        if intercept_initval is not None:
            intercept_initval = np.array([intercept_initval] * self.n_groups)

        with model:
            slope = pm.Normal(
                f"lt_{self.model_idx} - slope",
                slope_mu,
                self.slope_sd,
                initval=slope_initval,
                shape=self.n_groups,
            )

            delta_sd = self.delta_sd
            if self.delta_sd is None:
                delta_sd = pm.Exponential(f"lt_{self.model_idx} - tau", 1.5)

            delta = pm.Laplace(
                f"lt_{self.model_idx} - delta",
                self.delta_mean,
                delta_sd,
                initval=delta_initval,
                shape=(self.n_groups, self.n_changepoints),
            )

            intercept = pm.Normal(
                f"lt_{self.model_idx} - intercept",
                intercept_mu,
                self.intercept_sd,
                initval=intercept_initval,
                shape=self.n_groups,
            )

            t = np.array(data["t"])
            hist_size = int(np.floor(data.shape[0] * self.changepoint_range))
            cp_indexes = (
                np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                .round()
                .astype(int)
            )
            self.s = np.array(data.iloc[cp_indexes]["t"].tail(-1))
            A = (t[:, None] > self.s) * 1

            gamma = -self.s * delta[self.group, :]
            trend = pm.Deterministic(
                f"lt_{self.model_idx} - trend",
                (slope[self.group] + pm.math.sum(A * delta[self.group], axis=1)) * t
                + (intercept[self.group] + pm.math.sum(A * gamma, axis=1)),
            )

        return trend

    def _get_slope_mus_and_intercept_mus(self, data, prev):
        if self.pool_type != "individual":
            new_A = (np.array(data["prev_t"][:1])[:, None] > self.s) * 1

        slope_mus = []
        intercept_mus = []
        for group_code in self.groups_.keys():
            if self.pool_type == "individual":
                s = self.s[group_code]
                new_A = (np.array(data["prev_t"][:1])[:, None] > self.s[group_code]) * 1
            else:
                s = self.s

            slope_mus.append(
                (
                    prev["map_approx"][f"lt_{self.model_idx} - slope"][group_code]
                    + np.dot(
                        new_A,
                        prev["map_approx"][f"lt_{self.model_idx} - delta"][group_code],
                    )
                )[0]
            )
            intercept_mus.append(
                (
                    prev["map_approx"][f"lt_{self.model_idx} - intercept"][group_code]
                    + np.dot(
                        new_A,
                        (
                            -s
                            * prev["map_approx"][f"lt_{self.model_idx} - delta"][
                                group_code
                            ]
                        ),
                    )
                )[0]
            )

        return slope_mus, intercept_mus

    def _predict_map(self, future, map_approx):
        forecasts = []
        if self.pool_type != "individual":
            new_A = (np.array(future["t"])[:, None] > self.s) * 1

        for group_code in self.groups_.keys():
            if self.pool_type == "individual":
                s = self.s[group_code]
                new_A = (np.array(future["t"])[:, None] > self.s[group_code]) * 1
            else:
                s = self.s

            forecasts.append(
                np.array(
                    (
                        map_approx[f"lt_{self.model_idx} - slope"][group_code]
                        + np.dot(
                            new_A,
                            map_approx[f"lt_{self.model_idx} - delta"][group_code],
                        )
                    )
                    * future["t"]
                    + (
                        map_approx[f"lt_{self.model_idx} - intercept"][group_code]
                        + np.dot(
                            new_A,
                            (
                                -s
                                * map_approx[f"lt_{self.model_idx} - delta"][group_code]
                            ),
                        )
                    )
                )
            )
            future[f"lt_{self.model_idx}_{group_code}"] = forecasts[-1]

        return np.vstack(forecasts)

    def _predict_mcmc(self, future, trace):
        forecasts = []
        if self.pool_type == "individual":
            new_A = (np.array(future["t"])[:, None] > self.s[self.group]) * 1
        else:
            new_A = (np.array(future["t"])[:, None] > self.s) * 1

        for group_code in self.groups_.keys():
            delta = (
                trace["posterior"][f"lt_{self.model_idx} - delta"]
                .to_numpy()[:, :, group_code]
                .mean(0)
            )
            slope = (
                trace["posterior"][f"lt_{self.model_idx} - slope"]
                .to_numpy()[:, :, group_code]
                .mean(0)
            )
            intercept = (
                trace["posterior"][f"lt_{self.model_idx} - intercept"]
                .to_numpy()[:, :, group_code]
                .mean(0)
            )

            forecasts.append(
                (
                    (slope + np.dot(new_A, delta.T)).T * future["t"].to_numpy()
                    + (intercept + np.dot(new_A, (-self.s * delta).T)).T
                ).mean(0)
            )
            future[f"lt_{self.model_idx}_{group_code}"] = forecasts[-1]

        return np.vstack(forecasts)

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"lt_{self.model_idx}")
        plt.grid()

        for group_code, group_name in self.groups_.items():
            plt.plot(
                future["ds"],
                future[f"lt_{self.model_idx}_{group_code}"],
                lw=1,
                label=group_name,
            )

        plt.legend()

    def __str__(self):
        return f"LT(n={self.n_changepoints},r={self.changepoint_range},at={self.allow_tune},{self.pool_type})"
