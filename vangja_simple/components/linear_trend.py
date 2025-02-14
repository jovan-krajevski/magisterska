import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from vangja_simple.time_series import TimeSeriesModel


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
        allow_tune=False,
        slope_mean_for_tune=None,
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
        self.slope_mean_for_tune = slope_mean_for_tune

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        with model:
            slope = pm.Normal(
                f"lt_{self.model_idx} - slope", self.slope_mean, self.slope_sd
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

            intercept = pm.Normal(
                f"lt_{self.model_idx} - intercept",
                self.intercept_mean,
                self.intercept_sd,
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

            gamma = -self.s * delta
            trend = (slope + pm.math.sum(A * delta, axis=1)) * t + (
                intercept + pm.math.sum(A * gamma, axis=1)
            )

        return trend

    def _tune(self, model, data, initvals, model_idxs, prev):
        if not self.allow_tune:
            return self.definition(model, data, initvals, model_idxs)

        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        with model:
            slope_key = f"lt_{self.model_idx} - slope"
            delta_key = f"lt_{self.model_idx} - delta"
            slope_mu_key = f"{slope_key} - beta_mu"
            slope_sd_key = f"{slope_key} - beta_sd"
            if self.slope_mean_for_tune is not None:
                prev[slope_mu_key] = self.slope_mean_for_tune
            else:
                if slope_mu_key not in prev:
                    prev[slope_mu_key] = (
                        prev["map_approx"][slope_key]
                        + prev["map_approx"][delta_key].sum()
                        if prev["trace"] is None
                        else (
                            prev["trace"]["posterior"][slope_key].to_numpy()
                            + prev["trace"]["posterior"][delta_key]
                            .to_numpy()
                            .sum(axis=2)
                        ).mean()
                    ) / (len(prev["trace"]["observed_data"]["obs"]) / len(data))

            if slope_sd_key not in prev:
                prev[slope_sd_key] = (
                    self.slope_sd
                    if prev["trace"] is None
                    else (
                        (
                            prev["trace"]["posterior"][slope_key].to_numpy()
                            + prev["trace"]["posterior"][delta_key]
                            .to_numpy()
                            .sum(axis=2)
                        )
                        / (len(prev["trace"]["observed_data"]["obs"]) / len(data))
                    ).std()
                )

            slope = pm.Normal(
                f"lt_{self.model_idx} - slope",
                pt.as_tensor_variable(prev[slope_mu_key]),
                pt.as_tensor_variable(prev[slope_sd_key]),
            )

            intercept = pm.Normal(
                f"lt_{self.model_idx} - intercept",
                self.intercept_mean,
                self.intercept_sd,
            )

            t = np.array(data["t"])

        return slope * t + intercept

    def _set_initval(self, initvals, model: pm.Model):
        slope_initval = initvals.get("slope", None)
        intercept_initval = initvals.get("intercept", None)
        delta_initval = initvals.get("delta", None)
        if delta_initval is not None:
            delta_initval = np.array([delta_initval] * self.n_changepoints)

        model.set_initval(
            model.named_vars[f"lt_{self.model_idx} - slope"], slope_initval
        )
        model.set_initval(
            model.named_vars[f"lt_{self.model_idx} - intercept"], intercept_initval
        )
        if f"lt_{self.model_idx} - delta" in model.named_vars:
            model.set_initval(
                model.named_vars[f"lt_{self.model_idx} - delta"], delta_initval
            )

    def _predict_map(self, future, map_approx):
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

    def _predict_mcmc(self, future, trace):
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

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"LinearTrend({self.model_idx})")
        plt.grid()

        plt.plot(future["ds"], future[f"lt_{self.model_idx}"], lw=1)

        plt.legend()

    def __str__(self):
        return f"LT(n={self.n_changepoints},r={self.changepoint_range},at={self.allow_tune})"
