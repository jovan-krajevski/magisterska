import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from vangja_hierarchical.time_series import TimeSeriesModel
from vangja_hierarchical.types import PoolType, TuneMethod
from vangja_hierarchical.utils import get_group_definition


class LinearTrend(TimeSeriesModel):
    def __init__(
        self,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        slope_mean: float = 0,
        slope_sd: float = 5,
        intercept_mean: float = 0,
        intercept_sd: float = 5,
        delta_mean: float = 0,
        delta_sd: float = 0.05,
        pool_type: PoolType = "partial",
        tune_method: TuneMethod | None = "parametric",
        override_slope_mean_for_tune: np.ndarray | None = None,
        override_slope_sd_for_tune: np.ndarray | None = None,
        shrinkage_strength: float = 100,
        # loss_factor_for_tune: float = 0,
    ):
        """
        Crate a Linear Trend model component.

        Parameters
        ----------
        n_changepoints: int
            The number of points at which the linear trend changes its slope.
        changepoint_range: float
            The portion of the time axis at which the potential changes in the slope
            are allowed.
        slope_mean: float
            The mean of the Normal prior for the slope parameter.
        slope_sd: float
            The standard deviation of the Normal prior for the slope parameter.
        intercept_mean: float
            The mean of the Normal prior for the intercept parameter.
        intercept_sd: float
            The standard devation of the Normal prior for the intercept parameter.
        delta_mean: float
            The mean of the Laplace prior for the slope change in the potential
            changepoints.
        delta_sd: float | None
            The standard deviation of the Laplace prior for the slope change in the
            potential changepoints. If delta_sd is None, the standard deviation becomes
            a random variable with a Exponential(lam=1.5) prior.
        pool_type: PoolType
            Type of pooling performed when sampling.
        tune_method: TuneMethod | None
            How the transfer learning is to be performed. One of "parametric" or
            "prior_from_idata". If set to None, this component will not be tuned even if
            idata is provided.
        override_slope_mean_for_tune: np.ndarray | None
            Override the mean of the Normal prior for the slope parameter with this
            value.
        override_slope_sd_for_tune: np.ndarray | None
            Override the standard deviation of the Normal prior for the slope parameter
            with this value.
        shrinkage_strength: float
            Shrinkage between groups for the hierarchical modeling.
        """
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.slope_mean = slope_mean
        self.slope_sd = slope_sd
        self.intercept_mean = intercept_mean
        self.intercept_sd = intercept_sd
        self.delta_mean = delta_mean
        self.delta_sd = delta_sd
        self.pool_type = pool_type

        self.tune_method = tune_method
        self.override_slope_mean_for_tune = override_slope_mean_for_tune
        self.override_slope_sd_for_tune = override_slope_sd_for_tune
        self.shrinkage_strength = shrinkage_strength

    def _get_slope_params_from_idata(self, idata: az.InferenceData):
        """
        Calculate the mean and the standard deviation of the Normal prior for the slope
        parameter from a provided posterior sample.

        Parameters
        ----------
        idata: az.InferenceData
            Sample from a posterior.
        """

        slope_key = f"lt_{self.model_idx} - slope"
        delta_key = f"lt_{self.model_idx} - delta"

        delta = (
            (idata["posterior"][delta_key].to_numpy().sum(axis=2))
            if delta_key in idata["posterior"]
            else 0
        )

        if self.override_slope_mean_for_tune is not None:
            slope_mean = self.override_slope_mean_for_tune
        else:
            slope_mean = (idata["posterior"][slope_key].to_numpy() + delta * 0).mean()

        if self.override_slope_sd_for_tune is not None:
            slope_sd = self.override_slope_sd_for_tune
        else:
            slope_sd = (idata["posterior"][slope_key].to_numpy() + delta * 0).std()

        return slope_mean, slope_sd

    def _get_skipped_deltas(self, data: pd.DataFrame, idata: az.InferenceData):
        """
        Calculate the sum of the skipped change point deltas for each time series.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        idata: az.InferenceData
            Sample from a posterior.
        """
        delta_key = f"lt_{self.model_idx} - delta"
        skipped_deltas = []

        for _, group_name in sorted(self.groups_.items()):
            series = data[data["series"] == group_name]
            delta = 0
            if delta_key in idata["posterior"]:
                # cp_before_min_t = (series["t"].min() > self.s).sum()
                delta = (
                    idata["posterior"][delta_key]
                    .to_numpy()
                    # .to_numpy()[:, :, :cp_before_min_t]
                    .sum(axis=2)
                    .mean()
                )

            skipped_deltas.append(delta)

        return np.array(skipped_deltas)

    def _complete_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ):
        """
        Add the LinearTrend parameters to the model when pool_type is complete.

        Parameters
        ----------
        model: TimeSeriesModel
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        priors: dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating the
            posterior sample in idata.
        idata: az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to set the
            parameters' priors in the model. If idata is not None, each component from
            the model should specify how idata should be used to set its parameters'
            priors.
        """
        with model:
            t = np.array(data["t"])
            slope_key = f"lt_{self.model_idx} - slope"
            skipped_deltas = 0

            if idata is not None and self.tune_method == "parametric":
                slope_mean, slope_sd = self._get_slope_params_from_idata(idata)
                skipped_deltas = self._get_skipped_deltas(data, idata)[0]
                slope = pm.Normal(slope_key, slope_mean, slope_sd) + skipped_deltas
            elif priors is not None and self.tune_method == "prior_from_idata":
                slope = pm.Deterministic(slope_key, priors[f"prior_{slope_key}"])
            else:
                slope = pm.Normal(slope_key, self.slope_mean, self.slope_sd)

            intercept = pm.Normal(
                f"lt_{self.model_idx} - intercept",
                self.intercept_mean,
                self.intercept_sd,
            )

            if self.n_changepoints > 0:
                delta_sd = self.delta_sd
                if self.delta_sd is None:
                    delta_sd = pm.Exponential(f"lt_{self.model_idx} - tau", 1.5)

                delta = pm.Laplace(
                    f"lt_{self.model_idx} - delta",
                    self.delta_mean,
                    delta_sd,
                    shape=self.n_changepoints,
                )
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

    def _partial_definition(
        self,
        model: TimeSeriesModel,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ):
        """
        Add the LinearTrend parameters to the model when pool_type is partial.

        Parameters
        ----------
        model: TimeSeriesModel
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        priors: dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating the
            posterior sample in idata.
        idata: az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to set the
            parameters' priors in the model. If idata is not None, each component from
            the model should specify how idata should be used to set its parameters'
            priors.
        """
        with model:
            slope_key = f"lt_{self.model_idx} - slope"
            t = np.array(data["t"])

            # calculate change points on first time series
            large_series = data[data["series"] == data["series"].iloc[0]]
            hist_size = int(np.floor(large_series.shape[0] * self.changepoint_range))
            cp_indexes = (
                np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                .round()
                .astype(int)
            )
            self.s = np.array(large_series.iloc[cp_indexes]["t"].tail(-1))
            A = (t[:, None] > self.s) * 1

            slope_shared = 0
            if idata is not None and self.tune_method == "parametric":
                slope_mu, slope_sd = self._get_slope_params_from_idata(idata)
                slope_shared = pm.Normal(
                    f"lt_{self.model_idx} - slope_shared", slope_mu, slope_sd
                )
                slope_sigma = pm.HalfCauchy(
                    f"lt_{self.model_idx} - slope_sigma",
                    beta=slope_sd / self.shrinkage_strength,
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                # TODO use delta somehow for slope shared?
                slope_mu, slope_sd = self._get_slope_params_from_idata(idata)
                slope_shared = pm.Deterministic(
                    f"lt_{self.model_idx} - slope_shared", priors[f"prior_{slope_key}"]
                )
                slope_sigma = pm.HalfCauchy(
                    f"lt_{self.model_idx} - slope_sigma",
                    beta=slope_sd / self.shrinkage_strength,
                )
            else:
                slope_sigma = pm.HalfCauchy(
                    f"lt_{self.model_idx} - slope_sigma",
                    beta=self.slope_sd / self.shrinkage_strength,
                )

            slope_z_offset = pm.Normal(
                f"lt_{self.model_idx} - slope_z_offset",
                mu=0,
                sigma=1,
                shape=self.n_groups,
            )
            slope = pm.Deterministic(
                f"lt_{self.model_idx} - slope",
                slope_shared + slope_z_offset * slope_sigma,
            )

            delta_sd = self.delta_sd
            if self.delta_sd is None:
                delta_sd = pm.Exponential(f"lt_{self.model_idx} - tau", 1.5)

            # delta_sigma = pm.HalfCauchy(
            #     f"lt_{self.model_idx} - delta_sigma", beta=delta_sd
            # )
            # delta_z_offset = pm.Laplace(
            #     f"lt_{self.model_idx} - delta_z_offset",
            #     0,
            #     1,
            #     shape=(self.n_groups, self.n_changepoints),
            # )
            # delta = pm.Deterministic(
            #     f"lt_{self.model_idx} - delta", delta_z_offset * delta_sigma
            # )
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
                shape=self.n_groups,
            )

            gamma = -self.s * delta

            return (slope[self.group] + pm.math.sum(A * delta, axis=1)) * t + (
                intercept[self.group] + pm.math.sum(A * gamma, axis=1)
            )

    def definition(
        self,
        model: TimeSeriesModel,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ):
        """
        Add the LinearTrend parameters to the model.

        Parameters
        ----------
        model: TimeSeriesModel
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        model_idxs: dict[str, int]
            Count of the number of components from each type.
        priors: dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating the
            posterior sample in idata.
        idata: az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to set the
            parameters' priors in the model. If idata is not None, each component from
            the model should specify how idata should be used to set its parameters'
            priors.
        """
        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        self.group, self.n_groups, self.groups_ = get_group_definition(
            data, self.pool_type
        )

        with model:
            if self.pool_type == "complete":
                return self._complete_definition(model, data, priors, idata)
            elif self.pool_type == "partial":
                return self._partial_definition(model, data, priors, idata)
            elif self.pool_type == "indivudual":
                pass

    def _get_initval(self, initvals: dict[str, float], model: pm.Model) -> dict:
        """Get the initval of the slope and the intercept of the linear trend.

        Parameters
        ----------
        initvals: dict[str, float]
            Calculated initvals based on data.
        model: pm.Model
            The model for which the initvals will be set.
        """
        slopes = []
        intercepts = []
        for key in sorted(self.groups_.keys()):
            slopes.append(initvals.get(f"slope_{key}", None))
            intercepts.append(initvals.get(f"intercept_{key}", None))

        return {
            model.named_vars[f"lt_{self.model_idx} - slope"]: np.array(slopes),
            model.named_vars[f"lt_{self.model_idx} - intercept"]: np.array(intercepts),
        }

    def _predict_map(self, future, map_approx):
        forecasts = []
        for group_code in self.groups_.keys():
            new_A = (np.array(future["t"])[:, None] > self.s) * 1
            forecasts.append(
                np.array(
                    (
                        map_approx[f"lt_{self.model_idx} - slope"][group_code]
                        + np.dot(
                            new_A,
                            map_approx[f"lt_{self.model_idx} - delta"],
                        )
                    )
                    * future["t"]
                    + (
                        map_approx[f"lt_{self.model_idx} - intercept"][group_code]
                        + np.dot(
                            new_A,
                            (-self.s * map_approx[f"lt_{self.model_idx} - delta"]),
                        )
                    )
                )
            )

            future[f"lt_{self.model_idx}_{group_code}"] = forecasts[-1]

        return np.vstack(forecasts)

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
            new_A = (np.array(future["t"])[:, None] <= self.s) * 1
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

    def _plot(self, plot_params, future, data, scale_params, y_true=None, series=""):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"LinearTrend({self.model_idx})")
        plt.grid()

        plt.plot(future["ds"], future[f"lt_{self.model_idx}{series}"], lw=1)

        plt.legend()

    def __str__(self):
        return f"LT(n={self.n_changepoints},r={self.changepoint_range},tm={self.tune_method})"
