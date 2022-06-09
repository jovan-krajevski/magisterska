import pickle
import time
from pathlib import Path

import aesara
import cloudpickle
import numpy as np
import pandas as pd
import pymc as pm
import yfinance
from scipy import stats as ss

indexes = ["^GSPC"]

OVERWRITE_ANYWAY = False

DATA_LOCATION = Path(".") / "data"
DATA_LOCATION.mkdir(exist_ok=True, parents=True)

start_time = time.time()

if OVERWRITE_ANYWAY or not (DATA_LOCATION / "indexes.pkl").is_file():
    daily_smp = yfinance.download(" ".join(indexes),
                                  period="max",
                                  interval="1d")
    daily_smp.to_pickle(DATA_LOCATION / "indexes.pkl")

daily_smp = pd.read_pickle(DATA_LOCATION / "indexes.pkl")

end_time = time.time()
print(f"{end_time - start_time:.2f}s")

train_smp = daily_smp[daily_smp.index < "01-01-2007"].copy()
test_smp = daily_smp[daily_smp.index >= "01-01-2007"].copy()

def transform_close(df):
    df["close"] = df["Adj Close"]
    df["close_return"] = df["close"].pct_change(periods=1)
    df["close_diff"] = df["close"].diff(periods=1)
    df["close_log_return"] = np.log(df["close"]) - np.log(df["close"].shift(1))
    df.dropna(inplace=True)


transform_close(train_smp)
transform_close(test_smp)

def get_rolling_windows(df, L=250):
    return [df.iloc[x:x + L] for x in range(len(df) - L + 1)]


start_time = time.time()

train_data = get_rolling_windows(train_smp)
test_data = get_rolling_windows(test_smp)

end_time = time.time()
print(f"{end_time - start_time:.2f}s")

def from_posterior(param, samples, testval, set_testval):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = ss.gaussian_kde(samples.data.flatten())(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    if set_testval:
        return pm.distributions.Interpolated(param, x, y, initval=testval)

    return pm.distributions.Interpolated(param, x, y)

def get_stats(sample):
    return [
        np.mean(sample),
        ss.tstd(sample),
        np.mean(sample > 0),
        ss.skew(sample),
        ss.kurtosis(sample),
    ] + [np.percentile(sample, p) for p in range(0, 101, 5)]

MODEL_LOCATION = Path(".") / "models"
DATA_LOCATION.mkdir(exist_ok=True, parents=True)


def read_stats_and_model(model_name):
    if not (MODEL_LOCATION / model_name).is_file():
        return [], None

    with open(MODEL_LOCATION / model_name, "rb") as f:
        return pickle.load(f)


def write_stats_and_model(model_name, stats, model):
    with open(MODEL_LOCATION / model_name, "wb") as f:
        cloudpickle.dump((stats, model), f)

def get_initial_normal_model(data, prior_mu_mean, prior_mu_sigma,
                             prior_std_sigma):
    mu_testval, std_testval = ss.norm.fit(data.get_value())
    model = pm.Model()
    with model:
        mu = pm.Normal("mu",
                       mu=prior_mu_mean,
                       sigma=prior_mu_sigma,
                       initval=mu_testval)
        std = pm.HalfNormal("std", sigma=prior_std_sigma, initval=std_testval)
        obs = pm.Normal("obs", mu=mu, sigma=std, observed=data)

    return model


def get_next_normal_model(data, trace, set_testval):
    mu_testval, std_testval = ss.norm.fit(data.get_value())
    model = pm.Model()
    with model:
        mu = from_posterior("mu", trace["posterior"]["mu"], mu_testval,
                            set_testval)
        std = from_posterior("std", trace["posterior"]["std"], std_testval,
                             set_testval)
        obs = pm.Normal("obs", mu=mu, sigma=std, observed=data)

    return model

def get_initial_laplace_model(data, prior_mu_mean, prior_mu_sigma,
                              prior_std_sigma):
    mu_testval, b_testval = ss.laplace.fit(data.get_value())
    model = pm.Model()
    with model:
        mu = pm.Normal("mu",
                       mu=prior_mu_mean,
                       sigma=prior_mu_sigma,
                       initval=mu_testval)
        b = pm.HalfNormal("b", sigma=prior_std_sigma, initval=b_testval)
        obs = pm.Laplace("obs", mu=mu, b=b, observed=data)

    return model


def get_next_laplace_model(data, trace, set_testval):
    mu_testval, b_testval = ss.laplace.fit(data.get_value())
    model = pm.Model()
    with model:
        mu = from_posterior("mu", trace["mu"], mu_testval, set_testval)
        b = from_posterior("b", trace["b"], b_testval, set_testval)
        obs = pm.Laplace("obs", mu=mu, b=b, observed=data)

    return model

start_time = time.time()

prior_mu_mean = np.array(
    [window["close_log_return"].mean() for window in train_data]).mean()
prior_mu_sigma = np.array(
    [window["close_log_return"].mean() for window in train_data]).std(ddof=1)
prior_std_sigma = np.array(
    [window["close_log_return"].std() for window in train_data]).std(ddof=1)

end_time = time.time()
print(f"{end_time - start_time:.2f}s")
print(prior_mu_mean, prior_mu_sigma, prior_std_sigma)

def train_model(
        model_name,
        get_initial_model_func,
        initial_model_args,
        get_next_model_func,
        update_priors_on=60,  # update priors after 60 windows
        save_every=10,  # save model after 10 windows
        verbosity=10,  # print info about progress after 10 windows
        draws=1000,
        chains=4,
        cores=4):
    success = False
    while not success:
        try:
            data_sample = aesara.shared(
                train_data[0]["close_log_return"].to_numpy())

            stats, model = read_stats_and_model(model_name)
            trace = None

            if not model:
                model = get_initial_model_func(data_sample,
                                               *initial_model_args)

            if len(stats) and len(stats) % update_priors_on == 0:
                # recalculate last window so that we have trace
                stats = stats[:-1]

            for idx, window in enumerate(train_data):
                if idx % verbosity == 0:
                    print(
                        f"Window {idx + 1}/{len(train_data)} ({(idx + 1)/len(train_data)*100:.2f}%)..."
                    )

                if idx < len(stats):
                    continue  # window is already processed

                data_sample.set_value(window["close_log_return"].to_numpy())

                if idx % update_priors_on == 0 and idx > 0:
                    next_model = get_next_model_func(data_sample, trace, True)
                    if not np.isnan(
                            np.array(list(
                                next_model.point_logps().values()))).any():
                        model = next_model
                    else:
                        with open("logs.txt", "a") as f:
                            f.write(
                                f"{model_name}_{idx} - Failed set_testval\n")

                        model = get_next_model_func(data_sample, trace, False)

                with model:
                    trace = pm.sample(draws=draws,
                                      chains=chains,
                                      cores=cores,
                                      progressbar=False)
                    posterior_obs = pm.sample_posterior_predictive(
                        trace, progressbar=False)

                stats.append(
                    get_stats(
                        posterior_obs["observed_data"]["obs"].data.flatten()))

                if idx % save_every == 0:
                    write_stats_and_model(model_name, stats, model)

            write_stats_and_model(model_name, stats, model)
            success = True
        except Exception as e:
            with open("logs.txt", "a") as f:
                f.write(str(e))
                f.write("\n")

train_model("fixed_normal.pkl", get_initial_normal_model,
            [prior_mu_mean, prior_mu_sigma, prior_std_sigma],
            get_next_normal_model)
train_model("fixed_laplace.pkl", get_initial_laplace_model,
            [prior_mu_mean, prior_mu_sigma, prior_std_sigma],
            get_next_laplace_model)
