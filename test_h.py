import jax
import gc
from tqdm import tqdm
from pathlib import Path
from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)
import pandas as pd
from vangja_hierarchical.components import LinearTrend, FourierSeasonality
import warnings
import arviz as az
from vangja_hierarchical.utils import metrics
from prophet import Prophet

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


dfs = download_data(Path("./data"))
indexes = process_data(dfs[0])
smp = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
gspc_tickers = process_data(dfs[1])


for shrinkage_strength in [1, 10, 50, 100, 500, 1000]:
    for point in pd.date_range("2015-01-01", "2017-01-01"):
        points = f"{point.year}-{'' if point.month > 9 else '0'}{point.month}-{'' if point.day > 9 else '0'}{point.day}"
        train_smp, test_smp, scales_smp = generate_train_test_df_around_point(
            window=365 * 40, horizon=365, dfs=smp, for_prophet=False, point=points
        )
        train_df, test_df, scales_df = generate_train_test_df_around_point(
            window=91, horizon=365, dfs=gspc_tickers, for_prophet=False, point=points
        )

        # test for equality left-right side
        # pr_model = Prophet(
        #     n_changepoints=25, weekly_seasonality=True, yearly_seasonality=True
        # )
        # pr_model.fit(train_smp)
        # pr_yhat = pr_model.make_future_dataframe(365)
        # pr_yhat = pr_model.predict(pr_yhat)

        # trend = LinearTrend(
        #     n_changepoints=25, pool_type="complete", tune_method=None, delta_side="left"
        # )
        # weekly = FourierSeasonality(
        #     period=7,
        #     series_order=3,
        #     pool_type="complete",
        #     tune_method=None,
        # )
        # yearly = FourierSeasonality(
        #     period=365.25,
        #     series_order=10,
        #     pool_type="complete",
        #     tune_method=None,
        # )
        # model = trend * (1 + weekly + yearly)
        # model.fit(train_smp)
        # left_yhat = model.predict(365)

        # trend.delta_side = "right"
        # model.fit(train_smp)
        # right_yhat = model.predict(365)

        # test for complete method
        # fb_train_df, fb_test_df, fb_scales_df = generate_train_test_df_around_point(
        #     window=91, horizon=365, dfs=gspc_tickers, for_prophet=True, point=points
        # )

        # fb_scores = []
        # for df_train, df_test in zip(tqdm(fb_train_df), fb_test_df):
        #     # pr_model = Prophet(
        #     #     n_changepoints=0, weekly_seasonality=False, yearly_seasonality=False
        #     # )
        #     # pr_model.fit(df_train)
        #     # pr_yhat = pr_model.make_future_dataframe(365)
        #     # pr_yhat = pr_model.predict(pr_yhat)

        #     trend = LinearTrend(
        #         n_changepoints=0, pool_type="complete", tune_method=None
        #     )
        #     weekly = FourierSeasonality(
        #         period=7, series_order=3, pool_type="complete", tune_method=None
        #     )

        #     model = trend * (1 + weekly)
        #     model.fit(df_train, progressbar=False)
        #     yhat = model.predict(365)
        #     m = metrics(df_test, yhat, "complete")
        #     fb_scores.append(m["mape"].iloc[0])
        #     # breakpoint()

        # print(sum(fb_scores) / len(fb_scores))

        train_data = pd.concat([train_smp, train_df])
        test_data = pd.concat([test_smp, test_df])

        trace_path = Path("./") / "models" / "simple_advi" / f"{points}" / "trace.nc"
        trace = az.from_netcdf(trace_path)

        t_scale_params = {
            "ds_min": train_smp["ds"].min(),
            "ds_max": train_smp["ds"].max(),
        }
        # for tune_method in [None, "parametric", "prior_from_idata"]:
        for tune_method in ["parametric"]:
            for loss_factor in [-1, 0, 1]:
                for lt_loss_factor in [-1, 0, 1]:
                    trend = LinearTrend(
                        n_changepoints=25,
                        pool_type="individual",
                        tune_method=tune_method,
                        delta_side="right",
                        # shrinkage_strength=sh,
                        loss_factor_for_tune=lt_loss_factor,
                    )
                    weekly = FourierSeasonality(
                        period=7,
                        series_order=3,
                        pool_type="individual",
                        tune_method=tune_method,
                        loss_factor_for_tune=loss_factor,
                        # shrinkage_strength=sh,
                    )
                    yearly = FourierSeasonality(
                        period=365.25,
                        series_order=10,
                        pool_type="individual",
                        tune_method=tune_method,
                        loss_factor_for_tune=loss_factor,
                        # shrinkage_strength=sh,
                    )
                    model = trend * (1 + weekly + yearly)
                    model.fit(
                        train_df,
                        idata=trace,
                        progressbar=True,
                        t_scale_params=t_scale_params,
                    )
                    yhat = model.predict(365)

                    m = metrics(test_df, yhat, "partial")
                    print(
                        f"{m['mape'].mean()} - lf {loss_factor} lt_lf {lt_loss_factor} tm {tune_method}"
                    )

        breakpoint()

        for tune_method in [None, "parametric", "prior_from_idata"]:
            options = [False, True] if tune_method is None else [True]
            for ws in options:
                for ys in options:
                    trend = LinearTrend(
                        n_changepoints=25,
                        pool_type="partial",
                        tune_method=tune_method,
                        delta_side="right",
                        # shrinkage_strength=sh,
                    )
                    weekly = FourierSeasonality(
                        period=7,
                        series_order=3,
                        pool_type="partial",
                        tune_method=tune_method,
                        # shrinkage_strength=sh,
                    )
                    yearly = FourierSeasonality(
                        period=365.25,
                        series_order=10,
                        pool_type="partial",
                        tune_method=tune_method,
                        # shrinkage_strength=sh,
                    )

                    seasonalities = 1
                    if ws:
                        seasonalities = seasonalities + weekly

                    if ys:
                        seasonalities = seasonalities + yearly

                    model = trend * seasonalities
                    model.fit(train_data, idata=trace, progressbar=True)
                    yhat = model.predict(365)

                    m = metrics(test_data, yhat, "partial")

                    print(f"{m['mape'].mean()} - tm {tune_method} ws {ws} ys {ys}")

                    del model
                    gc.collect()
                    jax.clear_caches()

        breakpoint()
