from pathlib import Path

import pandas as pd
import yfinance

from vangja.tickers import DJI_TICKERS, GSPC_TICKERS, INDEXES, IXIC_TICKERS


def download_data(download_folder: Path) -> list[pd.DataFrame]:
    start = "1900-01-01"
    end = "2025-01-01"
    dfs: list[pd.DataFrame] = []
    for tickers, filename in zip(
        [INDEXES, GSPC_TICKERS, DJI_TICKERS, IXIC_TICKERS],
        ["indexes.csv", "gspc.csv", "dji.csv", "ixic.csv"],
        strict=True,
    ):
        file_path = download_folder / filename
        if file_path.exists():
            dfs.append(pd.read_csv(file_path, header=[0, 1], index_col=[0]))
            continue

        dfs.append(
            yfinance.download(
                tickers,
                interval="1d",
                start=start,
                end=end,
            )
        )
        dfs[-1].to_csv(file_path)

    return dfs


def process_data(data: pd.DataFrame) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    df = (data["Close"] + data["Open"] + data["High"] + data["Low"]) / 4
    df.index = pd.to_datetime(df.index)
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_date_range).interpolate()
    df["ds"] = df.index
    df.reset_index(drop=True, inplace=True)
    for col in df.columns:
        if col == "ds":
            continue

        ticker = df[[col, "ds"]].copy()
        ticker["series"] = ticker.columns[0]
        ticker = ticker.rename(columns={ticker.columns[0]: "y"})
        dfs.append(ticker)

    return sorted(dfs, key=lambda x: x["series"].iloc[0])


def generate_train_test_df(
    start: int,
    window: int,
    horizon: int,
    dfs: list[pd.DataFrame],
    for_prophet: bool = False,
    perform_scaling: bool = True,
) -> (
    tuple[list[pd.DataFrame], list[pd.DataFrame], list[float]]
    | tuple[pd.DataFrame, pd.DataFrame, list[float]]
    | None
):
    y_col = "y"
    train_dfs: list[pd.DataFrame] = []
    test_dfs: list[pd.DataFrame] = []
    scales: list[float] = []
    for df in dfs:
        train_df = df[start : start + window].copy()
        test_df = df[start + window : start + window + horizon].copy()
        if train_df.isna().any().any() or test_df.isna().any().any():
            continue

        train_df["y"] = train_df[y_col]
        test_df["y"] = test_df[y_col]

        if perform_scaling:
            scales.append(train_df[y_col].max())
            train_df["y"] = train_df[y_col] / scales[-1]
            test_df["y"] = test_df[y_col] / scales[-1]

        train_dfs.append(train_df)
        test_dfs.append(test_df)

    if len(train_dfs) == 0:
        return None

    if for_prophet:
        return train_dfs, test_dfs, scales

    return pd.concat(train_dfs), pd.concat(test_dfs), scales


def generate_train_test_df_around_point(
    window: int,
    horizon: int,
    dfs: list[pd.DataFrame],
    point: str = "2009-09-01",
    for_prophet: bool = False,
    perform_scaling: bool = True,
) -> (
    tuple[list[pd.DataFrame], list[pd.DataFrame], list[float]]
    | tuple[pd.DataFrame, pd.DataFrame, list[float]]
    | None
):
    train_dfs: list[pd.DataFrame] = []
    test_dfs: list[pd.DataFrame] = []
    scales: list[float] = []

    for df in dfs:
        point_idx = len(df[df["ds"] < point])
        check = generate_train_test_df(
            start=point_idx - window,
            window=window,
            horizon=horizon,
            dfs=[df],
            for_prophet=for_prophet,
            perform_scaling=perform_scaling,
        )
        if check is None:
            continue

        train_df, test_df, scale = check

        scales += scale

        if for_prophet:
            train_dfs += train_df
            test_dfs += test_df
        else:
            train_dfs.append(train_df)
            test_dfs.append(test_df)

    if len(train_dfs) == 0:
        return None

    if for_prophet:
        return train_dfs, test_dfs, scales

    return pd.concat(train_dfs), pd.concat(test_dfs), scales
