import numpy as np

# PyTorch Forecasting imports
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


def calculate_log_returns(prices):
    """Calculate log returns from price data."""
    return np.log(prices / prices.shift(1))


def add_calendar_features(df):
    """Add calendar features to the dataframe."""
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_month_start"] = df["ds"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)
    return df


def add_rolling_volatility(df, window=21):
    """Add rolling volatility feature."""
    df["volatility"] = df["y"].rolling(window=window, min_periods=1).std()
    return df


def prepare_sp500_data(df):
    """Prepare S&P500 data for TFT training."""
    # Calculate log returns
    df["y"] = calculate_log_returns(df["y"])

    # Add calendar features
    df = add_calendar_features(df)

    # Add rolling volatility
    df = add_rolling_volatility(df)

    # Remove NaN values
    df = df.dropna().reset_index(drop=True)

    return df


def create_timeseries_dataset(
    df, encoder_length=60, decoder_length=5, is_stock_data=False
):
    """Create TimeSeriesDataSet for TFT training."""
    # Count unique series for embedding size
    unique_series = df["series"].nunique()

    # Define embedding sizes based on data type
    if is_stock_data:
        embedding_sizes = {
            "day_of_week": (7, 4),
            "month": (12, 6),
            "series": (
                min(unique_series, 500),
                32,
            ),  # Limit to 500 for memory efficiency
        }
    else:
        embedding_sizes = {
            "day_of_week": (7, 4),
            "month": (12, 6),
            "series": (unique_series, 32),
        }

    # Define dataset
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="y",
        group_ids=["series"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=decoder_length,
        max_prediction_length=decoder_length,
        static_categoricals=["series"],
        time_varying_known_categoricals=["day_of_week", "month"],
        time_varying_known_reals=["is_month_start", "is_month_end"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["y", "volatility"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None,
        embedding_sizes=embedding_sizes,
    )

    return dataset
