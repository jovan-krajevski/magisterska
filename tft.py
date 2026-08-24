import os

os.environ["TQDM_DISABLE"] = "1"

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# PyTorch Forecasting imports
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from torch.utils.data import DataLoader

from vangja.data_utils import (
    download_data,
    generate_train_test_df_around_point,
    process_data,
)

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")


def metrics(y_true, yhat, label="y"):
    """Calculate forecasting metrics."""
    y = y_true["y"]
    return pd.DataFrame(
        {
            "mse": {f"{label}": mean_squared_error(y, yhat)},
            "rmse": {f"{label}": root_mean_squared_error(y, yhat)},
            "mae": {f"{label}": mean_absolute_error(y, yhat)},
            "mape": {f"{label}": mean_absolute_percentage_error(y, yhat)},
        }
    )


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


def prepare_stock_data(df):
    """Prepare individual stock data for TFT training."""
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


def train_tft_model(dataset, val_dataset=None, pretraining=True):
    """Train TFT model."""
    # Get embedding sizes from dataset
    embedding_sizes = dataset.get_parameters()["embedding_sizes"]

    # Configure network and trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("tb_logs", name="tft_model")

    trainer = Trainer(
        max_epochs=80,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    # Create model
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=3e-4,
        hidden_size=128,
        lstm_layers=2,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=64,
        embedding_sizes=embedding_sizes,
        optimizer="AdamW",
        weight_decay=1e-5,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    )

    # Create dataloaders
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset, batch_size=64, shuffle=False, num_workers=0
        )
    else:
        val_dataloader = None

    # Train model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return tft, trainer


def fine_tune_tft_model(model, dataset, val_dataset=None, freeze_lower_layers=False):
    """Fine-tune pre-trained TFT model on stock data."""
    # If freezing lower layers, set requires_grad=False for specific layers
    if freeze_lower_layers:
        # Freeze LSTM and attention layers
        for name, param in model.named_parameters():
            if any(layer in name for layer in ["lstm", "attention"]):
                param.requires_grad = False

    # Get embedding sizes from dataset
    embedding_sizes = dataset.get_parameters()["embedding_sizes"]

    # Update model's embedding sizes
    model.hparams.embedding_sizes = embedding_sizes

    # Configure trainer for fine-tuning
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("tb_logs", name="tft_finetune")

    trainer = Trainer(
        max_epochs=80,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    # Update learning rate for fine-tuning
    model.hparams.learning_rate = 1e-4
    model.hparams.optimizer_params = {"weight_decay": 1e-6}

    # Create dataloaders
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset, batch_size=64, shuffle=False, num_workers=0
        )
    else:
        val_dataloader = None

    # Fine-tune model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return model, trainer


def load_pretrained_model(checkpoint_path, dataset):
    """Load a pre-trained TFT model from checkpoint."""
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    return model


def iterative_forecast(model, dataset, initial_data, horizon=365, step_size=5):
    """Perform iterative multi-step forecasting."""
    # Initialize predictions
    predictions = []

    # Create a copy of the initial data for forecasting
    forecast_data = initial_data.copy()

    # Perform iterative forecasting
    for i in range(0, horizon, step_size):
        print(f"Forecasting days {i + 1} to {min(i + step_size, horizon)}", end="\r")

        # Create dataset for current forecasting step
        current_dataset = create_timeseries_dataset(forecast_data, is_stock_data=True)
        dataloader = DataLoader(
            current_dataset, batch_size=64, shuffle=False, num_workers=0
        )

        # Get model predictions (quantiles)
        preds = model.predict(dataloader)

        # Extract median predictions (0.5 quantile)
        if isinstance(preds, dict):
            # If predictions contain quantiles, extract median
            median_preds = preds.get("0.5", preds.get("preds", preds))
        else:
            median_preds = preds

        # Store predictions
        predictions.append(median_preds)

        # Update forecast_data with predictions for next iteration
        # Convert predictions to dataframe format
        pred_df = pd.DataFrame(
            {
                "y": median_preds.flatten(),
                "ds": pd.date_range(
                    start=forecast_data["ds"].max() + pd.Timedelta(days=1),
                    periods=len(median_preds.flatten()),
                    freq="D",
                ),
                "series": forecast_data["series"].iloc[
                    0
                ],  # Keep same series identifier
                "day_of_week": pd.date_range(
                    start=forecast_data["ds"].max() + pd.Timedelta(days=1),
                    periods=len(median_preds.flatten()),
                    freq="D",
                ).dayofweek,
                "month": pd.date_range(
                    start=forecast_data["ds"].max() + pd.Timedelta(days=1),
                    periods=len(median_preds.flatten()),
                    freq="D",
                ).month,
                "is_month_start": pd.date_range(
                    start=forecast_data["ds"].max() + pd.Timedelta(days=1),
                    periods=len(median_preds.flatten()),
                    freq="D",
                ).is_month_start.astype(int),
                "is_month_end": pd.date_range(
                    start=forecast_data["ds"].max() + pd.Timedelta(days=1),
                    periods=len(median_preds.flatten()),
                    freq="D",
                ).is_month_end.astype(int),
            }
        )

        # Add volatility (simplified - using last known volatility)
        pred_df["volatility"] = forecast_data["volatility"].iloc[-1]
        pred_df["time_idx"] = range(
            forecast_data["time_idx"].max() + 1,
            forecast_data["time_idx"].max() + 1 + len(pred_df),
        )

        # Append predictions to forecast_data for next iteration
        forecast_data = pd.concat([forecast_data, pred_df], ignore_index=True)

        # Keep only necessary data (last 60+5 days) to manage memory
        if len(forecast_data) > 100:
            forecast_data = forecast_data.iloc[-100:].reset_index(drop=True)

    return np.concatenate(predictions)


def reconstruct_prices(initial_price, log_returns):
    """Reconstruct price paths from log returns."""
    prices = [initial_price]
    for ret in log_returns:
        prices.append(prices[-1] * np.exp(ret))
    return prices[1:]  # Exclude initial price


def evaluate_forecast(true_values, predicted_values, label="Forecast"):
    """Evaluate forecast performance using multiple metrics."""
    mse = mean_squared_error(true_values, predicted_values)
    rmse = root_mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    mape = mean_absolute_percentage_error(true_values, predicted_values)

    # Directional accuracy
    true_direction = np.diff(true_values) > 0
    pred_direction = np.diff(predicted_values) > 0
    directional_accuracy = np.mean(true_direction == pred_direction)

    metrics_df = pd.DataFrame(
        {
            "MSE": [mse],
            "RMSE": [rmse],
            "MAE": [mae],
            "MAPE": [mape],
            "Directional_Accuracy": [directional_accuracy],
        },
        index=[label],
    )

    return metrics_df


def main():
    """Main function to run TFT training and forecasting."""
    # Directory to save results
    csv_path = Path("./out/tft")
    csv_path.mkdir(parents=True, exist_ok=True)

    # Create checkpoints directory
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Download and process data
    dfs = download_data(Path("./data"))
    indexes = process_data(dfs[0])
    sp500 = [index for index in indexes if index["series"].iloc[0] == "^GSPC"]
    stock_tickers = process_data(dfs[1])

    # Prepare data for S&P500 pretraining
    print("Preparing S&P500 data...")
    sp500_df = prepare_sp500_data(sp500[0].copy())

    # Add time index for TFT
    sp500_df["time_idx"] = range(len(sp500_df))
    sp500_df["series"] = "SP500"  # Rename series for clarity

    # Split data for pretraining (35 years train, 5 years validation)
    train_end_idx = int(len(sp500_df) * 35 / 40)  # Approximately 35 years
    train_df = sp500_df[:train_end_idx].copy()
    val_df = sp500_df[train_end_idx:].copy()

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_timeseries_dataset(train_df, is_stock_data=False)
    val_dataset = create_timeseries_dataset(val_df, is_stock_data=False)

    # Train model on S&P500
    print("Training TFT model on S&P500...")
    model, trainer = train_tft_model(train_dataset, val_dataset)

    # Save model checkpoint
    checkpoint_path = "./checkpoints/tft_sp500.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Prepare stock data for fine-tuning
    print("Preparing stock data...")
    prepared_stocks = []
    stock_ids = []

    for i, stock_df in enumerate(stock_tickers[:500]):  # Limit to 500 stocks
        if len(stock_df) >= 91:  # Only use stocks with sufficient data
            prepared_stock = prepare_stock_data(stock_df.copy())
            prepared_stock["time_idx"] = range(len(prepared_stock))
            prepared_stock["series"] = f"STOCK_{i}"  # Assign unique ID
            prepared_stocks.append(prepared_stock)
            stock_ids.append(f"STOCK_{i}")

    if prepared_stocks:
        # Combine stock data
        all_stocks_df = pd.concat(prepared_stocks, ignore_index=True)

        # Split stock data (80 days train, 10 days validation)
        stock_train_dfs = []
        stock_val_dfs = []

        for _, group in all_stocks_df.groupby("series"):
            if len(group) >= 91:
                train_group = group[:80].copy()
                val_group = group[80:90].copy()  # 10 days validation
                stock_train_dfs.append(train_group)
                stock_val_dfs.append(val_group)

        if stock_train_dfs:
            stock_train_df = pd.concat(stock_train_dfs, ignore_index=True)
            stock_val_df = pd.concat(stock_val_dfs, ignore_index=True)

            # Update embedding sizes for stock data
            # This is a simplified approach - in practice, you might want to replace
            # only the ticker embedding layer

            # Create datasets for fine-tuning
        print("Creating stock datasets...")
        stock_train_dataset = create_timeseries_dataset(
            stock_train_df, is_stock_data=True
        )
        stock_val_dataset = create_timeseries_dataset(stock_val_df, is_stock_data=True)

        # Fine-tune model on stocks
        print("Fine-tuning TFT model on stocks...")
        model, trainer = fine_tune_tft_model(
            model, stock_train_dataset, stock_val_dataset
        )

        # Save fine-tuned model
        finetuned_checkpoint_path = "./checkpoints/tft_finetuned.ckpt"
        trainer.save_checkpoint(finetuned_checkpoint_path)
        print(f"Fine-tuned model saved to {finetuned_checkpoint_path}")

        # Perform forecasting on a sample stock
        print("Performing iterative forecasting...")
        sample_stock = stock_train_dfs[0]  # Take first stock as example

        # Get last 60 days for forecasting
        forecast_input = sample_stock.iloc[-60:].copy()

        # Get actual values for comparison (if available)
        actual_values = sample_stock.iloc[-30:].copy()  # Last 30 days for comparison

        # Perform iterative forecasting for 365 days
        forecast_returns = iterative_forecast(
            model, stock_train_dataset, forecast_input, horizon=365, step_size=5
        )

        # Reconstruct price path
        # Get the last actual price
        last_price = sample_stock["y"].iloc[-1]
        # Convert log returns to prices
        forecast_prices = reconstruct_prices(last_price, forecast_returns)

        print(f"Forecast completed. Generated {len(forecast_prices)} price points.")
        print("Sample forecast prices:", forecast_prices[:10])

        # Evaluate forecast if we have actual values for comparison
        if len(actual_values) > 0 and len(forecast_prices) > 0:
            # Align forecast with actual values (first N points)
            eval_length = min(len(actual_values), len(forecast_prices))
            actual_prices = actual_values["y"].values[:eval_length]
            predicted_prices = forecast_prices[:eval_length]

            # Evaluate forecast
            metrics_df = evaluate_forecast(
                actual_prices, predicted_prices, "TFT_Forecast"
            )
            print("Forecast Evaluation Metrics:")
            print(metrics_df)

            # Save metrics
            metrics_file = csv_path / "tft_forecast_metrics.csv"
            metrics_df.to_csv(metrics_file)
            print(f"Metrics saved to {metrics_file}")

    print("TFT implementation completed.")


if __name__ == "__main__":
    main()
