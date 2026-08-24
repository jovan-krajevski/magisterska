# Temporal Fusion Transformer for S&P500 and Stock Forecasting

This document explains how to implement a **Temporal Fusion Transformer (TFT)** for time series forecasting with **PyTorch Forecasting**.  
The workflow has two main phases:

1. **Pretraining on 40 years of daily S&P500 data** (long univariate series).  
2. **Fine-tuning on 500 individual stocks, each with only 91 days of history**.  

The goal is to **forecast 5 days ahead iteratively** until reaching **365 days into the future**.

---

## 1. Data Preparation

### 1.1 Input Series

- **S&P500 index (40 years daily close prices)**  
  - Frequency: daily.  
  - Total points: >10,000.  
  - Target: log returns (`r_t = log(p_t) - log(p_{t-1})`).  

- **500 stocks (91 days each)**  
  - Frequency: daily trading days.  
  - Target: log returns.  

### 1.2 Features

- **Time-varying known inputs**
  - Day-of-week (categorical, embedded).  
  - Month-of-year (categorical, embedded).  
  - Is-month-start / is-month-end.  

- **Time-varying observed inputs (past only)**
  - Target log returns.  
  - Rolling 21-day volatility (std of returns).  

- **Static features**
  - Ticker ID (for 500 stocks, as categorical embedding).  

### 1.3 Train / Validation Split

- **S&P500 pretraining:**  
  - Train: first 35 years.  
  - Validation: last 5 years.  

- **Stock fine-tuning:**  
  - Train: first 80 days.  
  - Validation: last 10 days.  

---

## 2. Model Configuration

### 2.1 Encoder / Decoder Length

- `encoder_length = 60` (≈ 2 months).  
- `decoder_length = 5` (forecast horizon).  

### 2.2 TFT Hyperparameters

Use the same hyperparameters for pretraining and fine-tuning:

- `hidden_size = 128`  
- `lstm_layers = 2`  
- `attention_head_size = 4`  
- `dropout = 0.15`  
- `hidden_continuous_size = 64`  
- `embedding_sizes = {"day_of_week": (7, 4), "month": (12, 6), "ticker_id": (500, 32)}`  
- `learning_rate = 3e-4`  
- `optimizer = AdamW`  
- `weight_decay = 1e-5`  
- `batch_size = 64`  
- `epochs = 80` with early stopping (patience = 10).  
- `loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])`  

---

## 3. Training Procedure

### 3.1 Pretraining

- Train TFT on the S&P500 dataset using the configuration above.  
- Save the model checkpoint after best validation performance.  

### 3.2 Fine-Tuning

- Load pretrained TFT weights.  
- Replace only the **ticker embedding layer** to include 500 stock IDs.  
- Fine-tune on each stock’s 91-day data:  
  - Train all stocks jointly (multi-series panel).  
  - This leverages shared patterns across stocks.  
- Freeze lower layers optionally (first LSTM + attention) for stability, or fine-tune all layers with a smaller learning rate (`1e-4`).  

---

## 4. Forecasting Strategy

### 4.1 Iterative Multi-Step Forecasting

Goal: forecast 365 days ahead in 5-day increments.

Procedure:

1. Prepare a `TimeSeriesDataSet` with input window = 60 days, prediction length = 5.  
2. Forecast 5 days ahead using the fine-tuned TFT.  
3. Append the forecasted 5 days (as if they were observed) to the input sequence.  
4. Shift window forward and forecast next 5 days.  
5. Repeat until 365 days are predicted (73 iterations).  

### 4.2 Important Notes

- Predicted values become model inputs for subsequent steps (recursive forecasting).  
- Keep track of uncertainty intervals (from quantile predictions).  
- After forecasting log returns, reconstruct price paths: `price_t = price_{t-1} * exp(predicted_return_t)`

---

## 5. Evaluation

### 5.1 Metrics

- **MAE / RMSE** for point forecasts.  
- **Pinball loss** for quantiles.  
- **Directional accuracy** (% of correctly predicted return signs).  

### 5.2 Validation

- Rolling window validation for robustness.  
- For stocks: validate on last 10 days of their 91-day history.  
- For long S&P500 series: validate on last 5 years.  

---

## 6. Implementation Notes

- Use **PyTorch Forecasting’s `TemporalFusionTransformer`** class.  
- Dataloaders: `TimeSeriesDataSet` → `DataLoader`.  
- Training loop: `Trainer` from PyTorch Lightning.  
- Save model checkpoint after pretraining.  
- Reload checkpoint and fine-tune for stocks.  
- Look at `neural_prophet.py` to see how to load the data, which metrics to use and how to store the output. Also, everything else that you need is there.

---

## 7. Summary of Key Settings

- **Encoder length:** 252  
- **Decoder length:** 5  
- **Hidden size:** 128  
- **LSTM layers:** 2  
- **Attention heads:** 4  
- **attention_head_size = 32**
- **Dropout:** 0.15  
- **Learning rate:** 3e-4 (fine-tune: 1e-4)  
- **Optimizer:** AdamW with weight decay 1e-5 (fine-tune 1e-6)
- **Batch size:** 64  
- **Epochs:** 80 with early stopping  
- **Loss:** QuantileLoss([0.1, 0.5, 0.9])  
- **Features:** calendar features, rolling volatility, ticker embedding  
- **Forecasting horizon:** 5 days recursive → 365 days  

---
