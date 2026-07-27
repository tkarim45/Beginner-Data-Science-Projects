# Weather Forecasting

A beginner-level **time-series forecasting** project that predicts **hourly temperature** at
Jena, Germany from its own recent history (lag features) plus calendar features, benchmarked
against naive and seasonal-naive baselines.

## Problem Statement

Given an hourly temperature series, forecast the next value. The key question for any forecaster:
does it beat the **seasonal-naive** baseline (same hour yesterday)? We use a chronological
train/test split (never shuffle time) and score MAE / RMSE / MAPE / R².

## Dataset

- **Source**: [Jena Climate (Max Planck Institute, via TF-Keras)](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)
- **Series**: 10-minute records **resampled to hourly**, years **2015 to 2016** → **17,471 hours**.
- Target: `temperature` (°C); also carries pressure and humidity.

> Dataset note: the checklist's Kaggle historical-hourly-weather set is replaced by the openly
> hosted Jena Climate dataset (a standard weather-forecasting benchmark).

## Project Structure

```
Weather Forecasting/
├── 01_eda.ipynb              # Series, STL decomposition, daily profile, ACF
├── 02_data_cleaning.ipynb    # Lag + rolling + calendar features, chronological split
├── 03_model_building.ipynb   # Baselines + 7 regressors, MAE/RMSE/MAPE/R², forecast plot
├── utils.py · requirements.txt · README.md
└── data/weather.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Features & Models

- **Features**: lags (1, 2, 3, 24, 48, 168 h), rolling mean/std (24 h, 168 h), calendar (hour,
  day, month, day-of-week, weekend).
- **Models**: naive + seasonal-naive baselines vs Linear, Ridge, Lasso, Decision Tree, Random
  Forest, Gradient Boosting, KNN.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. Chronological 80/20
split (3,461 test hours).

| Model | MAE (°C) | RMSE | MAPE % | R² |
|---|---|---|---|---|
| **Linear Regression** | **0.392** | **0.566** | 15.55 | **0.9953** |
| Ridge | 0.392 | 0.566 | 15.56 | 0.9953 |
| Random Forest | 0.412 | 0.586 | 18.46 | 0.9949 |
| Gradient Boosting | 0.464 | 0.641 | 24.59 | 0.9940 |
| Seasonal Naive | 2.367 | 3.057 | 104.25 | 0.8628 |
| Naive | 12.072 | 13.909 | 834.58 | −1.8412 |

## Key Findings

- **Next-hour temperature is highly forecastable**. Linear Regression reaches **MAE 0.39 °C**
  (R² 0.995), because temperature is smooth and strongly autocorrelated at lag 1.
- **ML crushes the baselines**. RMSE 0.57 vs seasonal-naive 3.06 (5×) and naive 13.9 (24×):
  lag + calendar features capture both the daily cycle and the slow trend.
- **A linear model wins**, with informative lag features the relationship is near-linear, so
  Linear/Ridge edge out the tree ensembles (and run far faster).
- **The lag-1 feature does most of the work**, temperature changes little hour-to-hour; the value
  of the model is in the multi-hour and daily-cycle features that refine that persistence.

## Tech Stack

- pandas, numpy, matplotlib, scikit-learn, statsmodels

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
