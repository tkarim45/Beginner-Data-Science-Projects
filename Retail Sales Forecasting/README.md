# Retail Sales Forecasting

A beginner-level **time-series forecasting** project that predicts **daily sales revenue** for a UK
online retailer, benchmarked against naive baselines, and an honest lesson in short, volatile series.

## Problem Statement

Given a daily revenue series, forecast the next day's sales. Does an ML model beat the
**seasonal-naive** baseline? Chronological split; MAE / RMSE / MAPE / R².

## Dataset

- **Source**: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), transactions
  aggregated to **daily total revenue** (Quantity × UnitPrice), **2010-12-01 → 2011-12-09** (**305 days**).

> Dataset note: the checklist's Kaggle Rossmann competition data is download-gated; we reuse the
> in-repo UCI Online Retail transactions as a real daily retail-sales series.

## Project Structure

```
Retail Sales Forecasting/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/sales.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Features & Models

- **Features**: lags (1, 2, 3, 7 days), rolling mean/std (7, 14), calendar (day-of-week, month, weekend).
- **Models**: naive + seasonal-naive baselines vs Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. Chronological 80/20 split (59 test days).

| Model | MAE | RMSE | MAPE % | R² |
|---|---|---|---|---|
| **Linear Regression** | **16,803** | **27,068** | 28.14 | **−0.0326** |
| Lasso | 17,056 | 27,349 | 28.31 | −0.0542 |
| Ridge | 17,043 | 27,370 | 28.42 | −0.0559 |
| Seasonal Naive | 24,618 | 33,234 | 51.01 | −0.5567 |
| Naive | 44,156 | 51,568 | 74.12 | −2.7481 |

## Key Findings

- **An honest hard case**, every model has **R² ≈ 0 or below**: with only ~290 usable days, a
  single store-wide revenue series, and a sharp pre-Christmas ramp, day-level revenue is barely
  forecastable from its own past.
- **ML still beats the baselines by a lot**. Linear RMSE 27,068 vs seasonal-naive 33,234 and naive
  51,568. The models *reduce* error substantially even when absolute R² is poor.
- **Negative R² ≠ useless**, it means "worse than predicting the test mean", but the test mean is an
  oracle you don't have in production; vs the honest naive baseline the model is clearly better.
- **The lesson**: short, spiky business series need more history, store-level granularity, or
  exogenous drivers (promotions, holidays), and you must always benchmark against naive baselines
  rather than trusting R² alone.

## Tech Stack

- pandas, numpy, matplotlib, scikit-learn, statsmodels

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
