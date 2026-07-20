# Cryptocurrency Price Trend Analysis

A beginner-level **time-series** project that analyses the **daily Bitcoin price (USD)** and tests
how forecastable it is, with an honest look at why high R² here is *persistence*, not skill.

## Problem Statement

Given the daily BTC price series, characterise its trend/volatility and forecast the next value.
The deeper question: do lag/calendar features add genuine predictive skill, or just recover the
random-walk "tomorrow ≈ today"? Chronological split; MAE / RMSE / MAPE / R².

## Dataset

- **Source**: [CoinGecko API](https://www.coingecko.com/en/api), daily BTC/USD close, most recent
  **365 days** (2025-07 → 2026-06).

> Dataset note: the checklist's Kaggle crypto-history set is replaced by the live CoinGecko API
> (free tier, no key needed for a 365-day daily series).

## Project Structure

```
Cryptocurrency Price Trend Analysis/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/btc.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Features & Models

- **Features**: lags (1, 2, 3, 7 days), rolling mean/std (7, 14), calendar (day-of-week, month, weekend).
- **Models**: naive + seasonal-naive baselines vs Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. Chronological 80/20 split (71 test days).

| Model | MAE (USD) | RMSE | MAPE % | R² |
|---|---|---|---|---|
| **Ridge** | **1,028** | **1,345** | 1.46 | **0.9684** |
| Lasso | 1,028 | 1,346 | 1.46 | 0.9684 |
| Linear Regression | 1,060 | 1,386 | 1.51 | 0.9665 |
| Seasonal Naive | 3,318 | 4,337 | 4.85 | 0.6720 |
| Naive | 6,718 | 7,853 | 9.98 | −0.0753 |

## Key Findings

- **The high R² (0.968) is misleading, it is mostly the `lag_1` feature.** BTC price is close to a
  **random walk**: "tomorrow ≈ today" already gives ~1.5% MAPE, and the regressors essentially learn
  that persistence. R² rewards tracking the *level*, which lag-1 does trivially.
- **The honest signal is the *return*, not the *price***, predicting the day-over-day % change (and
  thus direction) is what actually matters for trading, and that is **not** captured here; the models
  add little over pure persistence on returns.
- **Baselines tell the story**: the flat naive baseline fails (R² −0.08) because the price trends, while
  seasonal-naive (R² 0.67) and the lag models track the level, none of which is genuine price *prediction*.
- **Takeaway**: never trust a high R² on a price-*level* time series. Always test against a random-walk /
  naive baseline and evaluate returns, or you will fool yourself into thinking prices are forecastable.

## Tech Stack

- pandas, numpy, matplotlib, scikit-learn, statsmodels

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
