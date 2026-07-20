# Website Traffic Forecasting

A beginner-level **time-series forecasting** project that predicts **daily page views** of a
Wikipedia article from its own history, benchmarked against naive baselines.

## Problem Statement

Given a daily web-traffic series, forecast the next day's views. Does an ML model beat the
**seasonal-naive** baseline (same weekday last week)? Chronological split; MAE / RMSE / MAPE / R².

## Dataset

- **Source**: [Wikimedia Pageviews API](https://wikimedia.org/api/rest_v1/), daily views of the
  English Wikipedia article **"Python (programming language)"**, **2020 to 2023** (**1,461 days**).

> Dataset note: the checklist's Kaggle web-traffic competition data is download-gated; the openly
> available Wikimedia Pageviews API gives a real, clean daily web-traffic series with no account.

## Project Structure

```
Website Traffic Forecasting/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/traffic.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Features & Models

- **Features**: lags (1, 2, 3, 7, 14 days), rolling mean/std (7, 14), calendar (day-of-week, month, weekend).
- **Models**: naive + seasonal-naive baselines vs Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. Chronological 80/20 split (290 test days).

| Model | MAE | RMSE | MAPE % | R² |
|---|---|---|---|---|
| **Random Forest** | **1,518** | **3,386** | 12.69 | **0.2801** |
| Gradient Boosting | 1,537 | 3,466 | 12.73 | 0.2456 |
| Ridge | 1,729 | 3,560 | 14.62 | 0.2042 |
| Linear Regression | 1,727 | 3,565 | 14.50 | 0.2019 |
| Seasonal Naive | 2,617 | 5,197 | 23.11 | −0.6957 |
| Naive | 18,565 | 18,815 | 196.30 | −21.22 |

## Key Findings

- **Web traffic is only moderately forecastable**, best R² **0.28** (Random Forest), MAPE ~13%.
  Page views are noisy and driven by external events (news, releases) the model can't see.
- **But ML clearly beats the baselines**. RMSE 3,386 vs seasonal-naive 5,197 and a disastrous
  naive 18,815 (R² −21): lag + day-of-week features capture the strong weekday/weekend cycle.
- **Random Forest edges out linear models** here, traffic spikes are non-linear, so the ensemble
  handles the bursty tail better.
- **Honest ceiling**: a univariate model captures routine weekly rhythm but not traffic shocks;
  beating ~0.3 R² would need exogenous signals (events, referrals, search trends).

## Tech Stack

- pandas, numpy, matplotlib, scikit-learn, statsmodels

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
