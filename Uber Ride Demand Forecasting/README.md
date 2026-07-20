# Uber Ride Demand Forecasting

A beginner-level **time-series forecasting** project that predicts **daily Uber pickup demand** in
New York City from its own history, benchmarked against naive baselines.

## Problem Statement

Given the daily pickup-count series, forecast the next day's demand. Does an ML model beat the
**seasonal-naive** baseline (same weekday last week)? Chronological split; MAE / RMSE / MAPE / R².

## Dataset

- **Source**: [FiveThirtyEight Uber TLC FOIL data](https://github.com/fivethirtyeight/uber-tlc-foil-response)
, **4.53M** NYC Uber pickups (Apr, Sep 2014) aggregated to **daily counts** (**183 days**).

## Project Structure

```
Uber Ride Demand Forecasting/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/pickups.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Features & Models

- **Features**: lags (1, 2, 3, 7, 14 days), rolling mean/std (7), calendar (day-of-week, month, weekend).
- **Models**: naive + seasonal-naive baselines vs Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. Chronological 80/20 split (34 test days).

| Model | MAE | RMSE | MAPE % | R² |
|---|---|---|---|---|
| **Ridge** | **3,294** | **4,090** | 9.76 | **0.5009** |
| Linear Regression | 3,284 | 4,091 | 9.72 | 0.5006 |
| Lasso | 3,297 | 4,115 | 9.75 | 0.4946 |
| Seasonal Naive | 3,955 | 5,473 | 11.62 | 0.1063 |
| Random Forest | 5,726 | 7,101 | 15.92 | −0.5047 |
| Naive | 7,074 | 8,524 | 19.84 | −1.1683 |

## Key Findings

- **Daily demand is moderately forecastable**. Ridge reaches **R² 0.50**, MAPE ~10%, driven by a
  strong weekly cycle (busy weekends) and a clear upward trend over the six months.
- **Linear models win; tree ensembles overfit**, with only 169 usable rows, Random Forest and
  Gradient Boosting go **negative R²**, while regularised linear models generalise. A textbook
  small-n reminder to prefer simple models.
- **ML beats the baselines**. RMSE 4,090 vs seasonal-naive 5,473 and naive 8,524: the trend +
  day-of-week features add real value over "same as last week".
- **The growth trend is key**. Uber was expanding fast in 2014; the rolling-mean and lag features
  let linear models track that ramp, which seasonal-naive (no trend term) misses.

## Tech Stack

- pandas, numpy, matplotlib, scikit-learn, statsmodels

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
