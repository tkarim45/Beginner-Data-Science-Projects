# COVID-19 Trend Analysis

A beginner-level **time-series** project that analyses the trend of **daily new confirmed
COVID-19 cases worldwide** and forecasts the next days, benchmarked against naive baselines.

## Problem Statement

Given the global daily new-case series, characterise its trend/seasonality and forecast the next
value. Does an ML forecaster beat the **seasonal-naive** baseline (same weekday last week)? We use
a chronological split and score MAE / RMSE / MAPE / R².

## Dataset

- **Source**: [JHU CSSE COVID-19 time series](https://github.com/CSSEGISandData/COVID-19) (`time_series_covid19_confirmed_global.csv`)
- **Series**: world cumulative confirmed cases summed across all countries, differenced to **daily
  new cases**, **2020-01-22 → 2023-03-09** (**1,143 days**).

## Project Structure

```
COVID-19 Trend Analysis/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/covid.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Features & Models

- **Features**: lags (1, 2, 3, 7, 14 days), rolling mean/std (7, 14), calendar (day-of-week, month, weekend).
- **Models**: naive + seasonal-naive baselines vs Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. Chronological 80/20 split (226 test days).

| Model | MAE | RMSE | MAPE % | R² |
|---|---|---|---|---|
| **Linear Regression** | **65,059** | **98,967** | 16.10 | **0.8688** |
| Ridge | 65,005 | 99,003 | 16.09 | 0.8687 |
| Random Forest | 74,656 | 101,554 | 21.04 | 0.8618 |
| Seasonal Naive | 74,271 | 111,325 | 18.15 | 0.8339 |
| Naive | 742,448 | 785,583 | 312.32 | −7.27 |

## Key Findings

- **Linear Regression forecasts new cases well**. R² **0.869**, MAPE ~16%, thanks to a strong
  weekly reporting cycle and high day-to-day autocorrelation.
- **It beats the seasonal-naive baseline** (RMSE 99k vs 111k), and the plain naive baseline collapses
  (R² −7.3) because cases swing across waves, last value is a terrible flat forecast.
- **The weekly cycle is the dominant seasonal signal**, fewer cases are reported on weekends; the
  `lag_7` and day-of-week features capture this clearly (see the STL decomposition in notebook 01).
- **Caveat**: this forecasts *reported* cases (testing/reporting artifacts and policy changes), not
  true infections, and short-horizon autocorrelation, not epidemiological causality.

## Tech Stack

- pandas, numpy, matplotlib, scikit-learn, statsmodels

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
