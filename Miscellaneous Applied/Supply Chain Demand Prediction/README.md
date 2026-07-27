# Supply Chain Demand Prediction

A beginner-level **applied forecasting** project that predicts daily order demand from a supply-chain
dataset, benchmarked against naive and seasonal-naive baselines.

## Problem Statement
Given historical daily order demand, forecast the next day's demand. Does an ML lag-feature model beat
the naive baseline? Chronological split; scored by **MAE** and R².

## Dataset
- **Source**: [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (via HF `alalfi/SupplyChainDataset`), orders aggregated to **daily demand**.
- **`data/daily_demand.csv`**: ~1,004 days (2015 to 2017, after trimming the incomplete tail of the data dump); demand mean ~374, std ~39.

## Project Structure
```
Supply Chain Demand Prediction/
├── 01_eda.ipynb        # Demand over time, distribution, weekly seasonality
├── 02_forecast.ipynb   # Lag/rolling/calendar features, ML vs naive baselines
├── utils.py · requirements.txt · README.md
└── data/daily_demand.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed. Chronological 80/20 split.

| Model | MAE ↓ | R² |
|---|---|---|
| **Ridge / Linear** | **36.4** | −0.38 |
| Random Forest | 38.7 | −0.54 |
| Seasonal naive (lag-7) | 41.3 | −0.95 |
| Naive (lag-1) | 42.0 | −1.02 |

- **ML beats the naive baseline on MAE**. Ridge/Linear cut error to **~36.4 vs 42.0 for naive (~13% lower)**;
  lag + day-of-week features add real value over "same as yesterday".
- **R² is negative for every model**, because the series is **near-constant** (std ~39 around a mean of
  374), predicting the mean is a very strong reference and R² punishes that even when absolute error is
  small. **MAE is the honest metric here.**
- **The naive baseline is hard to beat by much** on a smooth demand series, the win is modest but real,
  and it's the right bar to measure against (a recurring lesson across the time-series projects).

## Tech Stack
- pandas, numpy, matplotlib, scikit-learn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
