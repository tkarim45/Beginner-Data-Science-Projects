"""
Retail Sales Forecasting — Utility Functions
Time-series forecasting: engineer calendar + lag/rolling features, then forecast
with classic regressors against naive / seasonal-naive baselines, using a
chronological train/test split (never shuffle time series).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Per-dataset configuration (the only thing that changes between TS projects):
DATE_COL = "date"          # raw date column name
VALUE_COL = "sales"        # raw target column name
FREQ = "D"              # "D" daily, "h" hourly
SEASONAL_PERIOD = 7 # 7 for daily (weekly), 24 for hourly (daily)
LAGS = [1, 2, 3, 7]                # list of lag steps to use as features
ROLL_WINDOWS = [7, 14]       # rolling-window sizes


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data(filepath="data/sales.csv"):
    """Load the series as a tidy frame with `date` (datetime) and `y` (target)."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={DATE_COL: "date", VALUE_COL: "y"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "y"]]


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

def add_calendar_features(df):
    """Calendar features from the timestamp."""
    df = df.copy()
    d = df["date"].dt
    df["month"] = d.month
    df["day"] = d.day
    df["dayofweek"] = d.dayofweek
    df["dayofyear"] = d.dayofyear
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    if FREQ == "h":
        df["hour"] = d.hour
    return df


def add_lag_features(df, lags=LAGS, windows=ROLL_WINDOWS):
    """Lagged values and rolling mean/std of the target (no future leakage)."""
    df = df.copy()
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)
    for w in windows:
        df[f"rollmean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"rollstd_{w}"] = df["y"].shift(1).rolling(w).std()
    return df


def build_features(df):
    """Full feature pipeline; drops rows with NaN lags at the start."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    return df.dropna().reset_index(drop=True)


def feature_columns(df):
    return [c for c in df.columns if c not in ("date", "y")]


# ─────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────

def chronological_split(df, test_frac=0.2):
    """Last `test_frac` of the timeline is the test set (no shuffling)."""
    n = len(df)
    cut = int(n * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ─────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────

def naive_forecast(train, test):
    """Predict every test point as the last observed training value."""
    return np.full(len(test), train["y"].iloc[-1])


def seasonal_naive_forecast(full_df, test, period=SEASONAL_PERIOD):
    """Predict y[t] = y[t-period] (e.g. same weekday last week / same hour yesterday)."""
    y = full_df["y"].values
    idx = test.index.values
    return np.array([y[i - period] if i - period >= 0 else y[i] for i in idx])


# ─────────────────────────────────────────────
# Models & Evaluation
# ─────────────────────────────────────────────

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=7),
    }


def evaluate(y_true, y_pred):
    """MAE / RMSE / MAPE(%) / R²."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "MAPE": round(float(mape), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def run_all(feat_df, test_frac=0.2):
    """Fit baselines + all regressors on a chronological split; return a results DataFrame."""
    train, test = chronological_split(feat_df, test_frac)
    cols = feature_columns(feat_df)
    Xtr, ytr = train[cols], train["y"]
    Xte, yte = test[cols], test["y"]
    rows = []
    rows.append({"model": "Naive", **evaluate(yte, naive_forecast(train, test))})
    rows.append({"model": "Seasonal Naive", **evaluate(yte, seasonal_naive_forecast(feat_df, test))})
    preds = {}
    for name, model in get_models().items():
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        preds[name] = p
        rows.append({"model": name, **evaluate(yte, p)})
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return res, train, test, preds
