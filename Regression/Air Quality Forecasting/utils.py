"""
Air Quality Forecasting — Utility Functions
Beijing PM2.5 hourly air-quality dataset (UCI). Target: pm2.5 (µg/m³, continuous).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data(filepath="data/PRSA_data.csv"):
    """
    Load the raw Beijing PM2.5 hourly dataset and build a proper datetime index.

    The raw file stores the timestamp across four integer columns (year, month,
    day, hour).  We combine them into a single `Datetime` column so pandas can
    treat the data as an ordered time series.
    """
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

def create_features(df):
    """
    Engineer calendar, lag, and rolling features from the sorted time series.

    Calendar features capture deterministic seasonality (time-of-day, day-of-week,
    month, etc.).  Lag and rolling features give the model recent historical context
    without leaking future information — lags are shifted by the lag length so every
    row only looks backward.  The categorical wind direction `cbwd` is one-hot
    encoded.

    Parameters
    ----------
    df : pd.DataFrame with 'Datetime' (datetime64), 'pm2.5' (target) and the raw
         weather columns, sorted ascending by Datetime.

    Returns
    -------
    pd.DataFrame with new feature columns; rows with NaN target or NaN lag/rolling
    warmup values are dropped.
    """
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    # Calendar features
    df["hour"]       = df["Datetime"].dt.hour
    df["dayofweek"]  = df["Datetime"].dt.dayofweek       # 0=Monday … 6=Sunday
    df["month"]      = df["Datetime"].dt.month
    df["quarter"]    = df["Datetime"].dt.quarter
    df["year"]       = df["Datetime"].dt.year
    df["dayofyear"]  = df["Datetime"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Lag features — shift by lag length to avoid data leakage
    df["lag_1"]  = df["pm2.5"].shift(1)      # previous hour
    df["lag_24"] = df["pm2.5"].shift(24)     # same hour, 1 day ago

    # Rolling mean features — window ends 1 period before the current row
    df["roll_24"]  = df["pm2.5"].shift(1).rolling(window=24).mean()
    df["roll_168"] = df["pm2.5"].shift(1).rolling(window=168).mean()

    # One-hot encode wind direction (categorical), drop_first to avoid collinearity
    df = pd.get_dummies(df, columns=["cbwd"], prefix="wind", drop_first=True)
    wind_cols = [c for c in df.columns if c.startswith("wind_")]
    df[wind_cols] = df[wind_cols].astype(int)

    # Drop rows with a missing target or lag/rolling warmup NaNs
    df = df.dropna(subset=["pm2.5", "lag_1", "lag_24", "roll_24", "roll_168"])
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────

def evaluate_model(name, y_true, y_pred):
    """Print and return a metrics dict: R2, MAE, RMSE, MAPE."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "Model": name,
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "RMSE":  rmse,
        "R2":    float(r2_score(y_true, y_pred)),
        "MAPE":  float(mean_absolute_percentage_error(y_true, y_pred)),
    }
    print(f"\n{'='*40}\n  {name}\n{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:6s}: {v:.4f}")
    return metrics


def plot_predictions(y_true, y_pred, name, n=500):
    """
    Overlay predicted vs actual PM2.5 for the first *n* test-set rows.
    A line plot over a slice of the test period reveals temporal alignment.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.array(y_true)[:n],  label="Actual",    color="steelblue",  linewidth=1.2)
    ax.plot(np.array(y_pred)[:n],  label="Predicted", color="tomato",     linewidth=1.0, alpha=0.85)
    ax.set_title(f"Predicted vs Actual — {name} (first {n} test rows)")
    ax.set_xlabel("Hour index (test set)")
    ax.set_ylabel("pm2.5 (µg/m³)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, name):
    """Side-by-side residuals-vs-predicted scatter and residual histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    residuals = np.array(y_true) - np.array(y_pred)
    axes[0].scatter(np.array(y_pred), residuals, alpha=0.3, s=8, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted (µg/m³)")
    axes[0].set_ylabel("Residual (µg/m³)")
    axes[0].set_title(f"Residuals vs Predicted — {name}")
    axes[1].hist(residuals, bins=50, color="seagreen", edgecolor="black")
    axes[1].set_xlabel("Residual (µg/m³)")
    axes[1].set_title("Residual Distribution")
    plt.tight_layout()
    return fig


def cross_validate_model(model, X, y, cv=5):
    """
    Time-series aware cross-validation using sklearn's TimeSeriesSplit.
    Returns array of R² scores for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
    print(f"  CV R² Scores : {scores.round(4)}")
    print(f"  Mean R²      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    """Return a DataFrame of model metrics sorted by R² descending."""
    df_results = pd.DataFrame(results_list)
    return df_results.sort_values("R2", ascending=False).reset_index(drop=True)
