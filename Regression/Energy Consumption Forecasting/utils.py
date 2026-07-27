"""
Energy Consumption Forecasting — Utility Functions
PJM East hourly electricity demand dataset. Target: PJME_MW (megawatts, continuous).
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

def load_data(filepath="data/PJME_hourly.csv"):
    """Load the raw PJM hourly dataset; parse Datetime column."""
    df = pd.read_csv(filepath, parse_dates=["Datetime"])
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
    row only looks backward.

    Parameters
    ----------
    df : pd.DataFrame with 'Datetime' (datetime64) and 'PJME_MW' columns,
         sorted ascending by Datetime.

    Returns
    -------
    pd.DataFrame with new feature columns; NaN rows (from lag/rolling) are dropped.
    """
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    # Calendar features
    df["hour"]       = df["Datetime"].dt.hour
    df["dayofweek"]  = df["Datetime"].dt.dayofweek       # 0=Monday … 6=Sunday
    df["month"]      = df["Datetime"].dt.month
    df["quarter"]    = df["Datetime"].dt.quarter
    df["year"]       = df["Datetime"].dt.year
    df["dayofyear"]  = df["Datetime"].dt.dayofyear
    df["weekofyear"] = df["Datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Lag features — shift by lag length to avoid data leakage
    df["lag_24"]  = df["PJME_MW"].shift(24)    # same hour, 1 day ago
    df["lag_168"] = df["PJME_MW"].shift(168)   # same hour, 1 week ago

    # Rolling mean features — window ends lag-1 period before current row
    df["roll_24"]  = df["PJME_MW"].shift(1).rolling(window=24).mean()
    df["roll_168"] = df["PJME_MW"].shift(1).rolling(window=168).mean()

    df = df.dropna().reset_index(drop=True)
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
    Overlay predicted vs actual consumption for the first *n* test-set rows.
    A line plot over a slice of the test period reveals temporal alignment.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.array(y_true)[:n],  label="Actual",    color="steelblue",  linewidth=1.2)
    ax.plot(np.array(y_pred)[:n],  label="Predicted", color="tomato",     linewidth=1.0, alpha=0.85)
    ax.set_title(f"Predicted vs Actual — {name} (first {n} test rows)")
    ax.set_xlabel("Hour index (test set)")
    ax.set_ylabel("PJME_MW")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, name):
    """Side-by-side residuals-vs-predicted scatter and residual histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    residuals = np.array(y_true) - np.array(y_pred)
    axes[0].scatter(np.array(y_pred), residuals, alpha=0.3, s=8, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted (MW)")
    axes[0].set_ylabel("Residual (MW)")
    axes[0].set_title(f"Residuals vs Predicted — {name}")
    axes[1].hist(residuals, bins=50, color="seagreen", edgecolor="black")
    axes[1].set_xlabel("Residual (MW)")
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
