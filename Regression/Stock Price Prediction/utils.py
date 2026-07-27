"""
Stock Price Prediction - Utility Functions
S&P 500 daily OHLCV data. Single-stock focus: AAPL.
Target: next-day closing price (next_close).
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
#  Data Loading
# ─────────────────────────────────────────────

def load_data(filepath="data/all_stocks_5yr.csv"):
    """Load the full S&P 500 5-year OHLCV dataset."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


# ─────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────

def create_features(df):
    """
    Engineer time-series features for next-day closing-price prediction.

    Input df must be a single-stock price series sorted by date ascending.
    Features created:
      - lag_1, lag_2, lag_3, lag_5  : lagged closing prices
      - ma_5, ma_10, ma_20           : rolling mean closing prices
      - std_5                        : rolling 5-day std of close
      - daily_return                 : pct change in close (day-over-day)
      - volume_change                : pct change in volume
      - high_low_range               : high minus low (daily trading range)
      - open_close_range             : open minus close (intra-day movement)
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Lagged closing prices
    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_3"] = df["close"].shift(3)
    df["lag_5"] = df["close"].shift(5)

    # Rolling statistics (computed on past values only — shift(1) prevents leakage)
    df["ma_5"]  = df["close"].shift(1).rolling(window=5).mean()
    df["ma_10"] = df["close"].shift(1).rolling(window=10).mean()
    df["ma_20"] = df["close"].shift(1).rolling(window=20).mean()
    df["std_5"] = df["close"].shift(1).rolling(window=5).std()

    # Return and volume change
    df["daily_return"]  = df["close"].pct_change()
    df["volume_change"] = df["volume"].pct_change()

    # Range features (current day — these are known by end of trading day)
    df["high_low_range"]   = df["high"] - df["low"]
    df["open_close_range"] = df["open"] - df["close"]

    return df


# ─────────────────────────────────────────────
#  Model Evaluation
# ─────────────────────────────────────────────

def evaluate_model(name, y_true, y_pred):
    """Print and return a dict of R², MAE, RMSE, MAPE for a regression model."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "Model": name,
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2":   float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }
    print(f"\n{'='*40}\n  {name}\n{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:6s}: {v:.4f}")
    return metrics


def plot_predictions(y_true, y_pred, name):
    """
    Plot predicted vs actual closing price as overlaid line charts
    across the test period (chronological order assumed).
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(np.array(y_true), label="Actual",    color="steelblue",  linewidth=1.5)
    ax.plot(np.array(y_pred), label="Predicted", color="darkorange", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Test Period (trading days)")
    ax.set_ylabel("Closing Price (USD)")
    ax.set_title(f"Predicted vs Actual Close — {name}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, name):
    """Residual scatter (vs predicted) and residual distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    residuals = np.array(y_true) - np.array(y_pred)

    axes[0].scatter(np.array(y_pred), residuals, alpha=0.4, s=15, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs Predicted — {name}")

    axes[1].hist(residuals, bins=40, color="seagreen", edgecolor="black")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    return fig


def cross_validate_model(model, X, y, cv=5):
    """
    Time-series aware cross-validation using TimeSeriesSplit.
    Returns array of R² scores from each fold.
    """
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
    print(f"  CV R² Scores : {scores.round(4)}")
    print(f"  Mean R²      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    """Return a DataFrame of all model results sorted by R² descending."""
    df_results = pd.DataFrame(results_list)
    return df_results.sort_values("R2", ascending=False).reset_index(drop=True)
