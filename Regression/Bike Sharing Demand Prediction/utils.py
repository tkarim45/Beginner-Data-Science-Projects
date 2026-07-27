"""
Bike Sharing Demand Prediction - Utility Functions
UCI Bike Sharing hourly dataset. Target: cnt (total rentals per hour).
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
from sklearn.model_selection import cross_val_score


def load_data(filepath="data/hour.csv"):
    """Load bike sharing hourly data; drop instant + dteday + leakage columns."""
    df = pd.read_csv(filepath)
    # casual + registered = cnt → drop both to avoid leakage
    drop_cols = [c for c in ("instant", "dteday", "casual", "registered") if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def create_features(df):
    """Engineer cyclic time features and rush-hour / weekend flags."""
    df = df.copy()

    if "hr" in df.columns:
        df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
        df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
        df["IsRushHour"] = ((df["hr"].between(7, 9)) | (df["hr"].between(17, 19))).astype(int)
        df["IsNight"] = ((df["hr"] >= 22) | (df["hr"] <= 5)).astype(int)
    if "mnth" in df.columns:
        df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
        df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    if "weekday" in df.columns:
        df["IsWeekend"] = df["weekday"].isin([0, 6]).astype(int)

    # Comfort index (warm + dry)
    if {"temp", "hum", "windspeed"}.issubset(df.columns):
        df["ComfortIndex"] = df["temp"] - 0.5 * df["hum"] - 0.3 * df["windspeed"]

    return df


def preprocess_data(df):
    df = create_features(df)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def evaluate_model(model_name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "Model": model_name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, np.maximum(y_pred, 1))),
    }
    print(f"\n{'='*40}\n  {model_name}\n{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:6s}: {v:.4f}")
    return metrics


def plot_pred_vs_actual(y_true, y_pred, model_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual — {model_name}"); ax.legend()
    return ax


def plot_residuals(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    residuals = np.array(y_true) - np.array(y_pred)
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs Predicted — {model_name}")
    axes[1].hist(residuals, bins=40, color="seagreen", edgecolor="black")
    axes[1].set_xlabel("Residual"); axes[1].set_title("Residual distribution")
    plt.tight_layout()
    return fig


def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"  CV R² Scores : {scores.round(4)}")
    print(f"  Mean R²      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    df_results = pd.DataFrame(results_list)
    return df_results.sort_values("R2", ascending=False).reset_index(drop=True)
