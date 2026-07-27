"""
Flight Price Prediction - Utility Functions
Indian flight fare dataset (~300k rows). Target: price (continuous, INR).
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


def load_data(filepath="data/flights.csv", sample_size=50_000, random_state=42):
    """Load flight prices. Subsample to make training tractable for KNN/SVR-like models."""
    df = pd.read_csv(filepath)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    if "flight" in df.columns:
        df = df.drop(columns=["flight"])  # high-cardinality, not predictive
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    return df


def create_features(df):
    """Engineer time-of-day buckets, ordinal stops, log-duration."""
    df = df.copy()

    if "stops" in df.columns:
        # 'zero', 'one', 'two_or_more' → 0, 1, 2
        stop_map = {"zero": 0, "one": 1, "two_or_more": 2}
        df["stops"] = df["stops"].map(stop_map).fillna(df["stops"])

    if "duration" in df.columns:
        df["log_duration"] = np.log1p(df["duration"])

    return df


def preprocess_data(df):
    df = create_features(df)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
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
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }
    print(f"\n{'='*40}\n  {model_name}\n{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:6s}: {v:.4f}")
    return metrics


def plot_pred_vs_actual(y_true, y_pred, model_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual — {model_name}"); ax.legend()
    return ax


def plot_residuals(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    residuals = np.array(y_true) - np.array(y_pred)
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color="steelblue")
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
