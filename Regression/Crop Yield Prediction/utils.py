"""
Crop Yield Prediction - Utility Functions
FAO/Kaggle crop yield dataset. Target: hg/ha_yield (hectograms per hectare).
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


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data(filepath="data/yield_df.csv"):
    """Load crop yield CSV; drop the unnamed row-index column if present."""
    df = pd.read_csv(filepath)
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


# ─────────────────────────────────────────────
# Data Cleaning
# ─────────────────────────────────────────────

def drop_duplicates(df):
    """Remove exact duplicate rows and reset the index."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Dropped {before - len(df)} duplicate rows (kept {len(df)})")
    return df


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_data(df):
    """Full pipeline: drop duplicates, one-hot encode Area and Item (drop_first=True),
    cast bool columns to int, and return the processed dataframe."""
    df = df.copy()
    df = drop_duplicates(df)
    cat_cols = ["Area", "Item"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


# ─────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────

def evaluate_model(name, y_true, y_pred):
    """Print and return a dict of R², MAE, RMSE, and MAPE for a regression model."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "Model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }
    print(f"\n{'='*40}\n  {name}\n{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:6s}: {v:.4f}")
    return metrics


def plot_predictions(y_true, y_pred, name):
    """Predicted-vs-actual scatter with a diagonal perfect-prediction reference line."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color="steelblue")
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect")
    ax.set_xlabel("Actual Yield (hg/ha)")
    ax.set_ylabel("Predicted Yield (hg/ha)")
    ax.set_title(f"Predicted vs Actual — {name}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, name):
    """Side-by-side residuals-vs-predicted scatter and residual distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    residuals = np.array(y_true) - np.array(y_pred)
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=12, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Yield (hg/ha)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs Predicted — {name}")
    axes[1].hist(residuals, bins=40, color="seagreen", edgecolor="black")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")
    plt.tight_layout()
    return fig


def cross_validate_model(model, X, y, cv=5):
    """Run k-fold cross-validation with R² scoring and print summary statistics."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"  CV R² Scores : {scores.round(4)}")
    print(f"  Mean R²      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    """Convert a list of evaluate_model dicts into a DataFrame sorted by R² descending."""
    df_results = pd.DataFrame(results_list)
    return df_results.sort_values("R2", ascending=False).reset_index(drop=True)
