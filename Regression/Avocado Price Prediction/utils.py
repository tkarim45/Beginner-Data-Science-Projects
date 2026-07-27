"""
Avocado Price Prediction - Utility Functions
Regression helpers for the Hass Avocado Board weekly retail dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath="data/avocado.csv"):
    """Load the Hass Avocado Board weekly dataset (18,249 rows)."""
    df = pd.read_csv(filepath)
    return df


# ──────────────────────────────────────────────
# Data Cleaning / Feature Engineering
# ──────────────────────────────────────────────

def drop_index_column(df):
    """Drop the leftover unnamed index column from the source CSV."""
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def parse_date(df):
    """Convert Date to datetime and extract month / week-of-year."""
    df_clean = df.copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean["month"] = df_clean["Date"].dt.month
    df_clean["week"] = df_clean["Date"].dt.isocalendar().week.astype(int)
    return df_clean


def rename_plu(df):
    """Rename the PLU-code columns to readable names."""
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={
        "Total Volume": "total_volume",
        "4046": "small_hass",
        "4225": "large_hass",
        "4770": "xl_hass",
        "Total Bags": "total_bags",
        "Small Bags": "small_bags",
        "Large Bags": "large_bags",
        "XLarge Bags": "xl_bags",
        "AveragePrice": "AveragePrice",
        "Date": "Date",
        "type": "type",
        "year": "year",
        "region": "region",
    })
    return df_clean


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """
    Engineer log volumes (heavily right-skewed) and bag-share ratios.
    """
    df_feat = df.copy()
    df_feat["log_total_volume"] = np.log1p(df_feat["total_volume"])
    df_feat["log_total_bags"]   = np.log1p(df_feat["total_bags"])

    # Share of small/large/xl out of total Hass volume
    plu_total = df_feat[["small_hass","large_hass","xl_hass"]].sum(axis=1).replace(0, np.nan)
    df_feat["small_share"] = df_feat["small_hass"] / plu_total
    df_feat["large_share"] = df_feat["large_hass"] / plu_total
    df_feat["xl_share"]    = df_feat["xl_hass"]    / plu_total

    return df_feat


def preprocess_data(df):
    """Full preprocessing pipeline."""
    df_clean = drop_index_column(df)
    df_clean = parse_date(df_clean)
    df_clean = rename_plu(df_clean)
    df_clean = create_features(df_clean)
    df_clean = df_clean.drop(columns=["Date"])

    cat_cols = ["type", "region"]
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
    return df_encoded


# ──────────────────────────────────────────────
# Model Evaluation (Regression)
# ──────────────────────────────────────────────

def evaluate_model(model_name, y_true, y_pred):
    metrics = {
        "Model": model_name,
        "R2":    r2_score(y_true, y_pred),
        "RMSE":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":   mean_absolute_error(y_true, y_pred),
        "MAPE":  mean_absolute_percentage_error(y_true, y_pred),
    }
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:6s}: {v:.4f}")
    return metrics


def plot_actual_vs_predicted(y_true, y_pred, model_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, color="steelblue", s=10)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    return ax


def plot_residuals(y_true, y_pred, model_name, ax=None):
    residuals = y_true - y_pred
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, color="salmon", s=10)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
    ax.set_title(f"Residuals — {model_name}")
    return ax


def cross_validate_model(model, X, y, cv=5, scoring="r2"):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"  CV {scoring} : {scores.round(4)}")
    print(f"  Mean      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values("R2", ascending=False).reset_index(drop=True)
    return df_results
