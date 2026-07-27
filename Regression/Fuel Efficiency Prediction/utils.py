"""
Fuel Efficiency Prediction (Auto MPG) - Utility Functions
Regression helpers for data loading, preprocessing, and model evaluation.
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

def load_data(filepath="data/auto_mpg.csv"):
    """Load the UCI Auto MPG dataset (398 cars, 1970-1982 model years)."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning / Feature Engineering
# ──────────────────────────────────────────────

def drop_car_name(df):
    """Drop the human-readable car name (keeping it would leak make/model)."""
    return df.drop(columns=["car_name"], errors="ignore")


def impute_horsepower(df, strategy="median"):
    """Auto MPG has 6 missing horsepower values; impute with median by default."""
    df_clean = df.copy()
    if df_clean["horsepower"].isnull().any():
        if strategy == "median":
            df_clean["horsepower"] = df_clean["horsepower"].fillna(df_clean["horsepower"].median())
        elif strategy == "mean":
            df_clean["horsepower"] = df_clean["horsepower"].fillna(df_clean["horsepower"].mean())
    return df_clean


def map_origin(df):
    """Map numeric origin codes to human-readable region labels."""
    df_clean = df.copy()
    df_clean["origin"] = df_clean["origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
    return df_clean


def create_features(df):
    """
    Engineer:
      - power_to_weight = horsepower / weight (×1000) — efficiency proxy
      - displacement_per_cyl = displacement / cylinders
      - is_v8 (1 if cylinders == 8)
      - decade-of-build (70s vs 80s)
    """
    df_feat = df.copy()
    df_feat["power_to_weight"] = df_feat["horsepower"] / df_feat["weight"] * 1000
    df_feat["displacement_per_cyl"] = df_feat["displacement"] / df_feat["cylinders"]
    df_feat["is_v8"] = (df_feat["cylinders"] == 8).astype(int)
    df_feat["decade"] = (df_feat["model_year"] // 10).astype(int) * 10
    return df_feat


def preprocess_data(df):
    """Full preprocessing pipeline."""
    df_clean = drop_car_name(df)
    df_clean = impute_horsepower(df_clean)
    df_clean = map_origin(df_clean)
    df_clean = create_features(df_clean)

    cat_cols = ["origin"]
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    return df_encoded


# ──────────────────────────────────────────────
# Model Evaluation (Regression)
# ──────────────────────────────────────────────

def evaluate_model(model_name, y_true, y_pred):
    """Print and return regression metrics."""
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
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", s=18)
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
    ax.scatter(y_pred, residuals, alpha=0.5, color="salmon", s=18)
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
