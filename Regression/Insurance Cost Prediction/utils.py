"""
Insurance Cost Prediction - Utility Functions
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

def load_data(filepath="data/insurance.csv"):
    """Load the medical-insurance charges dataset (1,338 records)."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning / Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """
    Engineer chemistry-aware predictors:
      - bmi_category    (underweight / normal / overweight / obese)
      - age_group       (young / middle / senior)
      - smoker_obese    (1 if smoker AND obese — known to multiply charges)
      - charges_log     (log transform of the target if present — handled in preprocess)
    """
    df_feat = df.copy()
    df_feat["bmi_category"] = pd.cut(
        df_feat["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    )
    df_feat["age_group"] = pd.cut(
        df_feat["age"], bins=[0, 30, 50, 100],
        labels=["young", "middle", "senior"],
    )
    if "smoker" in df_feat.columns:
        df_feat["smoker_obese"] = (
            (df_feat["smoker"] == "yes") & (df_feat["bmi"] >= 30)
        ).astype(int)
    return df_feat


def preprocess_data(df):
    """Full preprocessing pipeline: engineer features + one-hot encode categoricals."""
    df_feat = create_features(df)
    cat_cols = ["sex", "smoker", "region", "bmi_category", "age_group"]
    cat_cols = [c for c in cat_cols if c in df_feat.columns]
    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)
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
    """Scatter of actual vs predicted; perfect predictions lie on the diagonal."""
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
    """Residual scatter (residual vs predicted)."""
    residuals = y_true - y_pred
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="salmon", s=18)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
    ax.set_title(f"Residuals — {model_name}")
    return ax


def cross_validate_model(model, X, y, cv=5, scoring="r2"):
    """Run k-fold CV for regression. `scoring='r2'` (default) or 'neg_root_mean_squared_error'."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"  CV {scoring} : {scores.round(4)}")
    print(f"  Mean      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    """Create a sorted comparison DataFrame from a list of metric dicts (sorted by R^2 desc)."""
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values("R2", ascending=False).reset_index(drop=True)
    return df_results
