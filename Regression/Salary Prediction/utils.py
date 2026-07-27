"""
Salary Prediction Based on Experience - Utility Functions
Regression helpers for a tiny 30-row dataset (YearsExperience -> Salary).
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

def load_data(filepath="data/salary.csv"):
    """Load the salary-vs-experience dataset (30 rows)."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """
    Engineer:
      - exp_squared      (captures any non-linearity)
      - exp_log          (captures saturating returns)
      - experience_band  (entry / mid / senior)
    """
    df_feat = df.copy()
    df_feat["exp_squared"] = df_feat["YearsExperience"] ** 2
    df_feat["exp_log"] = np.log1p(df_feat["YearsExperience"])
    df_feat["experience_band"] = pd.cut(
        df_feat["YearsExperience"],
        bins=[0, 3, 7, 100],
        labels=["entry", "mid", "senior"],
    )
    return df_feat


def preprocess_data(df):
    """Full pipeline: engineer features, one-hot encode bands."""
    df_feat = create_features(df)
    df_encoded = pd.get_dummies(df_feat, columns=["experience_band"], drop_first=True)
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
        "R2":   r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
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
    ax.scatter(y_true, y_pred, alpha=0.7, color="steelblue", s=40)
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
    ax.scatter(y_pred, residuals, alpha=0.7, color="salmon", s=40)
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
