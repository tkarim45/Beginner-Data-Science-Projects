"""
Diamond Price Prediction - Utility Functions
Regression helpers for the classic ggplot2 diamonds dataset.
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


# Canonical ordinal orderings for the categorical features
CUT_ORDER     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER   = ["J", "I", "H", "G", "F", "E", "D"]      # D = best, J = worst -> we encode worst→best
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath="data/diamonds.csv"):
    """Load the ggplot2 diamonds dataset (53,940 rows)."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def drop_zero_dimensions(df):
    """A small fraction of rows have x/y/z == 0 — physically impossible. Drop them."""
    df_clean = df.copy()
    bad = (df_clean["x"] == 0) | (df_clean["y"] == 0) | (df_clean["z"] == 0)
    df_clean = df_clean[~bad].reset_index(drop=True)
    return df_clean


def encode_ordinals(df):
    """Encode cut, color, clarity as integers in the canonical worst→best order."""
    df_clean = df.copy()
    df_clean["cut"] = df_clean["cut"].map({k: i for i, k in enumerate(CUT_ORDER)})
    df_clean["color"] = df_clean["color"].map({k: i for i, k in enumerate(COLOR_ORDER)})
    df_clean["clarity"] = df_clean["clarity"].map({k: i for i, k in enumerate(CLARITY_ORDER)})
    return df_clean


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """
    Engineer:
      - volume       = x * y * z
      - density      = carat / volume   (carat is mass, volume from x,y,z)
      - price_log    (only used in EDA — not on the cleaned set)
      - log_carat
    """
    df_feat = df.copy()
    df_feat["volume"] = df_feat["x"] * df_feat["y"] * df_feat["z"]
    df_feat["density"] = df_feat["carat"] / df_feat["volume"].replace(0, np.nan)
    df_feat["log_carat"] = np.log1p(df_feat["carat"])
    return df_feat


def preprocess_data(df):
    """Full pipeline: drop zero-dim rows, ordinal-encode, engineer features."""
    df_clean = drop_zero_dimensions(df)
    df_clean = encode_ordinals(df_clean)
    df_clean = create_features(df_clean)
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    return df_clean


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
    ax.scatter(y_true, y_pred, alpha=0.3, color="steelblue", s=8)
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
    ax.scatter(y_pred, residuals, alpha=0.3, color="salmon", s=8)
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
