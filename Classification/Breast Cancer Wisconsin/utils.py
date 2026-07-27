"""
Breast Cancer Wisconsin - Utility Functions
Reusable helper functions for data loading, preprocessing, and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath="data/breast_cancer.csv"):
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset."""
    df = pd.read_csv(filepath)
    return df


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def encode_diagnosis(df):
    """Map the diagnosis column from M/B to 1/0."""
    df_clean = df.copy()
    df_clean["diagnosis"] = df_clean["diagnosis"].map({"M": 1, "B": 0})
    return df_clean


def drop_id(df):
    """Drop the patient id column — not a predictor."""
    return df.drop(columns=["id"], errors="ignore")


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """Engineer aggregate features from mean/se/worst groups."""
    df_feat = df.copy()
    base = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
            'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

    # Ratio of worst to mean — captures how extreme the worst region is
    for f in base:
        mean_col = f"{f}_mean"
        worst_col = f"{f}_worst"
        if mean_col in df_feat.columns and worst_col in df_feat.columns:
            df_feat[f"{f}_worst_to_mean"] = df_feat[worst_col] / df_feat[mean_col].replace(0, np.nan)

    # Size composite (radius * perimeter normalized)
    if {"radius_mean", "perimeter_mean"}.issubset(df_feat.columns):
        df_feat["size_composite"] = df_feat["radius_mean"] * df_feat["perimeter_mean"]

    return df_feat


def preprocess_data(df):
    """Full preprocessing pipeline: drop id, encode diagnosis, engineer features."""
    df_clean = drop_id(df)
    df_clean = encode_diagnosis(df_clean)
    df_clean = create_features(df_clean)

    # Fill any NaN created by ratios (extremely rare, only when mean is zero)
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    return df_clean


# ──────────────────────────────────────────────
# Model Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model_name, y_true, y_pred):
    """Print and return classification metrics."""
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:12s}: {v:.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, ax=None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    return ax


def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for multiple models on the same figure."""
    plt.figure(figsize=(10, 7))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def cross_validate_model(model, X, y, cv=5):
    """Run k-fold cross-validation and return F1 scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"  CV F1 Scores : {scores.round(4)}")
    print(f"  Mean F1      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    """Create a sorted comparison DataFrame from a list of metric dicts."""
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return df_results
