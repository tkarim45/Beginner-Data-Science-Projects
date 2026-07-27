"""
Thyroid Disease Classification - Utility Functions
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

def load_data(filepath="data/thyroid.csv"):
    """Load the UCI ann-thyroid combined dataset (3 classes; we binarize)."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def binarize_target(df):
    """
    The original `class` column is 3-valued:
      1 = hyperthyroid, 2 = hypothyroid, 3 = normal.
    We binarize it: 1 = diseased (hyper or hypo), 0 = normal.
    """
    df_clean = df.copy()
    df_clean["diseased"] = (df_clean["class"] != 3).astype(int)
    return df_clean


def drop_original_class(df):
    """Drop the original 3-class label once binarized."""
    return df.drop(columns=["class"], errors="ignore")


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """
    Engineer simple summary features from the lab measurements:
      - any_query: 1 if any query/treatment flag is set
      - any_treatment: 1 if any active medication / treatment flag is set
      - lab_TSH_high: TSH > 0.05 (normalized scale) — heuristic flag
    """
    df_feat = df.copy()
    query_cols = ["query_on_thyroxine", "query_hypothyroid", "query_hyperthyroid"]
    treat_cols = ["on_thyroxine", "on_antithyroid_medication", "I131_treatment", "lithium"]

    df_feat["any_query"] = df_feat[query_cols].sum(axis=1).clip(upper=1).astype(int)
    df_feat["any_treatment"] = df_feat[treat_cols].sum(axis=1).clip(upper=1).astype(int)
    df_feat["lab_TSH_high"] = (df_feat["TSH"] > 0.05).astype(int)
    return df_feat


def preprocess_data(df):
    """Full preprocessing pipeline: binarize target, drop original class, engineer features."""
    df_clean = binarize_target(df)
    df_clean = drop_original_class(df_clean)
    df_clean = create_features(df_clean)
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
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
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
        xticklabels=["Normal", "Diseased"],
        yticklabels=["Normal", "Diseased"],
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
