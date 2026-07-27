"""
Wine Quality Classification - Utility Functions
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

def load_data(filepath="data/winequality-red.csv"):
    """Load the UCI red wine quality dataset."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def binarize_quality(df, threshold=7):
    """
    Convert the integer quality score (3-8) into a binary target:
      1 if quality >= threshold ("good wine"), else 0.
    """
    df_clean = df.copy()
    df_clean["good_quality"] = (df_clean["quality"] >= threshold).astype(int)
    return df_clean


def drop_quality(df):
    """Drop the original integer quality column once we have the binary target."""
    return df.drop(columns=["quality"], errors="ignore")


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """Engineer chemistry-aware ratios and bins for wine."""
    df_feat = df.copy()

    # Acidity ratios
    df_feat["total_acidity"] = df_feat["fixed_acidity"] + df_feat["volatile_acidity"]
    df_feat["acidity_ratio"] = df_feat["fixed_acidity"] / df_feat["volatile_acidity"].replace(0, np.nan)

    # Sulfur ratio (free / total)
    df_feat["free_to_total_sulfur"] = (
        df_feat["free_sulfur_dioxide"] / df_feat["total_sulfur_dioxide"].replace(0, np.nan)
    )

    # Alcohol bin
    df_feat["alcohol_level"] = pd.cut(
        df_feat["alcohol"],
        bins=[0, 9.5, 11, 14],
        labels=["low", "medium", "high"],
    )

    return df_feat


def preprocess_data(df, threshold=7):
    """Full preprocessing pipeline: binarize target, engineer features, encode."""
    df_clean = binarize_quality(df, threshold=threshold)
    df_clean = drop_quality(df_clean)
    df_clean = create_features(df_clean)

    # One-hot encode the alcohol_level bin (drop_first to avoid collinearity)
    df_encoded = pd.get_dummies(df_clean, columns=["alcohol_level"], drop_first=True)

    # Fill any NaNs created by ratio operations on zero denominators
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
    return df_encoded


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
        xticklabels=["Not Good", "Good"],
        yticklabels=["Not Good", "Good"],
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
