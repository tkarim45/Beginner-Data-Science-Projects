"""
Credit Card Fraud Detection - Utility Functions
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
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath="data/creditcard.csv"):
    """Load the credit card fraud detection dataset."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def preprocess_data(df):
    """Scale Amount and Time with RobustScaler; V1..V28 are already PCA-scaled.

    RobustScaler is preferred for Amount because transaction amounts are
    highly right-skewed with many outliers — it uses the median and IQR
    instead of mean/std and is therefore robust to extreme values.
    Time is also scaled so all features live on a comparable scale.
    The 28 anonymized PCA components (V1..V28) are already standardized
    by the dataset creators and must NOT be re-scaled.
    """
    df_out = df.copy()
    scaler = RobustScaler()
    df_out[["Amount", "Time"]] = scaler.fit_transform(df_out[["Amount", "Time"]])
    return df_out


# ──────────────────────────────────────────────
# Model Evaluation
# ──────────────────────────────────────────────

def evaluate_model(name, y_true, y_pred, y_prob=None):
    """Print and return classification metrics with fraud as the positive class.

    Parameters
    ----------
    name   : str  — model label
    y_true : array-like of true binary labels (0 = legit, 1 = fraud)
    y_pred : array-like of predicted binary labels
    y_prob : array-like of predicted probabilities for the positive class (optional)

    Returns
    -------
    dict with keys: Model, Accuracy, Precision, Recall, F1, ROC-AUC
    """
    metrics = {
        "Model":     name,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan"),
    }

    print(f"\n{'='*42}")
    print(f"  {name}")
    print(f"{'='*42}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:12s}: {v:.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, name, labels=None):
    """Plot a labelled confusion matrix heatmap."""
    if labels is None:
        labels = ["Legit", "Fraud"]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    return ax


def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for multiple models on the same figure.

    Falls back to decision_function for models without predict_proba.
    """
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
    """Run k-fold cross-validation and return F1 scores (fraud = positive class)."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"  CV F1 Scores : {scores.round(4)}")
    print(f"  Mean F1      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    """Create a comparison DataFrame sorted by F1 descending."""
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values("F1", ascending=False).reset_index(drop=True)
    return df_results
