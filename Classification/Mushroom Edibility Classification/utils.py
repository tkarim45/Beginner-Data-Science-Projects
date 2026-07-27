"""
Mushroom Edibility Classification - Utility Functions
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

def load_data(filepath="data/mushrooms.csv"):
    """Load the UCI mushroom dataset (all categorical, with '?' missing markers)."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def encode_target(df):
    """Encode class (e=edible, p=poisonous) as 0/1."""
    df_clean = df.copy()
    df_clean["class"] = df_clean["class"].map({"e": 0, "p": 1})
    return df_clean


def handle_missing(df, strategy="mode"):
    """
    The UCI mushroom dataset uses '?' to mark missing stalk_root values
    (~30% of rows). Replace '?' with the column mode by default.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == object and (df_clean[col] == "?").any():
            if strategy == "mode":
                mode = df_clean[col].mode()
                fill = mode.iloc[0] if not mode.empty else "missing"
                df_clean[col] = df_clean[col].replace("?", fill)
            elif strategy == "category":
                df_clean[col] = df_clean[col].replace("?", "missing")
    return df_clean


def drop_constant(df):
    """Drop columns with only a single unique value (e.g. veil_type in this dataset)."""
    constant = [c for c in df.columns if df[c].nunique() <= 1]
    if constant:
        df = df.drop(columns=constant)
    return df, constant


# ──────────────────────────────────────────────
# Feature Engineering / Preprocessing
# ──────────────────────────────────────────────

def preprocess_data(df, missing_strategy="category"):
    """
    Full preprocessing pipeline:
      1. Encode target (e/p -> 0/1)
      2. Replace '?' with a "missing" category (default) or mode
      3. Drop constant columns
      4. One-hot encode all remaining categorical features (drop_first=True)
    """
    df_clean = encode_target(df)
    df_clean = handle_missing(df_clean, strategy=missing_strategy)
    df_clean, _ = drop_constant(df_clean)

    cat_cols = [c for c in df_clean.columns if c != "class" and df_clean[c].dtype == object]
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    # Convert bool columns from get_dummies into int (cleaner for downstream)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
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
        xticklabels=["Edible", "Poisonous"],
        yticklabels=["Edible", "Poisonous"],
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
