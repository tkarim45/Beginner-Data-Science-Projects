"""
Diabetes Prediction - Utility Functions
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
    classification_report,
)
from sklearn.model_selection import cross_val_score


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath="data/diabetes.csv"):
    """Load the Pima Indians Diabetes dataset."""
    df = pd.read_csv(filepath)
    return df


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def replace_zeros_with_nan(df, columns):
    """Replace 0 values with NaN for columns where 0 is not a valid value."""
    df_clean = df.copy()
    for col in columns:
        df_clean[col] = df_clean[col].replace(0, np.nan)
    return df_clean


def impute_missing(df, strategy="median"):
    """Impute NaN values using median (default) or mean."""
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() > 0:
            if strategy == "median":
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
            elif strategy == "mean":
                df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
    return df_imputed


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """Engineer new features from existing columns."""
    df_feat = df.copy()

    # BMI Category
    df_feat["BMI_Category"] = pd.cut(
        df_feat["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    # Age Group
    df_feat["AgeGroup"] = pd.cut(
        df_feat["Age"],
        bins=[20, 30, 40, 50, 60, 90],
        labels=["20s", "30s", "40s", "50s", "60+"],
    )

    # Glucose Level Category
    df_feat["Glucose_Category"] = pd.cut(
        df_feat["Glucose"],
        bins=[0, 99, 126, 300],
        labels=["Normal", "Prediabetes", "Diabetes"],
    )

    # Insulin Level Category
    df_feat["Insulin_Category"] = pd.cut(
        df_feat["Insulin"],
        bins=[0, 16, 166, 900],
        labels=["Low", "Normal", "High"],
    )

    # Blood Pressure Category
    df_feat["BP_Category"] = pd.cut(
        df_feat["BloodPressure"],
        bins=[0, 80, 89, 200],
        labels=["Normal", "High_Stage1", "High_Stage2"],
    )

    return df_feat


def preprocess_data(df):
    """Full preprocessing pipeline: clean zeros, impute, engineer features, encode."""
    # Columns where 0 is biologically impossible
    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    df_clean = replace_zeros_with_nan(df, zero_invalid_cols)
    df_clean = impute_missing(df_clean, strategy="median")
    df_clean = create_features(df_clean)

    # One-hot encode categorical features
    categorical_cols = ["BMI_Category", "AgeGroup", "Glucose_Category", "Insulin_Category", "BP_Category"]
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    return df_encoded


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
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
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
    """Run k-fold cross-validation and return scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"  CV F1 Scores : {scores.round(4)}")
    print(f"  Mean F1      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    """Create a comparison DataFrame from a list of metric dicts."""
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return df_results
