"""
Stroke Prediction - Utility Functions
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

def load_data(filepath="data/stroke.csv"):
    """Load the healthcare stroke prediction dataset."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def drop_id(df):
    """Drop the patient id column."""
    return df.drop(columns=["id"], errors="ignore")


def drop_rare_gender(df):
    """A single 'Other' row in `gender` makes one-hot encoding fragile; drop it."""
    if "gender" in df.columns:
        df = df[df["gender"] != "Other"].copy()
    return df


def impute_bmi(df, strategy="median"):
    """The dataset has ~201 missing BMI values; impute with the median by default."""
    df_clean = df.copy()
    if df_clean["bmi"].isnull().any():
        if strategy == "median":
            df_clean["bmi"] = df_clean["bmi"].fillna(df_clean["bmi"].median())
        elif strategy == "mean":
            df_clean["bmi"] = df_clean["bmi"].fillna(df_clean["bmi"].mean())
    return df_clean


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """Engineer age groups, BMI categories, glucose categories."""
    df_feat = df.copy()

    df_feat["age_group"] = pd.cut(
        df_feat["age"],
        bins=[-1, 18, 35, 50, 65, 120],
        labels=["child", "young_adult", "middle_aged", "senior", "elderly"],
    )

    df_feat["bmi_category"] = pd.cut(
        df_feat["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    )

    df_feat["glucose_category"] = pd.cut(
        df_feat["avg_glucose_level"],
        bins=[0, 100, 125, 1000],
        labels=["normal", "prediabetic", "diabetic"],
    )

    # Binary "high-risk" composite (any of: hypertension, heart_disease, diabetic glucose)
    df_feat["high_risk_flag"] = (
        (df_feat["hypertension"] == 1)
        | (df_feat["heart_disease"] == 1)
        | (df_feat["glucose_category"] == "diabetic")
    ).astype(int)

    return df_feat


def preprocess_data(df):
    """Full pipeline: drop id, drop 'Other' gender, impute bmi, engineer, one-hot encode."""
    df_clean = drop_id(df)
    df_clean = drop_rare_gender(df_clean)
    df_clean = impute_bmi(df_clean, strategy="median")
    df_clean = create_features(df_clean)

    cat_cols = [
        "gender", "ever_married", "work_type", "residence_type", "smoking_status",
        "age_group", "bmi_category", "glucose_category",
    ]
    cat_cols = [c for c in cat_cols if c in df_clean.columns]
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
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
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"],
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
