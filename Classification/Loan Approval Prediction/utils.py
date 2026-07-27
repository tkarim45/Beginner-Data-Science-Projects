"""
Loan Approval Prediction - Utility Functions
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

def load_data(filepath="data/loan.csv"):
    """Load the Analytics Vidhya / Dream Housing loan dataset."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def drop_id(df):
    """Drop Loan_ID — pure identifier, not a predictor."""
    return df.drop(columns=["Loan_ID"], errors="ignore")


def encode_target(df):
    """Encode Loan_Status: Y -> 1 (approved), N -> 0 (denied)."""
    df_clean = df.copy()
    df_clean["Loan_Status"] = df_clean["Loan_Status"].map({"Y": 1, "N": 0})
    return df_clean


def impute_missing(df):
    """
    Impute missing values:
      - Categorical columns -> mode
      - Numeric columns     -> median
      - Credit_History (which is binary 0/1 stored as float) -> mode
    """
    df_clean = df.copy()
    cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Property_Area", "Education"]
    for c in cat_cols:
        if c in df_clean.columns and df_clean[c].isnull().any():
            df_clean[c] = df_clean[c].fillna(df_clean[c].mode().iloc[0])

    if "Credit_History" in df_clean.columns and df_clean["Credit_History"].isnull().any():
        df_clean["Credit_History"] = df_clean["Credit_History"].fillna(
            df_clean["Credit_History"].mode().iloc[0]
        )

    num_cols = ["LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome"]
    for c in num_cols:
        if c in df_clean.columns and df_clean[c].isnull().any():
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    return df_clean


def normalize_dependents(df):
    """Convert '3+' string to integer 3 in Dependents."""
    df_clean = df.copy()
    if "Dependents" in df_clean.columns:
        df_clean["Dependents"] = (
            df_clean["Dependents"].astype(str).str.replace("+", "", regex=False)
        )
        df_clean["Dependents"] = pd.to_numeric(df_clean["Dependents"], errors="coerce").fillna(0).astype(int)
    return df_clean


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """Engineer income totals, EMI, balance income, log transforms for skew."""
    df_feat = df.copy()

    df_feat["TotalIncome"] = df_feat["ApplicantIncome"] + df_feat["CoapplicantIncome"]
    df_feat["EMI"] = df_feat["LoanAmount"] / df_feat["Loan_Amount_Term"].replace(0, np.nan)
    df_feat["BalanceIncome"] = df_feat["TotalIncome"] - df_feat["EMI"] * 1000  # LoanAmount is in thousands
    df_feat["LoanAmount_log"] = np.log1p(df_feat["LoanAmount"])
    df_feat["TotalIncome_log"] = np.log1p(df_feat["TotalIncome"])
    return df_feat


def preprocess_data(df):
    """Full pipeline: drop id, encode target, normalize dependents, impute, engineer, encode."""
    df_clean = drop_id(df)
    df_clean = encode_target(df_clean)
    df_clean = normalize_dependents(df_clean)
    df_clean = impute_missing(df_clean)
    df_clean = create_features(df_clean)

    cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
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
        xticklabels=["Denied", "Approved"],
        yticklabels=["Denied", "Approved"],
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
