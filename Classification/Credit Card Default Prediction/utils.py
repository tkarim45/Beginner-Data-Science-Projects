"""
Credit Card Default Prediction - Utility Functions
Reusable helpers for loading, preprocessing, and evaluating models on the
UCI "Default of Credit Card Clients" dataset.
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

def load_data(filepath="data/credit_card_default.csv"):
    """Load the UCI Credit Card Default dataset and rename target to 'default'."""
    df = pd.read_csv(filepath)
    if "default.payment.next.month" in df.columns:
        df = df.rename(columns={"default.payment.next.month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def clean_categories(df):
    """Map invalid EDUCATION (0,5,6) and MARRIAGE (0) codes to 'Other'."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """Engineer aggregated and ratio features from billing/payment history."""
    df = df.copy()

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]

    df["TOTAL_BILL"] = df[bill_cols].sum(axis=1)
    df["AVG_BILL"] = df[bill_cols].mean(axis=1)
    df["TOTAL_PAYMENT"] = df[pay_amt_cols].sum(axis=1)
    df["AVG_PAYMENT"] = df[pay_amt_cols].mean(axis=1)
    df["BILL_TO_LIMIT_RATIO"] = (df["AVG_BILL"] / df["LIMIT_BAL"]).clip(-5, 5)
    df["PAYMENT_TO_BILL_RATIO"] = (df["TOTAL_PAYMENT"] / df["TOTAL_BILL"].replace(0, np.nan)).fillna(0).clip(-5, 5)

    df["AVG_PAY_STATUS"] = df[pay_status_cols].mean(axis=1)
    df["MAX_PAY_STATUS"] = df[pay_status_cols].max(axis=1)
    df["NUM_DELAYED_MONTHS"] = (df[pay_status_cols] >= 1).sum(axis=1)

    df["AGE_GROUP"] = pd.cut(
        df["AGE"], bins=[0, 30, 40, 50, 60, 100],
        labels=["20s", "30s", "40s", "50s", "60+"]
    ).astype(str)
    df["LIMIT_BIN"] = pd.qcut(df["LIMIT_BAL"], q=4, labels=["Low", "Mid", "High", "VeryHigh"]).astype(str)

    return df


def preprocess_data(df):
    """Full preprocessing pipeline: clean categories, engineer features, encode."""
    df = clean_categories(df)
    df = create_features(df)
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE", "AGE_GROUP", "LIMIT_BIN"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


# ──────────────────────────────────────────────
# Model Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model_name, y_true, y_pred):
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


def plot_confusion_matrix(y_true, y_pred, model_name, labels=("No Default", "Default"), ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    return ax


def plot_roc_curves(models_dict, X_test, y_test):
    plt.figure(figsize=(10, 7))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.show()


def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"  CV F1 Scores : {scores.round(4)}")
    print(f"  Mean F1      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    df_results = pd.DataFrame(results_list)
    return df_results.sort_values("F1 Score", ascending=False).reset_index(drop=True)
