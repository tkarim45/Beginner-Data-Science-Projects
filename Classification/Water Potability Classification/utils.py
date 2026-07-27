"""
Water Potability Classification - Utility Functions
Predict drinkability (Potability) from 9 water quality measurements.
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


def load_data(filepath="data/water_potability.csv"):
    return pd.read_csv(filepath)


def impute_missing(df):
    """Median-imputation for ph, Sulfate, Trihalomethanes (the three columns with NaNs)."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0 and df[col].dtype != object:
            df[col] = df[col].fillna(df[col].median())
    return df


def create_features(df):
    """Engineer water-quality composite scores."""
    df = df.copy()

    df["pH_Category"] = pd.cut(
        df["ph"], bins=[0, 6.5, 7.5, 14], labels=["Acidic", "Neutral", "Alkaline"]
    ).astype(str)
    df["Hardness_Category"] = pd.cut(
        df["Hardness"], bins=[0, 60, 120, 180, 1000],
        labels=["Soft", "Moderate", "Hard", "VeryHard"]
    ).astype(str)
    df["Solids_Category"] = pd.qcut(
        df["Solids"], q=4, labels=["Low", "Mid", "High", "VeryHigh"]
    ).astype(str)

    # Composite chemistry score
    df["TotalChemicals"] = df["Chloramines"] + df["Sulfate"] / 100 + df["Trihalomethanes"] / 10
    df["pH_Hardness_Product"] = df["ph"] * df["Hardness"] / 1000

    return df


def preprocess_data(df):
    df = impute_missing(df)
    df = create_features(df)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def evaluate_model(model_name, y_true, y_pred):
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n{'='*40}\n  {model_name}\n{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:12s}: {v:.4f}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, labels=("Not Potable", "Potable"), ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
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
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.show()


def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"  CV F1 Scores : {scores.round(4)}")
    print(f"  Mean F1      : {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    df_results = pd.DataFrame(results_list)
    return df_results.sort_values("F1 Score", ascending=False).reset_index(drop=True)
