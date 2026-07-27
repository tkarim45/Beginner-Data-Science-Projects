"""
Employee Attrition Prediction - Utility Functions
Reusable helpers for the IBM HR Analytics Employee Attrition dataset.
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


def load_data(filepath="data/employee_attrition.csv"):
    """Load IBM HR data and binarize the `Attrition` target (Yes=1, No=0)."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    # Drop constant / id columns
    drop_cols = [c for c in ("EmployeeCount", "Over18", "StandardHours", "EmployeeNumber") if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def create_features(df):
    """Engineer tenure ratios, satisfaction composites, and salary buckets."""
    df = df.copy()

    df["TenureRatio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    df["YearsPerCompany"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    df["PromotionGap"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

    df["SatisfactionScore"] = (
        df["JobSatisfaction"]
        + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"]
        + df["WorkLifeBalance"]
    )
    df["IncomePerLevel"] = df["MonthlyIncome"] / (df["JobLevel"] + 0.0)

    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 25, 35, 45, 55, 100],
        labels=["Under25", "25-35", "35-45", "45-55", "55+"]
    ).astype(str)
    df["IncomeBin"] = pd.qcut(df["MonthlyIncome"], q=4, labels=["Low", "Mid", "High", "VeryHigh"]).astype(str)

    return df


def preprocess_data(df):
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


def plot_confusion_matrix(y_true, y_pred, model_name, labels=("Stay", "Leave"), ax=None):
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
