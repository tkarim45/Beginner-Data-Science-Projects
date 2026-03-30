"""
Utility functions for the Titanic Survival Prediction project.
Provides reusable helpers for data loading, preprocessing, feature engineering,
and model evaluation.
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
    classification_report,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath="data/titanic.csv"):
    """Load the Titanic dataset and return a DataFrame."""
    df = pd.read_csv(filepath)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def extract_title(name):
    """Extract the title (Mr, Mrs, Miss, etc.) from a passenger name."""
    title = name.split(",")[1].split(".")[0].strip()
    return title


def create_features(df):
    """
    Engineer new features from the raw Titanic data.

    New columns added:
        - Title: extracted from Name
        - FamilySize: SibSp + Parch + 1
        - IsAlone: 1 if FamilySize == 1
        - AgeGroup: binned age into categories
        - FareBin: binned fare into quartile categories
    """
    df = df.copy()

    # Title from Name
    df["Title"] = df["Name"].apply(extract_title)
    # Group rare titles
    rare_titles = df["Title"].value_counts()
    rare_titles = rare_titles[rare_titles < 10].index.tolist()
    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age groups
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 80],
        labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
    )

    # Fare bins
    df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "Very High"])

    return df


def preprocess_data(df):
    """
    Clean and preprocess the Titanic DataFrame for modeling.

    Steps:
        1. Fill missing Age with median per (Pclass, Sex) group
        2. Fill missing Embarked with mode
        3. Fill missing Fare with median
        4. Drop Cabin (too many missing), Name, Ticket, PassengerId
        5. Encode categorical variables
    """
    df = df.copy()

    # Fill missing Age by group median
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill remaining missing Age (edge cases) with overall median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill Embarked with mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fill Fare with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Drop columns with too many missing values or not useful for modeling
    drop_cols = ["Cabin", "Name", "Ticket", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categorical columns
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)

    # Encode Title if present
    if "Title" in df.columns:
        df = pd.get_dummies(df, columns=["Title"], drop_first=True, dtype=int)

    # Encode AgeGroup and FareBin if present
    for col in ["AgeGroup", "FareBin"]:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)

    return df


# ── Model Evaluation ──────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Print classification metrics and return them as a dict.
    """
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1 Score:  {metrics['F1 Score']:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Plot a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for multiple models on the same figure.

    Parameters:
        models_dict: dict of {model_name: fitted_model}
        X_test: test features
        y_test: test labels
    """
    plt.figure(figsize=(8, 6))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def cross_validate_model(model, X, y, cv=5):
    """Run cross-validation and print mean/std accuracy."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(results_list):
    """
    Take a list of metric dicts from evaluate_model() and return
    a comparison DataFrame sorted by F1 Score.
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return df
