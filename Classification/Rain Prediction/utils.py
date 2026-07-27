"""
Rain Prediction - Utility Functions
weatherAUS dataset. Target: RainTomorrow (Yes/No).
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


def load_data(filepath="data/weatherAUS.csv"):
    """Load weatherAUS, parse Date, drop columns with > 30% missing, drop rows with NaN target."""
    df = pd.read_csv(filepath, na_values=["NA"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # Drop columns with too many NaNs (e.g. Evaporation, Sunshine, Cloud9am, Cloud3pm)
    high_missing = [c for c in df.columns if df[c].isnull().mean() > 0.30]
    df = df.drop(columns=high_missing + ["Date"])

    # Drop rows with NaN target
    df = df.dropna(subset=["RainTomorrow"])
    df["RainTomorrow"] = (df["RainTomorrow"] == "Yes").astype(int)
    if "RainToday" in df.columns:
        df["RainToday"] = (df["RainToday"] == "Yes").astype(int)

    return df.reset_index(drop=True)


def impute_missing(df):
    """Median for numerics, mode for categoricals."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype.kind in "fi":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
    return df


def create_features(df):
    """Engineer temperature ranges, humidity composites, season indicators."""
    df = df.copy()

    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    df["AvgHumidity"] = (df["Humidity9am"] + df["Humidity3pm"]) / 2
    df["AvgPressure"] = (df["Pressure9am"] + df["Pressure3pm"]) / 2
    df["AvgWindSpeed"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2
    df["TempChange"] = df["Temp3pm"] - df["Temp9am"]
    df["PressureChange"] = df["Pressure3pm"] - df["Pressure9am"]
    df["HumidityChange"] = df["Humidity3pm"] - df["Humidity9am"]

    if "Month" in df.columns:
        # Australia: Dec-Feb summer; Mar-May autumn; Jun-Aug winter; Sep-Nov spring
        season_map = {12: "Summer", 1: "Summer", 2: "Summer",
                      3: "Autumn", 4: "Autumn", 5: "Autumn",
                      6: "Winter", 7: "Winter", 8: "Winter",
                      9: "Spring", 10: "Spring", 11: "Spring"}
        df["Season"] = df["Month"].map(season_map)

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


def plot_confusion_matrix(y_true, y_pred, model_name, labels=("No Rain", "Rain"), ax=None):
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
