"""
Anomaly Detection in Transactions — Utility Functions
Unsupervised anomaly detection on credit-card transactions (ULB dataset).
Models are fit WITHOUT labels; the true `Class` column is used only afterwards
to evaluate how well the unsupervised anomaly scores rank actual fraud.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data(filepath="data/creditcard.csv"):
    """Load the credit-card transaction CSV."""
    return pd.read_csv(filepath)


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess(df):
    """
    Split features / label and standardise.
      - X = all columns except `Class` (28 PCA features V1..V28 + Time + Amount)
      - Time and Amount are on very different scales to the PCA features, so we
        StandardScaler the whole matrix (distance / margin based detectors need it).
    Returns: X_scaled (np.array), y (np.array of true labels), scaler, feature names.
    NOTE: y is returned for EVALUATION ONLY — it is never passed to the detectors.
    """
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    y = df["Class"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, feature_cols


# ─────────────────────────────────────────────
# Anomaly Models
# ─────────────────────────────────────────────

def score_isolation_forest(X, contamination, random_state=42):
    """
    Isolation Forest. Returns an anomaly score per row where HIGHER = more anomalous.
    (score_samples gives higher=normal, so we negate.)
    Also returns the binary anomaly flag from .predict (-1 = anomaly).
    """
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    model.fit(X)
    anomaly_score = -model.score_samples(X)
    pred = (model.predict(X) == -1).astype(int)
    return anomaly_score, pred, model


def score_lof(X, contamination, n_neighbors=20):
    """
    Local Outlier Factor (unsupervised, fit_predict mode).
    negative_outlier_factor_ is lower for outliers, so anomaly score = -that.
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    pred = (model.fit_predict(X) == -1).astype(int)
    anomaly_score = -model.negative_outlier_factor_
    return anomaly_score, pred, model


def score_ocsvm(X, contamination, fit_sample=5000, random_state=42):
    """
    One-Class SVM (RBF). O(n^2) to fit, so we fit on a random subsample of `fit_sample`
    rows and score all rows. decision_function gives higher=normal → negate.
    """
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), min(fit_sample, len(X)), replace=False)
    model = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")
    model.fit(X[idx])
    anomaly_score = -model.decision_function(X)
    pred = (model.predict(X) == -1).astype(int)
    return anomaly_score, pred, model


# ─────────────────────────────────────────────
# Evaluation (labels used here ONLY)
# ─────────────────────────────────────────────

def evaluate_scores(name, anomaly_score, pred, y):
    """
    Score-based metrics (ROC-AUC, Average Precision = PR-AUC) measure how well the
    continuous anomaly score ranks true fraud. Flag-based metrics (precision, recall,
    F1) measure the model's own -1/+1 decision at its contamination threshold.
    precision@k / recall@k use the top-k highest-scored rows, k = number of true frauds.
    """
    k = int(y.sum())
    top_k_idx = np.argsort(anomaly_score)[::-1][:k]
    tp_at_k = int(y[top_k_idx].sum())

    flagged = pred.sum()
    tp = int(((pred == 1) & (y == 1)).sum())
    precision = tp / flagged if flagged else 0.0
    recall = tp / y.sum() if y.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y, anomaly_score), 4),
        "pr_auc": round(average_precision_score(y, anomaly_score), 4),
        "precision@k": round(tp_at_k / k, 4),
        "recall@k": round(tp_at_k / k, 4),
        "flag_precision": round(precision, 4),
        "flag_recall": round(recall, 4),
        "flag_f1": round(f1, 4),
        "n_flagged": int(flagged),
    }


def run_all_models(X, y, contamination):
    """Fit all three detectors and return a comparison DataFrame, plus scores dict."""
    results, scores = [], {}
    for name, fn in [
        ("Isolation Forest", lambda: score_isolation_forest(X, contamination)),
        ("Local Outlier Factor", lambda: score_lof(X, contamination)),
        ("One-Class SVM", lambda: score_ocsvm(X, contamination)),
    ]:
        s, pred, _ = fn()
        scores[name] = s
        results.append(evaluate_scores(name, s, pred, y))
    return pd.DataFrame(results), scores


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_pr_curves(scores, y, ax=None):
    """Precision-Recall curve per model (the right view under heavy class imbalance)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    base = y.mean()
    ax.axhline(base, ls="--", color="gray", label=f"random (AP={base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves"); ax.legend()
    return ax
