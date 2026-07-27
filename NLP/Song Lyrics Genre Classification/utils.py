"""
Song Lyrics Genre Classification — Utility Functions
TF-IDF + classic-ML text classification. Notebook 02 produces a `clean_text`
column; notebook 03 vectorizes it and trains several linear / Naive-Bayes models.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath="data/lyrics.csv"):
    """Load the raw dataset with `text` and `label` columns; drop empties."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NONALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text, remove_stopwords=True, min_token_len=2):
    """Lowercase, strip URLs/HTML/punctuation/digits, drop stopwords and short tokens."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NONALPHA_RE.sub(" ", text)
    tokens = _MULTISPACE_RE.sub(" ", text).strip().split()
    out = [t for t in tokens if len(t) >= min_token_len and not (remove_stopwords and t in STOPWORDS)]
    return " ".join(out)


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

def get_models():
    """Six fast classic-ML text classifiers (all efficient on sparse TF-IDF)."""
    return {
        "Multinomial NB": MultinomialNB(),
        "Complement NB": ComplementNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Ridge Classifier": RidgeClassifier(),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    }


# ──────────────────────────────────────────────
# Model Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model_name, y_true, y_pred):
    """Weighted accuracy / precision / recall / F1 (binary & multi-class)."""
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, labels=None, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    return ax


def compare_models(results_list):
    return pd.DataFrame(results_list).sort_values("F1 Score", ascending=False).reset_index(drop=True)
