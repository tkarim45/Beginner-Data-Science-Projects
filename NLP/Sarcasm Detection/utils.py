"""
Sarcasm Detection - Utility Functions
Reusable helpers for loading data, cleaning text, and evaluating classifiers.
"""

import re
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
)

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:  # pragma: no cover - fallback if nltk data missing
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
    def _stem(word):
        return _STEMMER.stem(word)
except Exception:  # pragma: no cover
    def _stem(word):
        return word


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath="data/sarcasm_headlines.json"):
    """Load the sarcasm headlines JSON-lines dataset."""
    df = pd.read_json(filepath, lines=True)
    return df[['headline', 'is_sarcastic']]


# ──────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NONALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")

def clean_text(text, remove_stopwords=True, stem=True, min_token_len=2):
    """Lowercase, strip URLs/HTML/punctuation/digits, drop stopwords, Porter-stem."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NONALPHA_RE.sub(" ", text)
    tokens = _MULTISPACE_RE.sub(" ", text).strip().split()
    out = []
    for tok in tokens:
        if len(tok) < min_token_len:
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(_stem(tok) if stem else tok)
    return " ".join(out)


# ──────────────────────────────────────────────
# Model Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model_name, y_true, y_pred):
    """Compute weighted accuracy / precision / recall / F1 (works for binary & multi-class)."""
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    print(f"\n{'='*40}\n  {model_name}\n{'='*40}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:12s}: {v:.4f}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    return ax


def compare_models(results_list):
    df_results = pd.DataFrame(results_list)
    return df_results.sort_values("F1 Score", ascending=False).reset_index(drop=True)
