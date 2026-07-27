"""
Plagiarism Detector — Utility Functions
Detect whether two texts are near-duplicates / paraphrases using similarity features
(TF-IDF cosine + character n-gram Jaccard). Evaluated on the MRPC paraphrase corpus,
where each sentence pair is labelled paraphrase (1) or not (0) — a proxy for plagiarism.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def load_pairs(path="data/pairs.csv"):
    return pd.read_csv(path)

def tfidf_cosine(a, b):
    """Cosine similarity of the TF-IDF vectors of two texts (word-level)."""
    v = TfidfVectorizer().fit([str(a), str(b)])
    m = v.transform([str(a), str(b)])
    return float(cosine_similarity(m[0], m[1])[0, 0])

def char_ngram_jaccard(a, b, n=3):
    """Jaccard overlap of character n-gram sets (catches copy-paste with edits)."""
    def grams(s): s=str(s).lower(); return {s[i:i+n] for i in range(len(s)-n+1)}
    A, Bs = grams(a), grams(b)
    return len(A & Bs)/len(A | Bs) if (A | Bs) else 0.0

def score_pairs(df):
    """Add tfidf_cosine + char_jaccard columns to a pairs DataFrame."""
    df = df.copy()
    df["tfidf_cosine"] = [tfidf_cosine(a, b) for a, b in zip(df.text1, df.text2)]
    df["char_jaccard"] = [char_ngram_jaccard(a, b) for a, b in zip(df.text1, df.text2)]
    return df

def best_threshold(df, score_col="tfidf_cosine", label_col="is_paraphrase"):
    """Sweep a decision threshold; return the one maximising F1 and its metrics."""
    best = {"f1": -1}
    for th in np.arange(0.05, 0.95, 0.025):
        pred = (df[score_col] >= th).astype(int)
        f1 = f1_score(df[label_col], pred)
        if f1 > best["f1"]:
            best = {"threshold": round(float(th), 3), "f1": round(f1, 4),
                    "accuracy": round(accuracy_score(df[label_col], pred), 4),
                    "precision": round(precision_score(df[label_col], pred), 4),
                    "recall": round(recall_score(df[label_col], pred), 4)}
    return best
