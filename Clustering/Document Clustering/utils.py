"""
Document Clustering — Utility Functions
K-Means clustering on TF-IDF text representations (20 Newsgroups).
This is an unsupervised project — true newsgroup labels are used ONLY
to evaluate cluster quality, never for training.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)


# ─────────────────────────── Data Loading ───────────────────────────

def load_newsgroups(categories):
    """Fetch the 20 Newsgroups subset and return a tidy DataFrame.

    Parameters
    ----------
    categories : list of str
        Newsgroup category names to include.

    Returns
    -------
    pd.DataFrame with columns:
        text          — raw document text
        true_category — newsgroup label string
    """
    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    df = pd.DataFrame({
        "text": dataset.data,
        "true_category": [dataset.target_names[t] for t in dataset.target],
    })
    return df


# ─────────────────────────── Data Cleaning ──────────────────────────

def clean_text(text):
    """Normalise a raw newsgroup document.

    Steps: lowercase → strip URLs → strip punctuation & digits →
    collapse whitespace.

    Parameters
    ----------
    text : str

    Returns
    -------
    str — cleaned text
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                   # keep only letters
    text = re.sub(r"\s+", " ", text).strip()                 # collapse whitespace
    return text


# ───────────────────────── Cluster Selection ────────────────────────

def find_optimal_k(X, k_range=range(2, 11)):
    """Compute inertia and silhouette score for each k in k_range.

    Parameters
    ----------
    X        : sparse or dense array of shape (n_docs, n_features)
    k_range  : iterable of int

    Returns
    -------
    inertia_list    : list of float
    silhouette_list : list of float
    """
    inertia_list = []
    silhouette_list = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertia_list.append(km.inertia_)
        silhouette_list.append(silhouette_score(X, labels, sample_size=2000, random_state=42))
    return inertia_list, silhouette_list


# ──────────────────────── Cluster Interpretation ────────────────────

def top_terms_per_cluster(kmeans, vectorizer, n=15):
    """Return the n most representative terms for each cluster centroid.

    Parameters
    ----------
    kmeans     : fitted KMeans instance
    vectorizer : fitted TfidfVectorizer (or similar)
    n          : int — number of terms to return per cluster

    Returns
    -------
    dict mapping cluster_id (int) → list of str
    """
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    result = {}
    for cluster_id in range(kmeans.n_clusters):
        top = [terms[i] for i in order_centroids[cluster_id, :n]]
        result[cluster_id] = top
    return result


# ──────────────────────── Cluster Evaluation ────────────────────────

def evaluate_clustering(true_labels, cluster_labels):
    """Compute external clustering quality metrics against ground-truth labels.

    Parameters
    ----------
    true_labels    : array-like of int or str — ground-truth category indices
    cluster_labels : array-like of int       — K-Means cluster assignments

    Returns
    -------
    dict with keys: ARI, NMI, homogeneity, completeness, v_measure
    """
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    hom, com, vme = homogeneity_completeness_v_measure(true_labels, cluster_labels)
    return {
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "homogeneity": round(hom, 4),
        "completeness": round(com, 4),
        "v_measure": round(vme, 4),
    }
