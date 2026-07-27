"""
Mall Customer Segmentation - Utility Functions
K-Means clustering on demographic + spending features.
This is an unsupervised project — no target variable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def load_data(filepath="data/Mall_Customers.csv"):
    """Load and rename columns for easier handling."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "Annual Income (k$)": "Income",
        "Spending Score (1-100)": "SpendingScore",
    })
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    return df


def preprocess_features(df, features=None):
    """One-hot encode Gender and standardize features."""
    df = df.copy()
    if "Gender" in df.columns:
        df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
        if "Gender_Male" in df.columns:
            df["Gender_Male"] = df["Gender_Male"].astype(int)

    if features is None:
        features = df.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return df, X_scaled, scaler, features


def find_optimal_k(X, k_range=range(2, 11)):
    """Compute inertia and silhouette for each k. Returns DataFrame."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def cluster_summary(df, labels, cluster_col="cluster"):
    df_with = df.copy()
    df_with[cluster_col] = labels
    numeric_cols = df_with.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)
    summary = df_with.groupby(cluster_col)[numeric_cols].mean().round(2)
    summary["Count"] = df_with[cluster_col].value_counts().sort_index().values
    return summary


def plot_elbow(metrics_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="steelblue")
    ax.set_xlabel("Number of clusters k"); ax.set_ylabel("Inertia (within-cluster SS)")
    ax.set_title("Elbow plot — choose k where the curve bends")
    return ax


def plot_silhouette(metrics_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="seagreen")
    ax.set_xlabel("Number of clusters k"); ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score — higher is better")
    return ax
