"""
Customer Segmentation — Utility Functions
RFM (Recency, Frequency, Monetary) analysis + K-Means clustering on
UCI Online Retail transaction data.
This is an unsupervised project — no target variable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data(filepath="data/online_retail.csv"):
    """Load the raw Online Retail transaction CSV."""
    df = pd.read_csv(filepath, encoding="utf-8")
    return df


# ─────────────────────────────────────────────
# Data Cleaning
# ─────────────────────────────────────────────

def clean_transactions(df):
    """
    Clean raw transaction rows:
      - Drop rows with missing CustomerID
      - Remove cancelled invoices (InvoiceNo starts with 'C')
      - Remove rows with Quantity <= 0 or UnitPrice <= 0
      - Parse InvoiceDate to datetime
    Returns the cleaned DataFrame and a dict with drop counts.
    """
    df = df.copy()
    stats = {}

    start = len(df)

    # Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    stats["missing_customer_id"] = start - len(df)

    # Remove cancellations
    before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    stats["cancellations"] = before - len(df)

    # Remove non-positive Quantity / UnitPrice
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    stats["non_positive"] = before - len(df)

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    stats["rows_kept"] = len(df)
    return df, stats


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

def build_rfm(df):
    """
    Aggregate cleaned transactions to one row per customer.
    Recency  = days between customer's last purchase and (max_date + 1 day)
    Frequency = number of unique invoices
    Monetary  = total spend (sum of Quantity * UnitPrice)
    Returns a DataFrame with columns: CustomerID, Recency, Frequency, Monetary.
    """
    df = df.copy()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_features(df, features=("Recency", "Frequency", "Monetary")):
    """
    Log-transform skewed RFM columns then apply StandardScaler.
    Returns: df_log (log-transformed copy), X_scaled (numpy array), scaler, features list.
    """
    df_log = df[list(features)].copy()
    for col in features:
        df_log[col] = np.log1p(df_log[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log[list(features)])
    return df_log, X_scaled, scaler, list(features)


# ─────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────

def find_optimal_k(X, k_range=range(2, 11)):
    """
    Fit K-Means for each k in k_range.
    Returns a DataFrame with columns: k, inertia, silhouette.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
        })
    return pd.DataFrame(rows)


def profile_clusters(df, labels):
    """
    Compute mean Recency, Frequency, Monetary per cluster plus cluster size.
    df must contain columns Recency, Frequency, Monetary.
    Returns a summary DataFrame indexed by cluster label.
    """
    df_with = df[["Recency", "Frequency", "Monetary"]].copy()
    df_with["Cluster"] = labels
    summary = df_with.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    summary["Count"] = df_with["Cluster"].value_counts().sort_index().values
    return summary
