"""
Road Accident Analysis — Utility Functions
Exploratory data analysis / visualization project. Loads the dataset and provides
light helpers; the analysis lives in the notebooks (this is an EDA project, no model).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath="data/accidents.csv"):
    """Load the raw dataset."""
    return pd.read_csv(filepath, low_memory=False)

def missing_report(df):
    """Per-column missing counts and percentages, worst first."""
    m = df.isnull().sum()
    out = pd.DataFrame({"missing": m, "pct": (100*m/len(df)).round(2)})
    return out[out["missing"] > 0].sort_values("missing", ascending=False)

def top_counts(series, n=10, sep=None):
    """Value counts (optionally splitting multi-value cells on `sep`)."""
    s = series.dropna()
    if sep:
        s = s.str.split(sep).explode().str.strip()
    return s.value_counts().head(n)
