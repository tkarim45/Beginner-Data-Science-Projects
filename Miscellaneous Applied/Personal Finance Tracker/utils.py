"""
Personal Finance Tracker with Spending Predictions — Utility Functions
Analyse a year of personal transactions: categorise spending, track monthly
totals, and predict spend. Data is a seeded simulation of realistic categorised
transactions (income + discretionary spend + rent).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_transactions(path="data/transactions.csv"):
    return pd.read_csv(path, parse_dates=["date"])

def monthly_spend(df):
    exp = df[df.amount < 0]
    return exp.groupby(exp.date.dt.to_period("M"))["amount"].sum().abs()

def category_totals(df):
    exp = df[df.amount < 0]
    return exp.groupby("category")["amount"].sum().abs().sort_values(ascending=False)

def classify_category(df):
    """Predict a transaction's category from amount + calendar features (RandomForest)."""
    d = df.copy()
    d["day"] = d.date.dt.day; d["dow"] = d.date.dt.dayofweek; d["absamt"] = d.amount.abs()
    X, y = d[["absamt", "day", "dow"]], d["category"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
    return clf, round(accuracy_score(yte, clf.predict(Xte)), 3)
