"""
Restaurant Revenue Prediction — Utility Functions
Regression on the TFI (Kaggle) restaurant dataset: predict annual `revenue`
from restaurant metadata (open date, city group, type) and 37 obfuscated
numeric features P1–P37.

This is a small dataset (137 rows), so cross-validation matters more than any
single train/test split — utilities below expose both.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data(filepath="data/train.csv"):
    """Load the raw TFI restaurant training CSV."""
    return pd.read_csv(filepath)


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

REFERENCE_DATE = pd.Timestamp("2015-01-01")  # competition snapshot; age measured from here

def create_features(df, reference_date=REFERENCE_DATE):
    """
    Engineer model-ready features from the raw frame:
      - restaurant_age_years : years between Open Date and the reference snapshot
      - drop Id (identifier) and City (34 high-cardinality levels — too granular for 137 rows)
      - keep City Group, Type (low-cardinality categoricals) and P1–P37 (numeric)
    Returns a new DataFrame including the `revenue` target (if present).
    """
    df = df.copy()
    df["Open Date"] = pd.to_datetime(df["Open Date"], format="%m/%d/%Y")
    df["restaurant_age_years"] = (reference_date - df["Open Date"]).dt.days / 365.25

    drop_cols = [c for c in ["Id", "Open Date", "City"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_data(df, target="revenue"):
    """
    Full pipeline: feature-engineer → one-hot encode categoricals (drop_first) →
    split X / y. Returns X (DataFrame), y (Series), feature names.
    """
    fe = create_features(df)
    fe = pd.get_dummies(fe, columns=[c for c in ["City Group", "Type"] if c in fe.columns],
                        drop_first=True)
    y = fe[target] if target in fe.columns else None
    X = fe.drop(columns=[target]) if target in fe.columns else fe
    return X, y, list(X.columns)


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

def get_models():
    """Return the standard 7-regressor dictionary."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=1000.0, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }


# ─────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Fit on train, predict on test, return R2 / RMSE / MAE dict."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return {
        "r2": round(r2_score(y_test, pred), 4),
        "rmse": round(rmse, 1),
        "mae": round(mean_absolute_error(y_test, pred), 1),
    }


def compare_models(models, X_train, X_test, y_train, y_test, X_all=None, y_all=None, cv=5):
    """
    Evaluate every model on the held-out test split and (if X_all/y_all given) with
    k-fold cross-validation on the full dataset — the more reliable signal at n=137.
    Returns a DataFrame sorted by CV R² (or test R²).
    """
    rows = []
    for name, model in models.items():
        m = evaluate_model(model, X_train, X_test, y_train, y_test)
        row = {"model": name, **{f"test_{k}": v for k, v in m.items()}}
        if X_all is not None:
            scores = cross_val_score(model, X_all, y_all, cv=cv, scoring="r2")
            row["cv_r2_mean"] = round(scores.mean(), 4)
            row["cv_r2_std"] = round(scores.std(), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_key = "cv_r2_mean" if "cv_r2_mean" in out.columns else "test_r2"
    return out.sort_values(sort_key, ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_pred_vs_actual(model, X_test, y_test, ax=None):
    """Scatter predicted vs actual revenue with the ideal y=x line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    pred = model.predict(X_test)
    ax.scatter(y_test, pred, alpha=0.6)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", label="perfect")
    ax.set_xlabel("actual revenue"); ax.set_ylabel("predicted revenue")
    ax.set_title("Predicted vs actual"); ax.legend()
    return ax


def plot_feature_importance(model, feature_names, top_n=15, ax=None):
    """Bar chart of the top-n feature importances for a tree-based model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    imp[::-1].plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} feature importances"); ax.set_xlabel("importance")
    return ax
