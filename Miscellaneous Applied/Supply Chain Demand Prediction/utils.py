"""
Supply Chain Demand Prediction — Utility Functions
Forecast daily order demand from lag/rolling/calendar features, benchmarked
against naive and seasonal-naive baselines (chronological split).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_demand(path="data/daily_demand.csv"):
    df = pd.read_csv(path, parse_dates=["order_date"]).sort_values("order_date")
    return df.reset_index(drop=True)

def build_features(df):
    df = df.copy()
    for L in (1, 7, 14): df[f"lag{L}"] = df["demand"].shift(L)
    df["dow"] = df["order_date"].dt.dayofweek
    df["month"] = df["order_date"].dt.month
    df["roll7"] = df["demand"].shift(1).rolling(7).mean()
    return df.dropna().reset_index(drop=True)

FEATURES = ["lag1", "lag7", "lag14", "dow", "month", "roll7"]

def run_models(df, test_frac=0.2):
    cut = int(len(df)*(1-test_frac)); tr, te = df[:cut], df[cut:]
    rows = []
    for name, pred in [("Naive (lag1)", te["lag1"]), ("Seasonal (lag7)", te["lag7"])]:
        rows.append({"model": name, "MAE": round(mean_absolute_error(te.demand, pred),2), "R2": round(r2_score(te.demand, pred),3)})
    for name, m in [("Linear", LinearRegression()), ("Ridge", Ridge(alpha=10)),
                    ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42))]:
        m.fit(tr[FEATURES], tr.demand); p = m.predict(te[FEATURES])
        rows.append({"model": name, "MAE": round(mean_absolute_error(te.demand, p),2), "R2": round(r2_score(te.demand, p),3)})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True), tr, te
