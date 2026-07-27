"""
Market Basket Analysis — Utility Functions
Association-rule mining (Apriori) on retail transactions: find products frequently
bought together, scored by support, confidence, and lift.
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def load_transactions(path="data/transactions.csv"):
    """Load the cleaned top-150-product transactions (invoice, product, quantity)."""
    return pd.read_csv(path)

def build_basket(df):
    """One-hot invoice x product matrix (True = product in that basket)."""
    basket = df.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
    return basket > 0

def mine_rules(basket, min_support=0.02, min_lift=2.0):
    """Frequent itemsets (Apriori) -> association rules sorted by lift."""
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    return freq, rules.sort_values("lift", ascending=False).reset_index(drop=True)
