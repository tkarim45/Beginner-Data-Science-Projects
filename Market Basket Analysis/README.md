# Market Basket Analysis (Apriori)

A beginner-level **applied** project that mines **association rules** from retail transactions, which
products are bought together far more often than chance, using the Apriori algorithm.

## Problem Statement
Given thousands of shopping baskets, find rules like "customers who buy X also buy Y", scored by
**support** (how common), **confidence** (P(Y|X)), and **lift** (how much more than chance). These
drive cross-sell, bundling, and store layout.

## Dataset
- **Source**: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), cleaned to
  positive-quantity, non-cancelled lines, restricted to the **top 150 products**.
- **`data/transactions.csv`**: 129,162 rows over **16,866 invoices** (InvoiceNo, Description, Quantity).

## Project Structure
```
Market Basket Analysis/
├── 01_eda.ipynb        # Best-sellers, basket-size distribution
├── 02_analysis.ipynb   # Apriori frequent itemsets → association rules by lift
├── utils.py · requirements.txt · README.md
└── data/transactions.csv
```

## Method
`mlxtend` Apriori for frequent itemsets (min support 2%) → association rules filtered to **lift ≥ 2**,
sorted by lift.

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **294 frequent itemsets → 312 rules** clear the lift≥2 bar.
- **The strongest rule is the Regency teacup set**, pink / green / roses teacups co-occur with
  **lift ~15** and confidence up to **0.90**: a shopper holding two of the three almost always buys the third.
- **Lift, not support, surfaces the interesting rules**, high-support pairs are often just two
  individually-popular items; high-lift pairs reveal genuine complementarity (matching sets, bundles).
- **Actionable**: bundle/cross-sell the high-lift groups and shelve them together.

## Tech Stack
- pandas, numpy, matplotlib, mlxtend

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
