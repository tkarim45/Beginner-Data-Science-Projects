# Personal Finance Tracker with Spending Predictions

A beginner-level **applied** project that analyses a year of personal transactions, spending by category,
monthly trends, auto-categorisation, and a simple spend forecast.

## Problem Statement
Given a stream of transactions, track where money goes, auto-categorise new transactions, and predict
monthly spend, the core of a personal-finance app.

## Dataset
- **Simulated** (seeded): **770 transactions** over 2024, monthly salary + rent + realistic discretionary
  spend across 8 categories (`data/transactions.csv`: date, category, amount, description).

> Note: the checklist's Kaggle personal-finance dataset isn't openly hosted; a seeded simulation of
> realistic categorised transactions is used instead (reproducible, and privacy-safe).

## Project Structure
```
Personal Finance Tracker/
├── 01_eda.ipynb        # Spending by category, monthly totals
├── 02_analysis.ipynb   # Auto-categorisation classifier + monthly-spend forecast
├── utils.py · requirements.txt · README.md
└── data/transactions.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **Where the money goes**: rent is the single biggest line (~$16,800/yr), then discretionary categories
  (shopping, groceries, dining); total spend ~$55.7k against ~$42k income in the simulation.
- **Monthly spend is stable** (~$4,600/month) and roughly forecastable by a simple linear trend, useful
  for budgeting.
- **Auto-categorisation from amount + timing alone is weak (~0.31 accuracy)**, better than chance (~0.11
  for 9 classes) but unreliable, because **spend amounts overlap heavily across categories** (a $40 charge
  could be groceries, dining, or transport).
- **The honest lesson**: real transaction categorisation needs the **merchant/description text** (an NLP
  problem), not just the number, amount is a weak feature. A practical tracker combines keyword rules on
  descriptions with this statistical layer.

## Tech Stack
- pandas, numpy, matplotlib, scikit-learn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
