# E-Commerce Purchase Behavior Analysis

A beginner-level **EDA / visualization** project exploring how Brazilians shop online, 99,441 Olist
marketplace orders with price, freight, payment method, installments, delivery status, and review score.

## Problem Statement
How do customers pay, what do they spend, and does delivery speed drive satisfaction? Exploratory
analysis; insight + visualizations, no model.

## Dataset
- **Source**: [Olist Brazilian E-Commerce](https://github.com/olist/work-at-olist-data), orders joined
  with order-items, payments, and reviews into one order-level table.
- **99,441 orders × 15 columns**: status, timestamps, n_items, price, freight, payment, installments,
  payment_type, review_score.

## Project Structure
```
E-Commerce Purchase Behavior Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, distributions
├── 02_analysis.ipynb   # Payment methods, order value & installments, reviews, delivery→satisfaction
├── utils.py · requirements.txt · README.md
└── data/orders.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **99,441 orders, 97% delivered**; mean order **R$137.75** + **R$22.87** freight.
- **Credit card dominates (76%)**, then **boleto (20%)**, a Brazil-specific bank slip, with voucher/debit trailing.
- **Installment culture is real**, about half of orders are single-payment, but a long tail spreads
  purchases across 2 to 12 monthly instalments (a core Brazilian e-commerce behaviour).
- **Customers are satisfied on average (mean review 4.09/5)**, though reviews are bimodal (many 5s + a chunk of 1s).
- **Slow delivery drags reviews down**, mean delivery time is markedly longer for 1-star than 5-star
  orders (negative delivery-days↔score correlation): logistics, not price, is the satisfaction lever.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
