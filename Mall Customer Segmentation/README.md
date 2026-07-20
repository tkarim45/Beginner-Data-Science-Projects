# Mall Customer Segmentation

A beginner-level **unsupervised clustering** project that segments mall shoppers using K-Means on demographic and behavioural features.

## Problem Statement

Given a customer's age, gender, annual income, and "spending score" (a mall-assigned 1-100 metric of in-store engagement), find natural customer segments to inform targeted marketing.

There is **no target column**, this is genuine unsupervised learning. Cluster quality is judged using the elbow plot, silhouette score, and interpretability of the resulting personas.

## Dataset

- **Source**: [Mall Customer Segmentation Data. Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Samples**: 200 customers
- **Features**: 4 used predictors (after dropping `CustomerID`)

| Feature | Description |
|---|---|
| Gender | Male / Female |
| Age | Customer age in years |
| Income | Annual income in thousands of USD |
| SpendingScore | Mall-assigned 1 to 100 score (higher = more spending / more frequent) |

## Project Structure

```
Mall Customer Segmentation/
├── 01_eda.ipynb              # Exploratory Data Analysis
├── 02_data_cleaning.ipynb    # Encoding & Standardisation
├── 03_model_building.ipynb   # K-Means + Hierarchical clustering
├── utils.py                  # Reusable clustering helpers
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    ├── Mall_Customers.csv         # Raw dataset (200 rows)
    ├── mall_customers_cleaned.csv # One-hot encoded (notebook 02)
    └── mall_customers_scaled.csv  # Standardised features (notebook 02)
```

## Results

K-Means is fit for k ∈ {2, …, 10}. Silhouette score increases roughly monotonically over this range, the algorithm picks **k = 10** as the silhouette-optimal partition for the 4-feature input (Age, Income, SpendingScore, Gender).

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2 | 588.80 | 0.252 | 1.61 | 71.0 |
| 3 | 476.79 | 0.260 | 1.36 | 66.8 |
| 4 | 388.72 | 0.298 | 1.28 | 69.1 |
| 5 | 331.31 | 0.304 | 1.17 | 69.0 |
| 6 | 276.41 | 0.331 | 1.02 | 73.5 |
| 7 | 236.20 | 0.357 | 0.98 | 76.8 |
| 8 | 199.75 | 0.388 | 0.94 | 82.4 |
| 9 | 174.24 | 0.403 | 0.88 | 85.7 |
| **10** | **152.03** | **0.421** | **0.83** | **89.9** |

The textbook view of this dataset (Income × SpendingScore only) suggests **k = 5**: high/high, high/low, mid/mid, low/high, low/low. With Age and Gender added as features, the algorithm finds finer sub-clusters within each spend group, which is what drives the silhouette toward k = 10.

### Persona summary (k = 10)

| Cluster | Avg Age | Avg Income | Avg Spend | Count | Persona |
|---|---|---|---|---|---|
| 0 | 58.9 | 48.7 | 39.9 | 26 | Older average shopper |
| 1 | 25.3 | 41.3 | 60.9 | 24 | Young, mid-income, engaged |
| 2 | 41.2 | 26.1 | 20.1 | 14 | Frugal, low income, low spend |
| 3 | 32.2 | 86.1 | 81.7 | 21 | **Target, high income, high spend (younger)** |
| 4 | 54.2 | 54.2 | 49.0 | 26 | Older mid-income shopper |
| 5 | 38.5 | 85.9 | 14.2 | 19 | Cautious, high income, low spend |
| 6 | 28.0 | 57.4 | 47.1 | 25 | Young mid-tier |
| 7 | 33.3 | 87.1 | 82.7 | 18 | **Target, high income, high spend (other)** |
| 8 | 25.5 | 25.7 | 80.5 | 13 | Impulsive, low income, high spend |
| 9 | 43.8 | 93.3 | 20.6 | 14 | Cautious, high income, low spend (older) |

## Key Findings

- **Income × SpendingScore is the dominant axis**, every cluster falls clearly into one of the four corners or the middle zone.
- **Two distinct "target" segments** emerge: clusters 3 and 7 are both high-income, high-spend, but cluster 3 skews younger. Marketing campaigns can be tailored differently to each.
- **The "cautious" segment (high income, low spend)** is the most interesting commercial opportunity, these shoppers have the means but are not engaging. Clusters 5 and 9 capture this pattern across two age bands.
- **Hierarchical clustering produces similar segments**, confirming the K-Means partition is genuine and not an artifact of the algorithm.
- **The silhouette plot doesn't have a clean "knee"**, adding Age and Gender to the feature set blurs the textbook 5-cluster picture, so the choice of k is partly a judgement call between interpretability and statistical fit.

## Tech Stack

- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
