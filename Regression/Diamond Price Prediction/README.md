# Diamond Price Prediction

A beginner-level **regression** project that predicts a diamond's price (USD) from the classic 4 Cs (carat, cut, color, clarity) and physical dimensions.

## Problem Statement

Given a diamond's mass (carat), quality grades (cut, color, clarity), and physical dimensions (depth, table, x, y, z), predict its retail price.

## Dataset

- **Source**: [ggplot2 diamonds dataset](https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv), bundled with R's ggplot2 package, mirrored on GitHub
- **Samples**: 53,940 diamonds (20 dropped during cleaning for impossible 0-mm dimensions)
- **Features**: 9 inputs + 1 numeric target

| Feature | Type | Description |
|---------|------|-------------|
| carat | float | Mass (1 carat = 0.2 g) |
| cut | ordinal | Fair → Good → Very Good → Premium → Ideal |
| color | ordinal | J (worst, slight yellow) → D (best, colorless) |
| clarity | ordinal | I1 → SI2 → SI1 → VS2 → VS1 → VVS2 → VVS1 → IF |
| depth | float | Total depth percentage = z / mean(x, y) × 100 |
| table | float | Width of the top of the diamond relative to widest point |
| x, y, z | float | Length / width / depth in mm |
| **price** | **int** | **Retail price in USD ($326, $18,823)** |

## Project Structure

```
Diamond Price Prediction/
├── 01_eda.ipynb              # Exploratory Data Analysis
├── 02_data_cleaning.ipynb    # Data Cleaning & Feature Engineering
├── 03_model_building.ipynb   # Model Building & Evaluation
├── utils.py                  # Reusable utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    ├── diamonds.csv          # Raw dataset (53,940 rows, 10 cols)
    └── diamonds_cleaned.csv  # Processed dataset (~53,920 rows after dropping zero-dim)
```

**Run notebooks in order**: `01_eda.ipynb` -> `02_data_cleaning.ipynb` -> `03_model_building.ipynb`

## Results

7 regression models were trained and the best baseline was tuned with `GridSearchCV`. To keep tuning manageable on 54k rows we used a small grid + 3-fold CV.

| Model | R² | RMSE ($) | MAE ($) | MAPE |
|-------|------|----------|---------|------|
| Gradient Boosting | 0.9827 | 527 | 271 | 0.075 |
| Random Forest | 0.9820 | 537 | 260 | 0.064 |
| Decision Tree | 0.9753 | 629 | 318 | 0.086 |
| KNN (K=11) | 0.9719 | 672 | 362 | 0.101 |
| Lasso | 0.9121 | 1,187 | 803 | 0.437 |
| Ridge | 0.7058 | 2,172 | 798 | 0.434 |
| Linear Regression | 0.6881 | 2,236 | 800 | 0.435 |
| **Random Forest (Tuned)** | **0.9820** | **537** | **263** | **0.066** |

Best GridSearchCV parameters: `n_estimators=200, max_depth=15, min_samples_leaf=1` (CV R² = 0.9804).

## Key Findings

- **Carat is by far the dominant predictor** (r ≈ 0.92 with price). Together with the physical dimensions x/y/z it explains most of the variance.
- **Tree ensembles dominate** (R² ≈ 0.98) because the underlying price function is highly non-linear in carat: price grows roughly cubically with linear dimensions.
- **Linear models cap at R² ≈ 0.69** in their raw form because the carat ↔ x ↔ y ↔ z multicollinearity confuses them; Lasso recovers some of this by zeroing redundant coefficients.
- **20 rows have x/y/z = 0**, physically impossible; we drop them during cleaning rather than impute.
- **Cut/color/clarity are weakly correlated with price marginally** but become important *within* a fixed carat range. Simpson's-paradox style: larger diamonds happen to have lower-graded cut/color/clarity, hiding their true effect.
- **A log-transform of `price`** would close the linear-vs-tree gap dramatically (price grows roughly with carat³); see "Next Steps" in the model notebook.

## Tech Stack

- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
