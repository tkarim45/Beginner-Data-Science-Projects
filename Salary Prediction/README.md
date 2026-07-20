# Salary Prediction Based on Experience

A beginner-level **regression** project that predicts a person's salary from their years of work experience. The dataset is famous as a textbook example of simple linear regression.

## Problem Statement

Given a single numeric feature (`YearsExperience`), predict an employee's annual `Salary` (USD).

## Dataset

- **Source**: [Salary Data (Kaggle)](https://www.kaggle.com/datasets/rsadiq/salary)
- **Samples**: 30 employees
- **Features**: 1 numeric input + 1 numeric target

| Feature | Type | Description |
|---------|------|-------------|
| YearsExperience | float | Years of professional work experience (1.1 to 10.5) |
| **Salary** | **float** | **Annual salary in USD ($37k, $122k)** |

The dataset is already clean, no missing values, no duplicates.

## Project Structure

```
Salary Prediction/
├── 01_eda.ipynb              # Exploratory Data Analysis
├── 02_data_cleaning.ipynb    # Feature Engineering
├── 03_model_building.ipynb   # Model Building & Evaluation
├── utils.py                  # Reusable utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    ├── salary.csv          # Raw dataset (30 rows, 2 cols)
    └── salary_cleaned.csv  # Processed dataset with engineered features
```

**Run notebooks in order**: `01_eda.ipynb` -> `02_data_cleaning.ipynb` -> `03_model_building.ipynb`

## Results

7 regression models were trained on the engineered feature set; the best linear-family model was tuned with `GridSearchCV`.

| Model | R² | RMSE ($) | MAE ($) | MAPE |
|-------|------|----------|---------|------|
| KNN (K=3) | 0.9441 | 5,344 | 4,665 | 0.058 |
| Ridge | 0.8944 | 7,346 | 6,228 | 0.076 |
| Random Forest | 0.8805 | 7,814 | 6,795 | 0.085 |
| Lasso | 0.8747 | 8,001 | 6,546 | 0.083 |
| Linear Regression | 0.8684 | 8,198 | 6,705 | 0.086 |
| Decision Tree | 0.8188 | 9,619 | 7,430 | 0.085 |
| Gradient Boosting | 0.8130 | 9,774 | 8,215 | 0.100 |
| **Ridge (Tuned)** | **0.8700** | **8,149** | **6,669** | **0.085** |

Best GridSearchCV parameters: `alpha=0.001` (CV R² = 0.8934).

## Key Findings

- **YearsExperience and Salary are nearly perfectly correlated** (Pearson r ≈ 0.978).
- **Each year of experience adds ≈ $9,500/year** to salary in this sample, with an entry-level intercept around $25k.
- **KNN (K=3) wins on the tiny held-out test set** because nearest-neighbor matching gives flexible local fits when there are only ~30 points. Its CV variance is high. Linear / Ridge are more honest "expected" performers.
- **Gradient Boosting actually under-performs** here: with 30 rows, it doesn't have enough data to fit the gradient steps well.
- **The dataset is so small that test-set metrics carry significant variance**. Take the rankings with a grain of salt and trust the 5-fold CV in section 12.
- **No missing values, no duplicates**, already clean.

## Tech Stack

- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
