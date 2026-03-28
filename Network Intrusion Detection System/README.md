# Network Intrusion Detection System

Builds a network intrusion detection system (NIDS) that classifies network connections as normal or one of several attack types. Multiple classifiers are compared, including Decision Tree, Random Forest, AdaBoost, XGBoost, Gradient Boosting, and HistGradientBoosting, with evaluation via accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset

KDD Cup 1999 dataset (`KDDCup Data 10 Percent.csv`) -- a widely used benchmark for network intrusion detection research. Supporting files include `kddcup.txt` (column names) and `training_attack_types.txt` (attack type mappings).

## Tech Stack

- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn (Decision Tree, Random Forest, AdaBoost, Gradient Boosting, HistGradientBoosting, StandardScaler, LabelEncoder, GridSearchCV)
- XGBoost

## Results

| Model | Accuracy |
|-------|----------|
| Random Forest | 99.76% |
| Decision Tree | 99.59% |
| KNN | 99.41% |
| XGBoost | 99.39% |
| AdaBoost | 94.54% |
| Hist Gradient Boosting | 93.10% |
| Gradient Boosting | 92.86% |

## How to Run

1. Ensure `KDDCup Data 10 Percent.csv`, `kddcup.txt`, and `training_attack_types.txt` are in the project directory.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`
3. Open and run `NIDS.ipynb` in Jupyter Notebook.
