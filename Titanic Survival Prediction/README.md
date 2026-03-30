# Titanic Survival Prediction

Predicts whether a passenger survived the Titanic disaster using machine learning. This is a classic binary classification problem using the famous Kaggle Titanic dataset. The project walks through the full data science workflow: exploratory analysis, data cleaning, feature engineering, and model comparison.

## Dataset

- **Source**: [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Samples**: 891 passengers
- **Features**: `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`
- **Target**: `Survived` (0 = Did not survive, 1 = Survived)

## Project Structure

```
Titanic Survival Prediction/
├── data/
│   └── titanic.csv              # Raw dataset
├── 01_eda.ipynb                 # Exploratory Data Analysis
├── 02_data_cleaning.ipynb       # Data Cleaning & Feature Engineering
├── 03_model_building.ipynb      # Model Training & Evaluation
├── utils.py                     # Reusable utility functions
├── requirements.txt             # Python dependencies
└── README.md
```

## Notebooks

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 1 | `01_eda.ipynb` | Dataset overview, missing value analysis, univariate/bivariate analysis, correlation heatmaps, survival patterns by sex, class, age, fare, and family size |
| 2 | `02_data_cleaning.ipynb` | Handling missing values (group median, mode), feature engineering (Title, FamilySize, IsAlone, AgeGroup, FareBin, HasCabin), encoding categoricals |
| 3 | `03_model_building.ipynb` | Train/test split, feature scaling, 7 classification models, cross-validation, hyperparameter tuning with GridSearchCV, ROC curves, model comparison |

## Models Tested

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline model |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble of decision trees |
| K-Nearest Neighbors | Distance-based classification |
| Support Vector Machine | Maximum-margin classifier |
| Gradient Boosting | Sequential ensemble method |
| Naive Bayes | Probabilistic classifier |

## Key Findings

- **Sex** is the strongest predictor — females survived at ~74%, males at ~19%
- **Passenger class** matters — 1st class: 63% survival vs 3rd class: 24%
- **Children** (age < 12) had higher survival rates
- **Small families** (2-4 members) survived at higher rates than solo travelers or large families
- Engineered features like **Title** and **HasCabin** improved model performance

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, Naive Bayes)

## How to Run

1. Clone the repository or download the project folder.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in order:
   ```bash
   jupyter notebook
   ```
   - Start with `01_eda.ipynb` for data exploration
   - Then `02_data_cleaning.ipynb` to prepare the data
   - Finally `03_model_building.ipynb` to train and evaluate models
