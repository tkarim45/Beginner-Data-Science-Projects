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

## Results

7 classification models were trained and compared. Best model was further tuned with GridSearchCV.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8380 | 0.7941 | 0.7826 | 0.7883 |
| SVM | 0.8268 | 0.7969 | 0.7391 | 0.7669 |
| Random Forest | 0.8156 | 0.7812 | 0.7246 | 0.7519 |
| Gradient Boosting | 0.8156 | 0.8103 | 0.6812 | 0.7402 |
| Decision Tree | 0.7989 | 0.7797 | 0.6667 | 0.7188 |
| Naive Bayes | 0.7765 | 0.6883 | 0.7681 | 0.7260 |
| KNN | 0.7709 | 0.7500 | 0.6087 | 0.6720 |
| **Random Forest (Tuned)** | **0.8045** | **0.7656** | **0.7101** | **0.7368** |

Best GridSearchCV parameters: `max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200`

5-Fold Cross-Validation (mean accuracy): Logistic Regression (0.8175), Gradient Boosting (0.8133), SVM (0.8104), Random Forest (0.8077)

## Key Findings

- **Logistic Regression was the best performer** — highest accuracy (83.8%) and F1 score (0.7883)
- **Sex** is the strongest predictor — females survived at ~74%, males at ~19%
- **Passenger class** matters — 1st class: 63% survival vs 3rd class: 24%
- **Fare correlates with survival** — survivors paid an average of $48.40 vs $22.12 for non-survivors
- **Overall survival rate**: 38.4% (342 survived out of 891)
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
