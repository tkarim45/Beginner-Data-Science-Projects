# Heart Failure Prediction

Predicts the likelihood of death due to heart failure using clinical data and supervised machine learning. Cardiovascular diseases are the leading cause of death globally, claiming an estimated 17.9 million lives each year. Early detection through predictive modeling can support timely medical intervention.

## Dataset

- **Source**: [Kaggle -- Heart Failure Prediction Dataset](https://www.kaggle.com/code/karnikakapoor/heart-failure-prediction-ann/input)
- **Features**: `age`, `anaemia`, `creatinine_phosphokinase`, `diabetes`, `ejection_fraction`, `high_blood_pressure`, `platelets`, `serum_creatinine`, `serum_sodium`, `sex`, `smoking`, `time`
- **Target**: `DEATH_EVENT` (binary classification)

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (Logistic Regression)

## Results

The Logistic Regression model achieves 84% accuracy on the test set, evaluated with a confusion matrix and classification report.

## How to Run

1. Clone the repository or download the project folder.
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `heart_failure.ipynb` in Jupyter Notebook and run all cells.
