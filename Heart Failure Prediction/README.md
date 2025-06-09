# 🫀 Heart Failure Prediction

## 🔍 Overview
This project uses clinical data to predict the likelihood of death due to heart failure using supervised machine learning.  
Cardiovascular diseases are the leading cause of death globally, claiming an estimated 17.9 million lives each year (31% of all global deaths). Early detection through predictive modeling can support timely medical intervention.

## 📁 Dataset
- **Source**: [Kaggle – Heart Failure Prediction Dataset](https://www.kaggle.com/code/karnikakapoor/heart-failure-prediction-ann/input)
- **Features**:  
  `age`, `anaemia`, `creatinine_phosphokinase`, `diabetes`, `ejection_fraction`, `high_blood_pressure`, `platelets`, `serum_creatinine`, `serum_sodium`, `sex`, `smoking`, `time`
- **Target**: `DEATH_EVENT` (binary classification)

## 🛠️ Tools & Libraries
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (Logistic Regression)

## 📈 Model Summary
- **Algorithm**: Logistic Regression
- **Accuracy**: 84%
- **Evaluation Metrics**: Confusion Matrix, Classification Report

## 🚀 How to Run
1. Clone the repository or download the project folder.
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
