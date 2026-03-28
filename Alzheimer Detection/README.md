# Alzheimer Disease Detection

Classifies patients into diagnostic categories (Cognitive Normal, Alzheimer's Disease, Late Mild Cognitive Impairment) using clinical, cognitive, and neuroimaging data. Merges multiple data sheets (diagnosis, cognitive scores, demographics/brain measures), performs feature engineering, and trains a Random Forest classifier.

## Dataset

`CSI_7_MAL_2324_CW_resit_data.xlsx` -- an Excel workbook with three sheets:
- **Diagnosis target**: diagnosis labels and FDG-PET values
- **Cognitive score**: CDRSB, ADAS, MMSE, RAVLT scores
- **Data**: demographics, genetics (ApoE4), and 361 brain structure measurements (volumes, surface areas, cortical thickness)

## Tech Stack

- pandas, numpy
- matplotlib
- scikit-learn (StandardScaler, LabelEncoder, RandomForestClassifier, train_test_split)

## Results

The Random Forest classifier achieves 100% accuracy on both the training and test sets, classifying patients into Cognitive Normal, Alzheimer's Disease, and Late Mild Cognitive Impairment categories.

## How to Run

1. Install dependencies: `pip install pandas numpy matplotlib scikit-learn openpyxl`
2. Place `CSI_7_MAL_2324_CW_resit_data.xlsx` in the project directory.
3. Run `Alzheimer.ipynb` from start to finish.
