# Resume Screening / Classification

A beginner-level **NLP text-classification** project that sorts resumes into job categories
(Data Science, HR, Java Developer, …) directly from their text, the core of an automated
resume-screening pipeline.

## Problem Statement

Given the raw text of a resume, predict its job category. This is **multi-class**
classification over **25 categories**, solved with a TF-IDF + classic-ML pipeline.

## Dataset

- **Source**: [Resume Dataset. Kaggle/HF (`Sachinkelenjaguri/Resume_dataset`)](https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset)
- **Resumes**: 962 · **Categories**: 25 (e.g. Data Science, HR, Advocate, Java Developer)

| Column | Description |
|---|---|
| `text` | Full resume text |
| `label` | Job category (target) |

## Project Structure

```
Resume Screening Classification/
├── 01_eda.ipynb              # Class balance, text length, frequent words
├── 02_data_cleaning.ipynb    # clean_text column (lowercase, strip, stopwords)
├── 03_model_building.ipynb   # TF-IDF + 6 classifiers, tuning, confusion matrix
├── utils.py                  # Loaders, cleaning, evaluation helpers
├── requirements.txt
├── README.md
└── data/
    └── resumes.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Models

TF-IDF (unigrams + bigrams, 20k features) → Multinomial NB, Complement NB, Logistic
Regression, Linear SVM, Ridge, Passive-Aggressive.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. 80/20 stratified
split; weighted metrics.

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| **Linear SVM** | **1.0000** | **1.0000** |
| Passive Aggressive | 1.0000 | 1.0000 |
| Ridge Classifier | 0.9948 | 0.9949 |
| Complement NB | 0.9948 | 0.9947 |
| Logistic Regression | 0.9896 | 0.9895 |
| Multinomial NB | 0.9534 | 0.9483 |

Tuned Logistic Regression (C=10): **accuracy 1.000, F1 1.000**.

## Key Findings

- **Resume categories are near-perfectly separable**. Linear SVM and Passive-Aggressive both
  reach **100% F1**. Job categories use very distinctive vocabulary (a Data Science resume and
  an HR resume share little jargon), so a linear model on TF-IDF separates them cleanly.
- **Even the weakest model is strong**. Multinomial NB still scores 0.95 F1; the task is
  genuinely easy given clean category labels.
- **The small dataset (962 resumes) is a caveat**, perfect scores partly reflect that each of
  the 25 categories is internally consistent and the test split is small; more diverse, noisier
  real-world resumes would lower these numbers.
- **Linear SVM is the right default** for high-dimensional sparse TF-IDF text classification.

## Tech Stack

- pandas, numpy, matplotlib, seaborn
- scikit-learn, nltk

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
