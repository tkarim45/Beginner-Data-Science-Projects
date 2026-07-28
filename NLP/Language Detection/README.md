# Language Detection

Identify which of 17 languages a piece of text is written in, using character n-gram TF-IDF features and classic machine-learning classifiers.

## Problem statement

Given a short text snippet, predict its language. Character-level patterns (letter combinations, scripts, diacritics) are highly language-specific, which makes this a strong fit for n-gram features even without word-level understanding.

## Dataset

The **Language Detection** dataset: 10,337 text samples across 17 languages (Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portuguese, Russian, Spanish, Swedish, Tamil, Turkish). Two columns: `Text` and `Language`. Source: [Kaggle Language Detection](https://www.kaggle.com/datasets/basilb2s/language-detection). The CSV is included under `data/`.

## Approach

- **Notebook 01 (EDA)**: class balance across the 17 languages, text-length distributions, script inspection.
- **Notebook 02 (cleaning)**: normalize text, produce a `clean_text` column, save `language_cleaned.csv`.
- **Notebook 03 (modeling)**: character n-gram `TfidfVectorizer` feeding 7 classifiers (Multinomial NB, Complement NB, Logistic Regression, Linear SVM, Ridge, Passive Aggressive, Random Forest), scored on a stratified split plus cross-validation.

Character n-grams (not words) are the key choice: they capture script and spelling patterns that separate languages cleanly.

## Project structure

```
Language Detection/
├── 01_eda.ipynb
├── 02_data_cleaning.ipynb
├── 03_model_building.ipynb
├── utils.py
├── requirements.txt
└── data/
    ├── language_detection.csv
    └── language_cleaned.csv
```

## Key findings (real output)

- Character n-gram TF-IDF + linear classifiers reach about **99% accuracy / F1** on the held-out test set (best single-split F1 ~0.990).
- Cross-validated F1: Linear SVM **0.9845**, Logistic Regression **0.9829**, Complement NB **0.9775**, Multinomial NB **0.9550**. The linear models lead.
- Languages with distinct scripts (Arabic, Greek, Hindi, Tamil, Kannada, Malayalam, Russian) are essentially perfectly separated; the small residual confusion is between closely related Latin-script European languages.
- Character n-grams beat word-level features here because language identity lives in the spelling and script, not the vocabulary.

## Tech stack

Python, pandas, scikit-learn (TF-IDF + linear classifiers), NLTK, Matplotlib, Jupyter.

## Getting started

```
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```

Run `01_eda`, then `02_data_cleaning`, then `03_model_building`.
