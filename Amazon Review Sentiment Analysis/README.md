# Amazon Review Sentiment Analysis

A beginner-level **NLP text-classification** project that predicts whether an Amazon product
review expresses **positive** or **negative** sentiment, using a TF-IDF + classic-ML pipeline.

## Problem Statement

Given the text of a product review (title + body), classify its sentiment as positive or
negative, **binary** classification.

## Dataset

- **Source**: [`amazon_polarity` (Hugging Face)](https://huggingface.co/datasets/amazon_polarity)
- **Subsample**: a **balanced 30,000 reviews** (15,000 positive + 15,000 negative) streamed from
  the full corpus.

> Dataset note: the checklist's Amazon Fine Food dataset is ~300 MB and Kaggle-gated; the openly
> hosted `amazon_polarity` (also Amazon product reviews) gives the same positive/negative
> sentiment task and streams without an account.

| Column | Description |
|---|---|
| `text` | Review title + body |
| `label` | `positive` / `negative` (target) |

## Project Structure

```
Amazon Review Sentiment Analysis/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/reviews.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Models

TF-IDF (unigrams + bigrams, 20k features) → Multinomial NB, Complement NB, Logistic Regression,
Linear SVM, Ridge, Passive-Aggressive.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. 80/20 stratified
split (balanced classes, so accuracy ≈ F1).

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| **Logistic Regression** | **0.8910** | **0.8910** |
| Multinomial NB | 0.8837 | 0.8837 |
| Complement NB | 0.8837 | 0.8837 |
| Ridge Classifier | 0.8837 | 0.8837 |
| Linear SVM | 0.8793 | 0.8793 |
| Passive Aggressive | 0.8553 | 0.8553 |

Tuned Logistic Regression (C=1): **accuracy 0.891, F1 0.891**.

## Key Findings

- **Logistic Regression on TF-IDF reaches 89% accuracy** on balanced positive/negative reviews, 
  a strong, fast baseline with no deep learning.
- **The models cluster tightly (0.86 to 0.89 F1)**, sentiment is largely carried by individual
  cue words ("great", "terrible", "refund"), which every linear/NB model picks up.
- **The gap to ~95%+ needs context the bag-of-words misses**, negation, sarcasm, and long-range
  dependencies ("not what I expected") cap a TF-IDF model; that's where transformers would help.
- **Tuning barely moves the needle**. TF-IDF + linear models are near their ceiling here;
  better features (n-grams, embeddings) matter more than hyperparameters.

## Tech Stack

- pandas, numpy, matplotlib, seaborn
- scikit-learn, nltk

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
