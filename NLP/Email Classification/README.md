# Email Classification

A beginner-level **NLP text-classification** project that classifies emails as **spam** or
**legitimate (ham)** from their subject + body, using a TF-IDF + classic-ML pipeline.

## Problem Statement

Given an email's subject and body text, predict whether it is spam or ham, **binary**
classification on real-world corporate email.

## Dataset

- **Source**: [`SetFit/enron_spam` (Hugging Face)](https://huggingface.co/datasets/SetFit/enron_spam), the Enron spam corpus
- **Subsample**: **12,000 emails** (subject + body concatenated), spam vs ham.

> Note: this uses real email (Enron) rather than the SMS-style data in the existing *Message Spam
> Filtering* project, longer documents, email-specific vocabulary, and a different domain.

| Column | Description |
|---|---|
| `text` | Email subject + body |
| `label` | `spam` / `ham` (target) |

## Project Structure

```
Email Classification/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/emails.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Models

TF-IDF (unigrams + bigrams, 20k features) → Multinomial NB, Complement NB, Logistic Regression,
Linear SVM, Ridge, Passive-Aggressive.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. 80/20 stratified split;
weighted metrics.

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| **Linear SVM** | **0.9871** | **0.9870** |
| Ridge Classifier | 0.9871 | 0.9870 |
| Passive Aggressive | 0.9866 | 0.9866 |
| Multinomial NB | 0.9858 | 0.9858 |
| Logistic Regression | 0.9850 | 0.9850 |
| Complement NB | 0.9841 | 0.9841 |

Tuned Logistic Regression (C=10): **accuracy 0.9875, F1 0.9875**.

## Key Findings

- **Spam vs ham is highly separable, ~98.7% accuracy.** Spam uses distinctive vocabulary
  (promotions, links, money words) that TF-IDF + a linear model captures almost perfectly.
- **All six models score within 0.3 points**, the signal is so strong that model choice barely
  matters; Linear SVM and Ridge tie for the lead.
- **Email beats the harder Amazon-sentiment task** (0.987 vs 0.891), topic/keyword separation
  (spam) is easier than nuanced opinion (sentiment), which depends on negation and context.
- **A linear TF-IDF classifier is production-viable for spam filtering**, fast, interpretable
  (you can read the top spam-weighted tokens), and accurate.

## Tech Stack

- pandas, numpy, matplotlib, seaborn
- scikit-learn, nltk

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
