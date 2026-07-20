# Toxic Comment Classification

A beginner-level **NLP text-classification** project that flags **toxic** online comments
(insults, threats, hate, obscenity) versus **clean** ones, using a TF-IDF + classic-ML pipeline.

## Problem Statement

Given an online comment, predict whether it is toxic, **binary** classification. Built on the
Jigsaw "Toxic Comment Classification" Wikipedia-comment data (the original is multi-label;
here we collapse it to a single toxic/clean target).

## Dataset

- **Source**: [`tasksource/jigsaw` (Hugging Face)](https://huggingface.co/datasets/tasksource/jigsaw), the Jigsaw / Conversation-AI Wikipedia comments
- **Subsample**: **40,000 comments**, 12,000 toxic + 28,000 clean (≈30% toxic, deliberately
  enriched from the ~10% natural rate so the minority class is learnable).

> Dataset note: the original Kaggle Jigsaw competition data is download-gated; `tasksource/jigsaw`
> mirrors the same `comment_text` + toxicity labels openly.

| Column | Description |
|---|---|
| `text` | Comment text |
| `label` | `1` = toxic, `0` = clean (target) |

## Project Structure

```
Toxic Comment Classification/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/comments.csv
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
| **Linear SVM** | **0.9153** | **0.9140** |
| Logistic Regression | 0.9076 | 0.9043 |
| Ridge Classifier | 0.9040 | 0.9009 |
| Multinomial NB | 0.9008 | 0.8973 |
| Complement NB | 0.8938 | 0.8953 |
| Passive Aggressive | 0.8878 | 0.8875 |

Tuned Logistic Regression (C=10): **accuracy 0.913, F1 0.912**.

## Key Findings

- **Linear SVM leads at ~91.5% accuracy / 0.914 F1**, explicit slurs and aggressive phrasing are
  strong lexical signals that TF-IDF captures well.
- **The minority (toxic) class is the hard part**, accuracy looks high partly because clean
  comments dominate; the weighted F1 (0.914) is the honest figure, and the confusion matrix in
  notebook 03 shows where toxic comments are missed.
- **Class enrichment matters**, training at ~30% toxic (vs the ~10% natural rate) gives the
  models enough toxic examples to learn the minority class instead of trivially predicting "clean".
- **Bag-of-words misses disguised toxicity**, obfuscated slurs, sarcasm, and context-dependent
  insults evade TF-IDF; closing that gap needs subword/transformer models.

## Tech Stack

- pandas, numpy, matplotlib, seaborn
- scikit-learn, nltk

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
