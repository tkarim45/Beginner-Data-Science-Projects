# Song Lyrics Genre Classification

A beginner-level **NLP text-classification** project that predicts a song's **genre** (rap, pop,
rock, r&b) from its **lyrics**, using a TF-IDF + classic-ML pipeline.

## Problem Statement

Given the lyrics of a song, predict its genre, **multi-class** classification over 4 genres. A
genuinely harder text task than spam or topic classification, since genres share a lot of
everyday vocabulary.

## Dataset

- **Source**: [`sebastiandizon/genius-song-lyrics` (Hugging Face)](https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics)
- **Subsample**: a **balanced 6,000 songs**, 1,500 each of **rap, pop, rock, r&b** (English,
  lyrics ≥ 200 chars), streamed from the full corpus.

> Dataset note: the checklist's "scraped lyrics from 6 genres" set is large and Portuguese-heavy;
> this uses the openly hosted Genius lyrics corpus, balanced across the 4 genres with enough
> English data (the corpus is rap-dominated, so country/other were too sparse to include).

| Column | Description |
|---|---|
| `text` | Song lyrics |
| `label` | Genre: `rap` / `pop` / `rock` / `rb` (target) |

## Project Structure

```
Song Lyrics Genre Classification/
├── 01_eda.ipynb · 02_data_cleaning.ipynb · 03_model_building.ipynb
├── utils.py · requirements.txt · README.md
└── data/lyrics.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Models

TF-IDF (unigrams + bigrams, 20k features) → Multinomial NB, Complement NB, Logistic Regression,
Linear SVM, Ridge, Passive-Aggressive.

## Results

All figures produced by executing `03_model_building.ipynb`, not assumed. Balanced 4-class data,
80/20 stratified split; weighted metrics.

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| **Logistic Regression** | **0.6967** | **0.6937** |
| Ridge Classifier | 0.6892 | 0.6854 |
| Linear SVM | 0.6867 | 0.6832 |
| Passive Aggressive | 0.6708 | 0.6681 |
| Complement NB | 0.6400 | 0.6193 |
| Multinomial NB | 0.6358 | 0.6148 |

Tuned Logistic Regression (C=10): **accuracy 0.6925, F1 0.6895**.

## Key Findings

- **Genre from lyrics is hard, ~70% accuracy** (vs a 25% random baseline for 4 balanced classes).
  Logistic Regression leads, but the ceiling is far below the spam/resume tasks.
- **Why it's harder**: genres share a large common vocabulary (love, night, time…); the
  distinguishing signal is subtle (slang density, themes, repetition) rather than a few keyword
  cues, so TF-IDF separates them only partially.
- **rap is the most separable genre** (distinctive slang and vocabulary); pop/rock/r&b blur into
  each other, see the confusion matrix in notebook 03.
- **An honest, instructive result**: not every text-classification task is "easy". This is a good
  case for richer features (lyric structure, embeddings) or accepting that genre boundaries are
  inherently fuzzy.

## Tech Stack

- pandas, numpy, matplotlib, seaborn
- scikit-learn, nltk

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
