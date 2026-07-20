# Keyword Extraction from Articles

A beginner-level **NLP-fundamentals** project: extract the most important words/phrases from news
articles using three classic unsupervised methods, benchmarked against the articles' reference
summaries.

## Problem Statement

Given an article, automatically surface its keywords/keyphrases (for tagging, indexing, or search).
We compare **TF-IDF**, **RAKE**, and **TextRank**, evaluating each by how well its keywords overlap
the content words of the article's human reference summary (a proxy for "key" content).

*This is an NLP-fundamentals project and does not use the classification template, it has its own
keyword methods and a methods/evaluate notebook flow.*

## Dataset

- **Source**: [CNN/DailyMail (Hugging Face)](https://huggingface.co/datasets/cnn_dailymail), 200 news articles + reference summaries.

> Dataset note: the checklist's "all-the-news" set is very large; a 200-article CNN/DailyMail sample
> (which conveniently ships reference summaries for the proxy evaluation) keeps this laptop-friendly.

| File | Contents |
|---|---|
| `data/articles.csv` | `article`, `summary` (the summary is used only for evaluation) |

## Project Structure

```
Keyword Extraction/
├── 01_eda.ipynb              # Text, vocabulary, why frequency alone fails
├── 02_methods.ipynb          # TF-IDF / RAKE / TextRank with example keywords
├── 03_evaluation.ipynb       # precision@10 / recall vs reference-summary words
├── utils.py                  # Keyword methods + evaluation
├── requirements.txt
├── README.md
└── data/articles.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Methods

- **TF-IDF**, words frequent in this article but rare across the corpus.
- **RAKE**. Rapid Automatic Keyword Extraction: candidate phrases (split on stopwords) scored by
  word degree/frequency; returns multi-word **phrases**.
- **TextRank**. PageRank over a word co-occurrence graph; the most central words.

## Results

All figures produced by executing `03_evaluation.ipynb`, not assumed. Top-10 keywords per article,
scored against reference-summary content words, averaged over 200 articles.

| Method | Precision@10 | Recall |
|---|---|---|
| **TextRank** | **0.3810** | 0.2268 |
| TF-IDF | 0.3540 | 0.2110 |
| RAKE | 0.0937 | **0.2317** |

## Key Findings

- **TextRank has the best precision** (P@10 **0.38**), graph centrality reliably surfaces the words
  that matter; **TF-IDF is close behind** (0.35).
- **RAKE has the best recall (0.23) but low precision (0.09)**, it returns multi-word *phrases*, so
  it covers more key content but dilutes per-word precision.
- **Single-word vs phrase extraction serve different needs**. TF-IDF/TextRank produce tag-like index
  terms; RAKE produces descriptive keyphrases. Pick by use case.
- **The proxy metric is imperfect** (a good keyword need not appear in the human summary) but gives a
  consistent, fully runnable comparison, and the qualitative examples in notebook 02 confirm the
  extracted keywords are on-topic.

## Tech Stack

- pandas, numpy, matplotlib
- scikit-learn

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
