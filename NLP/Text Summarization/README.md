# Text Summarization

A beginner-level **NLP-fundamentals** project: extractive summarization of news articles, with
three classic methods benchmarked against human reference summaries using ROUGE.

## Problem Statement

Given a news article, produce a short summary by **selecting** its most important sentences
(extractive, no text generation). We compare three unsupervised methods and measure each with
ROUGE against the human-written reference summaries.

*This is an NLP-fundamentals project and does not use the classification template, it has its own
summarizers and a build/evaluate notebook flow.*

## Dataset

- **Source**: [CNN/DailyMail (`cnn_dailymail`, 3.0.0, Hugging Face)](https://huggingface.co/datasets/cnn_dailymail)
- **Sample**: **300 articles** (median ~480 words) each with a human reference summary
  (`highlights`, ~34 words → ~7% compression).

> Dataset note: the full CNN/DailyMail set is large; a 300-article streamed sample keeps the
> project laptop-friendly while giving a stable ROUGE estimate.

| File | Contents |
|---|---|
| `data/articles.csv` | `article`, `summary` (reference) |

## Project Structure

```
Text Summarization/
├── 01_eda.ipynb              # Article/summary lengths, compression, sample
├── 02_summarizers.ipynb      # Lead-3, TF-IDF, TextRank with example outputs
├── 03_evaluation.ipynb       # ROUGE-1/2/L over all articles + chart
├── utils.py                  # Summarizers + ROUGE evaluation
├── requirements.txt
├── README.md
└── data/articles.csv
```

Run notebooks in order: `01` → `02` → `03`.

## Methods

- **Lead-3**, the first 3 sentences (strong news baseline).
- **TF-IDF**, rank sentences by the summed TF-IDF weight of their words; take the top 3.
- **TextRank**. PageRank over a TF-IDF cosine sentence-similarity graph; take the 3 most central.

## Results

All figures produced by executing `03_evaluation.ipynb`, not assumed. ROUGE F-measure averaged
over 300 articles.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| **Lead-3** | **0.3099** | **0.1235** | **0.2106** |
| TextRank | 0.2833 | 0.1012 | 0.1951 |
| TF-IDF | 0.2397 | 0.0756 | 0.1594 |

## Key Findings

- **Lead-3 wins**, the trivial "take the first 3 sentences" baseline beats both TextRank and
  TF-IDF. This is the well-known **lead bias** of news writing: journalists front-load the key
  facts, so position is the single strongest importance signal.
- **TextRank beats TF-IDF**, graph centrality (a sentence matters if it's similar to many others)
  is a better importance measure than raw TF-IDF sentence weight, but neither overtakes lead-3.
- **Extractive ROUGE is capped** because the reference summaries are *abstractive* (reworded by
  humans), an extractive system can only reuse sentences verbatim. Still, it's fast, unsupervised,
  and never hallucinates.
- **Takeaway**: always benchmark a summarizer against lead-3 on news before claiming it "works", 
  the trivial baseline is famously hard to beat.

## Tech Stack

- pandas, numpy, matplotlib
- scikit-learn, nltk, rouge-score

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
