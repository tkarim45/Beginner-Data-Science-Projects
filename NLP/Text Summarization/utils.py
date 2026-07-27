"""
Text Summarization — Utility Functions
Extractive summarization: select the most important sentences from an article.
Two methods are compared against a lead-3 baseline and scored with ROUGE against
human reference summaries (CNN/DailyMail highlights):
  - TF-IDF scoring : rank sentences by the summed TF-IDF weight of their words
  - TextRank       : PageRank over a sentence-similarity graph

NLP-fundamentals project — does not use the classification template.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    sent_tokenize("test. test")
except Exception:
    def sent_tokenize(text):
        return re.split(r"(?<=[.!?])\s+", text.strip())


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_articles(path="data/articles.csv"):
    """Load article + reference summary pairs."""
    return pd.read_csv(path)


def split_sentences(text):
    """Sentence-tokenize, dropping very short fragments."""
    return [s.strip() for s in sent_tokenize(str(text)) if len(s.strip()) > 15]


# ─────────────────────────────────────────────
# Summarizers
# ─────────────────────────────────────────────

def summarize_lead(text, n=3):
    """Baseline: the first n sentences (very strong for news)."""
    sents = split_sentences(text)
    return " ".join(sents[:n])


def summarize_tfidf(text, n=3):
    """Score each sentence by the sum of its words' TF-IDF weights; take the top n (in order)."""
    sents = split_sentences(text)
    if len(sents) <= n:
        return " ".join(sents)
    vec = TfidfVectorizer(stop_words="english")
    M = vec.fit_transform(sents)
    scores = np.asarray(M.sum(axis=1)).ravel()
    top = sorted(sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)[:n])
    return " ".join(sents[i] for i in top)


def summarize_textrank(text, n=3, d=0.85, iters=50):
    """PageRank over a TF-IDF cosine sentence-similarity graph; take the top n sentences."""
    sents = split_sentences(text)
    if len(sents) <= n:
        return " ".join(sents)
    vec = TfidfVectorizer(stop_words="english")
    M = vec.fit_transform(sents)
    sim = cosine_similarity(M)
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = sim / row_sums
    N = len(sents)
    r = np.ones(N) / N
    for _ in range(iters):
        r = (1 - d) / N + d * (P.T @ r)
    top = sorted(sorted(range(N), key=lambda i: r[i], reverse=True)[:n])
    return " ".join(sents[i] for i in top)


# ─────────────────────────────────────────────
# Evaluation (ROUGE)
# ─────────────────────────────────────────────

def evaluate_rouge(df, summarizer, n=3, text_col="article", ref_col="summary"):
    """
    Run `summarizer` on each article and average ROUGE-1/2/L F-measure against the
    reference summary. Returns a dict of mean scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for _, row in df.iterrows():
        pred = summarizer(row[text_col], n=n)
        sc = scorer.score(str(row[ref_col]), pred)
        for k in agg:
            agg[k].append(sc[k].fmeasure)
    return {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
