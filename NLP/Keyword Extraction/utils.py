"""
Keyword Extraction — Utility Functions
Extract the most important words/phrases from an article with three classic,
unsupervised methods, evaluated by how well the extracted keywords overlap the
article's reference-summary content words (a proxy for "key" content):
  - TF-IDF   : top terms by TF-IDF weight
  - RAKE     : Rapid Automatic Keyword Extraction (phrase degree/frequency)
  - TextRank : PageRank over a word co-occurrence graph

NLP-fundamentals project — does not use the classification template.
"""

import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

STOP = set(ENGLISH_STOP_WORDS)
_WORD = re.compile(r"[a-z]+")


def load_articles(path="data/articles.csv"):
    return pd.read_csv(path)


def tokens(text):
    """Lowercase content tokens (alpha, non-stopword, len>2)."""
    return [w for w in _WORD.findall(str(text).lower()) if w not in STOP and len(w) > 2]


# ─────────────────────────────────────────────
# Keyword methods (single document)
# ─────────────────────────────────────────────

def keywords_tfidf(text, n=10):
    """Top-n terms of a single document by TF-IDF weight (fit on its own sentences)."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()]
    if len(sents) < 2:
        sents = [str(text)]
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z]{3,}")
    M = vec.fit_transform([s.lower() for s in sents])
    scores = np.asarray(M.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    return list(terms[np.argsort(scores)[::-1][:n]])


def keywords_rake(text, n=10):
    """RAKE: candidate phrases (split on stopwords/punctuation) scored by word degree/frequency."""
    text_l = str(text).lower()
    phrases = re.split(r"[^a-z]+", re.sub(r"\b(" + "|".join(STOP) + r")\b", "|", text_l))
    phrases = [p.strip().split() for p in " ".join(
        re.split(r"\b(?:" + "|".join(map(re.escape, STOP)) + r")\b", text_l)).split("  ") if p.strip()]
    freq, degree = Counter(), defaultdict(int)
    for ph in phrases:
        ph = [w for w in ph if len(w) > 2]
        if not ph:
            continue
        deg = len(ph) - 1
        for w in ph:
            freq[w] += 1
            degree[w] += deg + 1
    word_score = {w: degree[w] / freq[w] for w in freq}
    scored = []
    for ph in phrases:
        ph = [w for w in ph if len(w) > 2]
        if ph:
            scored.append((" ".join(ph), sum(word_score.get(w, 0) for w in ph)))
    seen, out = set(), []
    for kw, _ in sorted(scored, key=lambda x: x[1], reverse=True):
        if kw not in seen:
            seen.add(kw); out.append(kw)
        if len(out) >= n:
            break
    return out


def keywords_textrank(text, n=10, window=4, d=0.85, iters=30):
    """TextRank: PageRank over a word co-occurrence graph (sliding window)."""
    toks = tokens(text)
    if len(toks) < 2:
        return toks[:n]
    vocab = list(dict.fromkeys(toks))
    idx = {w: i for i, w in enumerate(vocab)}
    N = len(vocab)
    A = np.zeros((N, N))
    for i in range(len(toks)):
        for j in range(i + 1, min(i + window, len(toks))):
            a, b = idx[toks[i]], idx[toks[j]]
            if a != b:
                A[a, b] += 1; A[b, a] += 1
    rs = A.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    P = A / rs
    r = np.ones(N) / N
    for _ in range(iters):
        r = (1 - d) / N + d * (P.T @ r)
    return [vocab[i] for i in np.argsort(r)[::-1][:n]]


# ─────────────────────────────────────────────
# Evaluation (overlap with reference-summary content words)
# ─────────────────────────────────────────────

def evaluate(df, method, n=10, text_col="article", ref_col="summary"):
    """
    For each article, extract n keywords and compare to the set of content words in its
    reference summary. Returns mean precision@n and recall over the corpus.
    """
    precs, recs = [], []
    for _, row in df.iterrows():
        kws = method(row[text_col], n=n)
        kw_words = set(w for kw in kws for w in str(kw).split())
        ref = set(tokens(row[ref_col]))
        if not ref:
            continue
        hits = len(kw_words & ref)
        precs.append(hits / max(1, len(kw_words)))
        recs.append(hits / len(ref))
    return {"precision@k": round(float(np.mean(precs)), 4),
            "recall": round(float(np.mean(recs)), 4), "k": n}
