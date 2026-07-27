"""
Autocorrect Tool — Utility Functions
A Norvig-style spelling corrector: given a possibly-misspelled word, return the
most probable correct word using edit distance (1-2 edits) ranked by a word's
frequency in a reference corpus (Brown corpus word counts).

This is an NLP-fundamentals project — it does not use the classification template.
"""

import pandas as pd
from collections import Counter


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_frequencies(path="data/word_frequency.csv"):
    """Load the word→count frequency table into a Counter."""
    df = pd.read_csv(path)
    return Counter(dict(zip(df["word"].astype(str), df["count"].astype(int))))


# ─────────────────────────────────────────────
# Spelling Corrector (Norvig algorithm)
# ─────────────────────────────────────────────

class SpellCorrector:
    """Probabilistic spelling corrector: P(correction) ∝ corpus frequency."""

    LETTERS = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, freq):
        self.freq = freq
        self.total = sum(freq.values())

    def probability(self, word):
        """Relative frequency of `word` in the corpus."""
        return self.freq.get(word, 0) / self.total

    def known(self, words):
        """The subset of `words` that appear in the corpus."""
        return {w for w in words if w in self.freq}

    def edits1(self, word):
        """All strings one edit (delete/transpose/replace/insert) away from `word`."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.LETTERS]
        inserts = [L + c + R for L, R in splits for c in self.LETTERS]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """All strings two edits away from `word`."""
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def candidates(self, word):
        """Best candidate set: the word itself, then 1-edit, then 2-edit, then itself."""
        return (self.known([word]) or self.known(self.edits1(word))
                or self.known(self.edits2(word)) or [word])

    def correct(self, word):
        """Most probable correction of `word`."""
        word = word.lower()
        return max(self.candidates(word), key=self.probability)

    def correct_sentence(self, text):
        """Correct each alphabetic token in a sentence, preserving others."""
        out = []
        for tok in text.split():
            out.append(self.correct(tok) if tok.isalpha() else tok)
        return " ".join(out)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(corrector, pairs):
    """
    pairs: iterable of (wrong, correct). Returns accuracy and a per-item result list.
    A 'hit' is when the corrector maps the misspelling to the intended word.
    """
    rows, hits = [], 0
    for wrong, right in pairs:
        pred = corrector.correct(wrong)
        ok = pred == right
        hits += ok
        rows.append({"wrong": wrong, "expected": right, "predicted": pred, "correct": ok})
    acc = hits / len(rows) if rows else 0.0
    return acc, pd.DataFrame(rows)
