"""
Chatbot (Rule-Based) — Utility Functions
A retrieval-based intent chatbot: match the user's message to the closest known
intent using TF-IDF cosine similarity over example patterns, then reply with that
intent's response. Below a similarity threshold it falls back to "I don't understand".

NLP-fundamentals project — does not use the classification template.
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_intents(path="data/intents.json"):
    """Load the intents file (tags, patterns, responses)."""
    with open(path) as f:
        return json.load(f)["intents"]


def intents_to_frame(intents):
    """Flatten intents into a (pattern, tag) DataFrame — one row per example pattern."""
    rows = [{"pattern": p, "tag": it["tag"]} for it in intents for p in it["patterns"]]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────

class RuleBasedChatbot:
    """TF-IDF retrieval chatbot with a confidence threshold and fallback reply."""

    FALLBACK = "Sorry, I didn't quite understand that. Could you rephrase?"

    def __init__(self, intents, threshold=0.2):
        self.intents = intents
        self.threshold = threshold
        self.responses = {it["tag"]: it["responses"] for it in intents}

    def fit(self, patterns, tags):
        self.tags = list(tags)
        self.vec = TfidfVectorizer(ngram_range=(1, 2))
        self.P = self.vec.fit_transform(patterns)
        return self

    def predict(self, text):
        """Return (best_tag, similarity score) for the closest matching pattern."""
        v = self.vec.transform([text])
        sims = cosine_similarity(v, self.P).ravel()
        i = int(sims.argmax())
        return self.tags[i], float(sims[i])

    def respond(self, text):
        """Reply with the matched intent's (first) response, or the fallback."""
        tag, score = self.predict(text)
        if score < self.threshold:
            return self.FALLBACK, "fallback", round(score, 3)
        return self.responses[tag][0], tag, round(score, 3)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(chatbot, X_test, y_test):
    """Intent-classification accuracy on held-out patterns."""
    preds = [chatbot.predict(t)[0] for t in X_test]
    acc = float(np.mean([p == y for p, y in zip(preds, y_test)]))
    return acc, preds
