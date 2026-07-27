"""
ngram.py — Statistical (n-gram) language modeling toolkit.

A self-contained, dependency-light implementation of classic n-gram language
models (unigram / bigram / trigram) with:

  * Maximum-likelihood estimation (MLE) of conditional probabilities.
  * Add-k (Laplace when k=1) smoothing so unseen n-grams get non-zero mass.
  * Perplexity evaluation on held-out text.
  * Sampling-based text generation.

Plus corpus helpers for loading and tokenizing the NLTK Brown and Reuters
corpora, building a vocabulary with ``<unk>`` handling for out-of-vocabulary
words, and a reproducible train/test split.

The model is sentence-based: every sentence is padded with ``<s>`` (start) and
``</s>`` (end) markers so the model learns where sentences begin and end.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable, Sequence

# Sentence boundary markers and the out-of-vocabulary token.
BOS = "<s>"      # beginning of sentence
EOS = "</s>"     # end of sentence
UNK = "<unk>"    # unknown / out-of-vocabulary word


# ---------------------------------------------------------------------------
# Corpus loading & tokenization
# ---------------------------------------------------------------------------
def load_sentences(corpora: Sequence[str] = ("brown", "reuters"),
                   lowercase: bool = True,
                   alpha_only: bool = True) -> list[list[str]]:
    """Load and tokenize sentences from one or more NLTK corpora.

    Parameters
    ----------
    corpora : sequence of str
        NLTK corpus names. Supported: ``"brown"``, ``"reuters"``.
    lowercase : bool
        Lowercase every token (recommended — shrinks the vocabulary).
    alpha_only : bool
        Keep only alphanumeric tokens, dropping pure punctuation. This gives
        a cleaner word-level model.

    Returns
    -------
    list[list[str]]
        A list of sentences, each a list of token strings. Sentences are NOT
        yet padded with ``<s>`` / ``</s>`` — padding happens at train time so
        the same raw tokens can be reused for vocab building.
    """
    from nltk.corpus import brown, reuters

    available = {"brown": brown, "reuters": reuters}
    sentences: list[list[str]] = []
    for name in corpora:
        if name not in available:
            raise ValueError(f"Unsupported corpus {name!r}. "
                             f"Choose from {list(available)}.")
        for sent in available[name].sents():
            tokens = list(sent)
            if alpha_only:
                tokens = [t for t in tokens if t.isalnum()]
            if lowercase:
                tokens = [t.lower() for t in tokens]
            if tokens:
                sentences.append(tokens)
    return sentences


def train_test_split(sentences: Sequence[Sequence[str]],
                     test_size: float = 0.2,
                     seed: int = 42) -> tuple[list, list]:
    """Shuffle and split sentences into train / test sets reproducibly."""
    rng = random.Random(seed)
    idx = list(range(len(sentences)))
    rng.shuffle(idx)
    cut = int(len(sentences) * (1.0 - test_size))
    train = [list(sentences[i]) for i in idx[:cut]]
    test = [list(sentences[i]) for i in idx[cut:]]
    return train, test


def build_vocab(sentences: Iterable[Sequence[str]],
                min_count: int = 2) -> set[str]:
    """Build a vocabulary keeping words seen at least ``min_count`` times.

    Rare words are excluded; at apply time they are mapped to ``<unk>``. This
    both controls model size and lets the model assign probability mass to
    words it has never seen in test data.
    """
    counts: Counter[str] = Counter()
    for sent in sentences:
        counts.update(sent)
    vocab = {w for w, c in counts.items() if c >= min_count}
    vocab.update({BOS, EOS, UNK})
    return vocab


def replace_oov(sentences: Iterable[Sequence[str]],
                vocab: set[str]) -> list[list[str]]:
    """Map any token not in ``vocab`` to ``<unk>``."""
    return [[w if w in vocab else UNK for w in sent] for sent in sentences]


def pad_sentence(sentence: Sequence[str], n: int) -> list[str]:
    """Pad a sentence with start/end markers for an order-``n`` model.

    A bigram (n=2) model needs one ``<s>`` so the first real word has a
    context; a trigram (n=3) needs two. A single ``</s>`` lets the model learn
    to terminate sentences.
    """
    n_start = max(n - 1, 1)
    return [BOS] * n_start + list(sentence) + [EOS]


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Return all contiguous n-grams (as tuples) from a token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class NgramModel:
    """An n-gram language model with add-k smoothing.

    Parameters
    ----------
    n : int
        Order of the model. ``1`` = unigram, ``2`` = bigram, ``3`` = trigram.
    k : float
        Add-k smoothing constant. ``k=0`` is pure MLE (no smoothing — any
        unseen n-gram has probability 0, so perplexity is undefined/infinite
        on unseen test events). ``k=1`` is classic Laplace smoothing.

    Notes
    -----
    The model stores, for every context (the first ``n-1`` tokens of an
    n-gram), a Counter of the following words. Smoothed probability is::

        P(w | context) = (count(context, w) + k)
                         / (count(context) + k * V)

    where ``V`` is the vocabulary size.
    """

    def __init__(self, n: int, k: float = 1.0):
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2, or 3.")
        if k < 0:
            raise ValueError("k must be >= 0.")
        self.n = n
        self.k = k
        # context-tuple -> Counter of next-word counts
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        # context-tuple -> total count of that context
        self.context_totals: dict[tuple[str, ...], int] = defaultdict(int)
        self.vocab: set[str] = set()
        self.V: int = 0

    # -- training -----------------------------------------------------------
    def fit(self, sentences: Sequence[Sequence[str]],
            vocab: set[str] | None = None) -> "NgramModel":
        """Train the model from a list of (already OOV-mapped) sentences."""
        if vocab is None:
            vocab = set()
            for sent in sentences:
                vocab.update(sent)
            vocab.update({BOS, EOS, UNK})
        self.vocab = vocab
        self.V = len(vocab)

        for sent in sentences:
            padded = pad_sentence(sent, self.n)
            for gram in ngrams(padded, self.n):
                context, word = gram[:-1], gram[-1]
                self.counts[context][word] += 1
                self.context_totals[context] += 1
        return self

    # -- probability --------------------------------------------------------
    def prob(self, word: str, context: tuple[str, ...] = ()) -> float:
        """Smoothed conditional probability P(word | context)."""
        context = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        ctx_total = self.context_totals.get(context, 0)
        word_count = self.counts.get(context, {}).get(word, 0)
        return (word_count + self.k) / (ctx_total + self.k * self.V)

    def logprob(self, word: str, context: tuple[str, ...] = ()) -> float:
        """Base-2 log of the smoothed conditional probability."""
        p = self.prob(word, context)
        return math.log2(p) if p > 0 else float("-inf")

    # -- evaluation ---------------------------------------------------------
    def perplexity(self, test_sentences: Sequence[Sequence[str]]) -> float:
        """Perplexity over a set of test sentences.

        Perplexity = 2 ** (cross-entropy), the average per-token surprise.
        Lower is better. OOV words in the test set are mapped to ``<unk>``
        (so the score is well-defined even on unseen vocabulary). With ``k=0``
        any unseen n-gram drives perplexity to infinity.
        """
        log_prob_sum = 0.0
        n_tokens = 0
        for sent in test_sentences:
            sent = [w if w in self.vocab else UNK for w in sent]
            padded = pad_sentence(sent, self.n)
            for gram in ngrams(padded, self.n):
                context, word = gram[:-1], gram[-1]
                lp = self.logprob(word, context)
                if lp == float("-inf"):
                    return float("inf")
                log_prob_sum += lp
                n_tokens += 1
        if n_tokens == 0:
            return float("inf")
        cross_entropy = -log_prob_sum / n_tokens
        return 2 ** cross_entropy

    # -- generation ---------------------------------------------------------
    def generate(self, seed: Sequence[str] | None = None,
                 length: int = 20, seed_random: int | None = None) -> list[str]:
        """Generate text by sampling from the model's distributions.

        Parameters
        ----------
        seed : sequence of str, optional
            Optional starting words. The model continues from here. If not
            given it starts from sentence-start padding.
        length : int
            Maximum number of words to generate (excluding markers). Stops
            early if ``</s>`` is sampled.
        seed_random : int, optional
            RNG seed for reproducible generation.
        """
        rng = random.Random(seed_random)
        n_start = max(self.n - 1, 1)
        history = [BOS] * n_start
        if seed:
            history += [w.lower() for w in seed]

        out: list[str] = list(seed) if seed else []
        for _ in range(length):
            context = tuple(history[-(self.n - 1):]) if self.n > 1 else ()
            dist = self.counts.get(context)
            if not dist:
                # Unknown context — fall back to the unigram (empty) context.
                dist = self.counts.get((), None)
                context = ()
                if not dist:
                    break
            words = list(dist.keys())
            weights = [self.prob(w, context) for w in words]
            total = sum(weights)
            if total <= 0:
                break
            weights = [w / total for w in weights]
            nxt = rng.choices(words, weights=weights, k=1)[0]
            if nxt == EOS:
                break
            if nxt not in (BOS, UNK):
                out.append(nxt)
            history.append(nxt)
        return out


# ---------------------------------------------------------------------------
# Interpolated model (linear / Jelinek-Mercer interpolation)
# ---------------------------------------------------------------------------
class InterpolatedNgramModel:
    """A trigram model with linear interpolation over lower orders.

    A naive add-k trigram badly over-smooths on a corpus of this size: most
    trigram contexts in held-out text were never seen in training, so the
    estimate collapses to the smoothing prior and perplexity *rises* above the
    bigram. The textbook fix is interpolation — blend the trigram, bigram and
    unigram estimates::

        P(w | w_2, w_1) = l3 * P_tri  +  l2 * P_bi  +  l1 * P_uni

    with ``l1 + l2 + l3 = 1``. The lower-order models supply robust mass when
    the high-order context is sparse, so the interpolated trigram genuinely
    beats the bigram. Each sub-model keeps a light add-k floor for unseen
    unigrams (``<unk>`` handling).
    """

    def __init__(self, lambdas: tuple[float, float, float] = (0.1, 0.3, 0.6),
                 k: float = 0.001):
        l1, l2, l3 = lambdas
        if abs((l1 + l2 + l3) - 1.0) > 1e-9:
            raise ValueError("lambdas must sum to 1.0 (uni, bi, tri).")
        self.lambdas = lambdas
        self.k = k
        self.uni = NgramModel(1, k=k)
        self.bi = NgramModel(2, k=k)
        self.tri = NgramModel(3, k=k)
        self.vocab: set[str] = set()

    def fit(self, sentences, vocab=None):
        self.uni.fit(sentences, vocab=vocab)
        self.bi.fit(sentences, vocab=vocab)
        self.tri.fit(sentences, vocab=vocab)
        self.vocab = self.uni.vocab
        return self

    def prob(self, word, context=()):
        l1, l2, l3 = self.lambdas
        ctx = tuple(context)
        p_uni = self.uni.prob(word, ())
        p_bi = self.bi.prob(word, ctx[-1:]) if ctx else p_uni
        p_tri = self.tri.prob(word, ctx[-2:]) if len(ctx) >= 2 else p_bi
        return l1 * p_uni + l2 * p_bi + l3 * p_tri

    def perplexity(self, test_sentences):
        log_prob_sum = 0.0
        n_tokens = 0
        for sent in test_sentences:
            sent = [w if w in self.vocab else UNK for w in sent]
            padded = pad_sentence(sent, 3)
            for gram in ngrams(padded, 3):
                context, word = gram[:-1], gram[-1]
                p = self.prob(word, context)
                if p <= 0:
                    return float("inf")
                log_prob_sum += math.log2(p)
                n_tokens += 1
        if n_tokens == 0:
            return float("inf")
        return 2 ** (-log_prob_sum / n_tokens)

    def generate(self, seed=None, length=20, seed_random=None):
        """Sample text from the interpolated trigram distribution."""
        rng = random.Random(seed_random)
        history = [BOS, BOS]
        if seed:
            history += [w.lower() for w in seed]
        out = list(seed) if seed else []
        words = [w for w in self.vocab if w not in (BOS, UNK)]
        for _ in range(length):
            context = tuple(history[-2:])
            # Restrict candidate set to words the trigram/bigram context has
            # actually seen (plus a unigram fallback) — keeps sampling fast.
            cand = set(self.tri.counts.get(context, {}).keys())
            cand |= set(self.bi.counts.get(context[-1:], {}).keys())
            if not cand:
                cand = set(self.uni.counts.get((), {}).keys())
            cand = [w for w in cand if w != BOS]
            if not cand:
                break
            weights = [self.prob(w, context) for w in cand]
            total = sum(weights)
            if total <= 0:
                break
            weights = [w / total for w in weights]
            nxt = rng.choices(cand, weights=weights, k=1)[0]
            if nxt == EOS:
                break
            if nxt != UNK:
                out.append(nxt)
            history.append(nxt)
        return out


# ---------------------------------------------------------------------------
# Convenience one-shot pipeline
# ---------------------------------------------------------------------------
def prepare_data(corpora: Sequence[str] = ("brown", "reuters"),
                 test_size: float = 0.2,
                 min_count: int = 2,
                 seed: int = 42):
    """End-to-end data prep: load -> split -> build vocab -> map OOV.

    Returns ``(train, test, vocab)`` where ``train``/``test`` are lists of
    sentences with rare words already replaced by ``<unk>``.
    """
    sentences = load_sentences(corpora)
    train, test = train_test_split(sentences, test_size=test_size, seed=seed)
    vocab = build_vocab(train, min_count=min_count)
    train = replace_oov(train, vocab)
    test = replace_oov(test, vocab)
    return train, test, vocab
