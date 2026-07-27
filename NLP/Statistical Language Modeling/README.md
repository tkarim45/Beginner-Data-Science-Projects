# Statistical Language Modeling

A beginner-level NLP project that builds classic **n-gram language models**
(unigram / bigram / trigram) from scratch, estimating word probabilities with
**Maximum Likelihood Estimation (MLE)**, making them robust with **add-k
smoothing** and **linear interpolation**, evaluating them with **perplexity**,
and **generating text** by sampling.

## What is a statistical language model?

A language model assigns a probability to a sequence of words, it answers
*"how likely is this sentence?"* and *"what word comes next?"*. Using the chain
rule plus the **Markov assumption** (a word depends only on the previous *n-1*
words), an **n-gram** model estimates each next-word probability from simple
counts:

$$P(w_i \mid \text{context}) = \frac{\text{count}(\text{context}, w_i)}{\text{count}(\text{context})}$$

- **Unigram** (n=1): each word independent of context, a bag of words.
- **Bigram** (n=2): condition on the previous word.
- **Trigram** (n=3): condition on the previous two words.

N-gram models are the historical foundation of language modeling, the same
*predict-the-next-token* objective that powers modern neural and transformer
LLMs, but learned by counting instead of gradient descent. They are the ideal
way to build intuition for **smoothing**, **out-of-vocabulary handling**, and
**perplexity** before touching neural models.

## Dataset

Two standard corpora from the **NLTK** library, combined:

| Corpus | Character | Sentences | Tokens | Word types |
|---|---|---|---|---|
| **Brown** | Balanced general American English (news, fiction, academic,...) | 57,101 | 988,331 | 41,018 |
| **Reuters** | Domain-specific financial newswire | 54,711 | 1,476,856 | 30,951 |
| **Combined** | Mixed everyday + specialised language | **111,812** | **2,465,187** | **57,611** |

Tokenization keeps alphanumeric tokens only and lowercases everything. The
vocabulary is heavily **Zipfian**: the top word (`the`) appears 139,248 times,
while **37.3%** of all word types are *hapax legomena* (seen exactly once). This
sparsity is the central modeling challenge, and the reason smoothing and
`<unk>` handling exist.

Corpora download automatically via NLTK (no manual data files):

```bash
python -m nltk.downloader brown reuters punkt punkt_tab
```

## Project Structure

```
Statistical Language Modeling/
├── 01_corpus_exploration.ipynb   # Load corpora, tokenize, vocab stats, Zipf plot
├── 02_ngram_modeling.ipynb       # MLE, smoothing, perplexity, interpolation, generation
├── ngram.py                      # Reusable NgramModel / InterpolatedNgramModel + corpus helpers
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

There is no `data/` folder, corpora are pulled from NLTK at runtime.

## Results: Perplexity by order & smoothing

Models are trained on an 80/20 train/test split (**89,449** train / **22,363**
test sentences) over an `<unk>`-capped vocabulary of **18,059** words (words seen
≥ 5 times). **Perplexity** = the average per-word "surprise" of the model on the
held-out set; **lower is better**.

| Model | Smoothing | Perplexity |
|---|---|---|
| Unigram | add-k (k=1, Laplace) | 849.5 |
| Unigram | add-k (k=0.01) | 849.0 |
| Bigram | add-k (k=1, Laplace) | 1,200.7 |
| Bigram | add-k (k=0.01) | 327.9 |
| Trigram | add-k (k=1, Laplace) | 5,767.6 |
| Trigram | add-k (k=0.01) | 1,327.7 |
| **Trigram (interpolated)** | **interp + k=0.001** | **227.0** |

Cleanest head-to-head (all at the best add-k, plus interpolation):

| Order | Perplexity ↓ |
|---|---|
| Unigram | 849.0 |
| Bigram | 327.9 |
| Trigram (naive add-k) | 1,327.7 |
| **Trigram (interpolated)** | **227.0** |

## Key Findings

- **Context is king.** Moving unigram → bigram cuts perplexity from **849 → 328**
, knowing just the previous word is enormously predictive of the next.
- **Naive high-order models over-smooth.** A plain add-k *trigram* (1,328) is
  actually *worse* than the bigram. With an 18k-word vocabulary almost every
  trigram context in the test set was never seen in training, so the estimate
  collapses onto the tiny smoothing prior. This is a real, important effect, not
  a bug.
- **Smoothing strength matters most at high order.** Laplace (`k=1`) is
  catastrophic for the trigram (5,768) and even hurts the bigram (1,201); shrinking
  `k` to 0.01 recovers most of the signal. The unigram barely moves, it has dense
  counts and little to smooth.
- **Interpolation is the fix.** Linearly blending trigram + bigram + unigram
  (Jelinek, Mercer) lets the model fall back on robust lower-order estimates when a
  context is unseen. The interpolated trigram reaches **227**, the best of all and
  the expected *higher order → lower perplexity*.
- **Generation quality tracks perplexity.** Unigram output is word salad; the
  bigram is locally plausible but drifts; the interpolated trigram produces the
  most newswire-like phrasing. All n-gram models still wander globally, exactly
  the long-range-coherence limitation that neural language models were built to
  solve.

## Reusable code: `ngram.py`

- `NgramModel(n, k)`, unigram/bigram/trigram with add-k (Laplace) smoothing;
  `.fit()`, `.prob()`, `.perplexity()`, `.generate()`.
- `InterpolatedNgramModel(lambdas, k)`, linear-interpolation trigram.
- Corpus helpers: `load_sentences()`, `train_test_split()`, `build_vocab()`,
  `replace_oov()` (`<unk>` mapping), `pad_sentence()`, and a one-shot
  `prepare_data()` pipeline.

```python
from ngram import prepare_data, InterpolatedNgramModel

train, test, vocab = prepare_data(("brown", "reuters"), min_count=5)
model = InterpolatedNgramModel(lambdas=(0.1, 0.3, 0.6)).fit(train, vocab=vocab)
print(model.perplexity(test))                 # 227.0
print(" ".join(model.generate(seed=["the"], length=15)))
```

## Tech Stack

- numpy
- matplotlib
- nltk (Brown + Reuters corpora)
- Pure-Python n-gram model (no scikit-learn needed)

## Getting Started

```bash
pip install -r requirements.txt
python -m nltk.downloader brown reuters punkt punkt_tab
jupyter notebook 01_corpus_exploration.ipynb
```

Run `01_corpus_exploration.ipynb` first (corpus stats + Zipf plots), then
`02_ngram_modeling.ipynb` (models, perplexity, generation).
