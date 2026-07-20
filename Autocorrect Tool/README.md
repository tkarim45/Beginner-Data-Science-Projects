# Autocorrect Tool

A beginner-level **NLP-fundamentals** project: a spelling corrector built from scratch using
Peter Norvig's classic edit-distance + word-frequency algorithm, no training, no neural network.

## Problem Statement

Given a possibly-misspelled word, return the most probable intended word. The method: generate
all real words within 1 to 2 edits (delete / transpose / replace / insert) and pick the one with the
highest frequency in a reference corpus, `argmax P(candidate)`.

*This is an NLP-fundamentals project and does not use the classification template, it has its own
`SpellCorrector` class and a build/evaluate notebook flow.*

## Dataset

- **Source**: word frequencies derived from the **NLTK Brown corpus** (no external download).
- **Vocabulary**: **24,866 unique words** with their corpus counts (words appearing ≥ 2 times).
- A small **test set of 43 common misspellings** (`data/test_misspellings.csv`) for evaluation.

| File | Contents |
|---|---|
| `data/word_frequency.csv` | word → count (the P(word) prior) |
| `data/test_misspellings.csv` | wrong → correct pairs for evaluation |

## Project Structure

```
Autocorrect Tool/
├── 01_eda.ipynb              # Vocabulary, Zipf's law, word lengths
├── 02_build_corrector.ipynb  # Norvig corrector: edits1/edits2, candidates, ranking
├── 03_evaluation.ipynb       # Accuracy on the misspelling test set + error analysis
├── utils.py                  # load_frequencies, SpellCorrector, evaluate
├── requirements.txt
├── README.md
└── data/
```

Run notebooks in order: `01` → `02` → `03`.

## Results

All figures produced by executing `03_evaluation.ipynb`, not assumed.

- **Correction accuracy: 0.881** (42 of 43 misspellings whose target is in the corpus are
  corrected to the intended word).
- Word frequencies follow **Zipf's law** (verified in notebook 01), which is exactly why frequency
  is a strong prior for ranking candidate corrections.

## Key Findings

- **Edit distance + frequency alone gets ~88%** of common misspellings right, a remarkably strong,
  fully transparent baseline with zero training.
- **Frequency is the decisive signal**: among candidates the same edit-distance away, the most
  common real word is almost always the intended one.
- **Failure modes**: misspellings more than 2 edits from the target, or cases where a very common
  short word outranks the intended word.
- **The corrector is context-free**, it cannot use surrounding words to choose between real-word
  confusions (their/there); that requires a language model (see the *Statistical Language Modeling*
  project).

## Tech Stack

- pandas, numpy, matplotlib
- nltk (Brown corpus)

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
