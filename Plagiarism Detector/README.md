# Plagiarism Detector

A beginner-level **applied NLP** project that detects near-duplicate / paraphrased text using
similarity features (TF-IDF cosine + character n-gram Jaccard), evaluated on a labelled paraphrase corpus.

## Problem Statement
Given two texts, decide whether one is a plagiarised/paraphrased version of the other. We treat it as a
similarity-threshold classifier and evaluate on **MRPC**, where each sentence pair is labelled
*paraphrase* (1) or *not* (0), a realistic, labelled proxy for plagiarism.

## Dataset
- **Source**: [Microsoft Research Paraphrase Corpus (MRPC)](https://huggingface.co/datasets/nyu-mll/glue) (GLUE `mrpc`)
- **`data/pairs.csv`**: **3,668 sentence pairs**, 67% paraphrase (`text1`, `text2`, `is_paraphrase`).

> Note: the checklist's UCI plagiarism dataset isn't openly downloadable; MRPC gives real,
> human-labelled near-duplicate pairs, the same detection problem.

## Project Structure
```
Plagiarism Detector/
├── 01_eda.ipynb        # Example pairs, similarity distribution by label
├── 02_analysis.ipynb   # Threshold tuning, precision/recall/F1, feature comparison
├── utils.py · requirements.txt · README.md
└── data/pairs.csv
```

## Method
Two training-free similarity features, **TF-IDF word cosine** and **character 3-gram Jaccard**, 
thresholded to flag plagiarism, with the threshold swept to maximise F1.

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **TF-IDF cosine at threshold ~0.30 gives F1 0.82** (accuracy ~0.71), a strong, transparent,
  training-free plagiarism flag.
- **Paraphrase pairs average 0.60 cosine vs 0.46** for unrelated pairs; the distributions overlap, which
  is why no single threshold separates them perfectly.
- **Character n-gram Jaccard** catches copy-paste-with-edits well but is weaker on true paraphrase
  (reworded, little word overlap), the two features are complementary.
- **The ceiling is semantic**, heavy paraphrasing that preserves meaning while changing words evades
  lexical similarity; catching that needs sentence embeddings / a trained model. An honest limit of the
  bag-of-words approach.

## Tech Stack
- pandas, numpy, matplotlib, scikit-learn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
