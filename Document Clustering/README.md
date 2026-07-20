# Document Clustering

A beginner-level **unsupervised text clustering** project that groups newsgroup posts into topics using K-Means on TF-IDF vectors.

## Problem Statement

Given a collection of raw text documents from six different newsgroups, can we discover the underlying topics without any labels? We convert each document into a TF-IDF vector and apply K-Means clustering, then measure how well the unsupervised partition recovers the true newsgroup categories.

There is **no target column used during training**, this is genuine unsupervised learning. The true newsgroup labels are used **only** to evaluate cluster quality after the fact, via metrics such as Adjusted Rand Index and V-measure.

## Dataset

- **Source**: [20 Newsgroups, scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
- **Subset used**: 6 categories (fetched via `sklearn.datasets.fetch_20newsgroups`)
- **Documents**: 5,631 (after removing near-empty and duplicate posts)
- **Preprocessing**: headers, footers, and quoted replies removed at fetch time

| Category | Documents |
|---|---|
| rec.sport.hockey | 967 |
| sci.med | 950 |
| sci.space | 945 |
| comp.graphics | 943 |
| rec.autos | 920 |
| talk.politics.mideast | 906 |

## Project Structure

```
Document Clustering/
├── 01_eda.ipynb              # Exploratory Data Analysis
├── 02_data_cleaning.ipynb    # Text cleaning & saving cleaned CSV
├── 03_model_building.ipynb   # TF-IDF + K-Means clustering
├── utils.py                  # Reusable text clustering helpers
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    └── newsgroups.csv        # Cleaned text + true_category (written by notebook 02)
```

Run notebooks in order: `01 → 02 → 03`. Notebook 02 writes `data/newsgroups.csv` which notebook 03 reads.

## Results

K-Means is fit for k ∈ {2, …, 10}. The model is evaluated at **k = 6** (matching the true number of categories).

### External clustering metrics (k = 6)

| Metric | Score | Interpretation |
|---|---|---|
| Adjusted Rand Index (ARI) | 0.2269 | Above-chance agreement with true labels |
| Normalized Mutual Info (NMI) | 0.4808 | ~48 % of label entropy captured by clusters |
| Homogeneity | 0.4179 | Individual clusters are reasonably pure |
| Completeness | 0.5658 | Most category members share a cluster |
| **V-measure** | **0.4808** | Harmonic mean of hom. and completeness |

### Cluster mapping

| Cluster | Size | Dominant category | Match |
|---|---|---|---|
| 0 | 403 | rec.autos | Clean (401 / 403 autos docs) |
| 1 | 71 | sci.med (noise cluster) | Weak, very small |
| 2 | 3,159 | Mixed (catch-all) | sci.space, sci.med, comp.graphics bleed together |
| 3 | 740 | comp.graphics | Moderate (612 / 740 graphics docs) |
| 4 | 696 | rec.sport.hockey | Clean (695 / 696 hockey docs) |
| 5 | 562 | talk.politics.mideast | Clean (559 / 562 mideast docs) |

## Key Findings

- **rec.sport.hockey** and **talk.politics.mideast** cluster almost perfectly, their vocabulary (*goal, nhl, players* vs *israel, armenian, jews*) is highly specific and non-overlapping.
- **rec.autos** also separates cleanly; terms like *car, engine, dealer, oil* are distinctive.
- **sci.space**, **sci.med**, and **comp.graphics** collapse into one large catch-all cluster, all three use overlapping general-purpose technical vocabulary and shorter jargon-free posts that TF-IDF cannot easily discriminate.
- Completeness (0.57) exceeds homogeneity (0.42), which is typical in text clustering: individual clusters are fairly pure but a single newsgroup (especially sci categories) may spread across multiple clusters.
- The TruncatedSVD 2D projection shows the sport and politics groupings as visually tight, distinct clouds, while the science/computing documents form a diffuse central mass.

## Tech Stack

- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- wordcloud

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
