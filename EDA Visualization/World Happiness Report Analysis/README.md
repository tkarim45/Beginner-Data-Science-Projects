# World Happiness Report Analysis

A beginner-level **EDA / visualization** project exploring what makes countries happy, using the
2019 World Happiness Report, 156 countries scored on subjective wellbeing, decomposed into six
contributing factors.

## Problem Statement
The "happiness Score" is a survey-based ladder rating; the report attributes it to six factors
(GDP, social support, health, freedom, generosity, corruption). Which factors actually track
happiness, and who ranks top/bottom? This is exploratory analysis, the deliverable is insight +
visualizations, not a model.

## Dataset
- **Source**: [World Happiness Report 2019](https://worldhappiness.report/) (via HF `nateraw/world-happiness`)
- **156 countries × 10 columns**: Overall rank, Score, GDP per capita, Social support, Healthy life
  expectancy, Freedom, Generosity, Perceptions of corruption.

## Project Structure
```
World Happiness Report Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, distributions, top categories
├── 02_analysis.ipynb   # Rankings, factor correlations, GDP-vs-Score, heatmap
├── utils.py · requirements.txt · README.md
└── data/happiness.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **Happiest: Finland, Denmark, Norway. Least happy: South Sudan, Central African Republic, Afghanistan.**
- **Economy, health and social support dominate**. Score correlates **0.79** with GDP per capita,
  **0.78** with healthy life expectancy, **0.78** with social support.
- **Freedom matters moderately (0.57); corruption weakly (0.39).**
- **Generosity barely correlates (0.08)**, a generous population is *not* what makes a country score
  high; material security, health, and having someone to count on are.
- Read: national happiness is largely explained by wealth + health + social safety, not sentiment.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
