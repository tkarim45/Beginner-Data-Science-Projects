# Netflix Content Analysis

A beginner-level **EDA / visualization** project exploring the Netflix catalog, 8,807 titles with
type, country, date added, rating, duration, and genres.

## Problem Statement
What does Netflix's catalog look like, movies vs shows, where content comes from, how it grew over
time, and which genres/ratings dominate? Exploratory analysis; insight + visualizations, no model.

## Dataset
- **Source**: [Netflix Titles](https://www.kaggle.com/datasets/shivamb/netflix-shows) (via HF `hugginglearners/netflix-shows`)
- **8,807 titles × 12 columns**: type, title, director, cast, country, date_added, release_year,
  rating, duration, listed_in (genres), description.

## Project Structure
```
Netflix Content Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, distributions
├── 02_analysis.ipynb   # Movie/TV split, countries, catalog growth, genres & ratings
├── utils.py · requirements.txt · README.md
└── data/netflix.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **~70% Movies (6,131) vs ~30% TV Shows (2,676).**
- **The US dominates** production (3,689 titles), then **India (1,046)** and the **UK (804)**, 
  Netflix's India investment is clearly visible.
- **Catalog additions peaked in 2019 (~1,999 titles)** and dipped afterward, content growth slowed post-2019.
- **International Movies, Dramas, and Comedies** are the largest genres; **TV-MA / TV-14** dominate ratings
  (a mature-leaning catalog).
- Netflix is movie-heavy, US-centric but globalizing, and matured its content pace after 2019.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
