# FIFA Player Statistics Analysis

A beginner-level **EDA / visualization** project exploring 16,155 football players from the FIFA
game, age, overall rating, potential, market value, wage, position, and nationality.

## Problem Statement
Who are the best players, where do players come from, how are age and ratings distributed, and what
drives market value? Exploratory analysis; insight + visualizations, no model.

## Dataset
- **Source**: [FIFA players dataset](https://github.com/rashida048/Datasets) (`fifa.csv`)
- **16,155 players × 81 columns**: short/long name, age, dob, height, weight, nationality, club,
  league, overall, potential, value_eur, wage_eur, positions, and per-skill attributes.

## Project Structure
```
FIFA Player Statistics Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, age/overall/value distributions
├── 02_analysis.ipynb   # Best players, age & rating dists, nationalities, value drivers
├── utils.py · requirements.txt · README.md
└── data/fifa.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **16,155 players, mean age 24.8, mean overall 63.8**, a roughly bell-shaped rating curve centred in the low 60s.
- **Top-rated: Messi (93), Ronaldo (92), Robben (90).**
- **England (1,627), Spain, France, Argentina, Italy** supply the most players. Europe + South America dominate.
- **Value tracks rating** (correlation **0.57** raw; ~**0.8** on log-value) but explodes **non-linearly** at
  the top, elite ratings command exponential transfer fees.
- The rating distribution is bell-shaped; the superstar tail is thin and disproportionately valuable.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
