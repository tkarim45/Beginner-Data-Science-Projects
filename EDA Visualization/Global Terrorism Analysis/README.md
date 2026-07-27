# Global Terrorism Analysis

A beginner-level **EDA / visualization** project exploring global terrorism from 1970 to 2017 to 181,691
attacks from the Global Terrorism Database (GTD) with country, region, attack type, target, weapon, and
perpetrator group.

## Problem Statement
Where and when do attacks concentrate, how do tactics break down, and who are the main actors?
Exploratory analysis; insight + visualizations, no model.

## Dataset
- **Source**: Global Terrorism Database (START, University of Maryland), via the openly hosted
  `synavate/csv_terrorism_pre_encoding_0x0.csv` mirror.
- **181,691 attacks × 15 columns**: eventid, year/month/day, country, region, city, lat/lon,
  attack type, target type, weapon type, group name (`gname`), success, suicide.

> Dataset note: the official GTD download requires START registration; this analysis uses an openly
> mirrored extract. That extract **excludes casualty counts**, so the analysis focuses on attack
> frequency, geography, tactics, and actors (not deaths/wounded).

## Project Structure
```
Global Terrorism Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, distributions
├── 02_analysis.ipynb   # Attacks over time, top countries/regions, attack & target types, groups
├── utils.py · requirements.txt · README.md
└── data/terrorism.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **181,691 attacks, 1970 to 2017**, with a dramatic **post-2011 surge peaking in 2014 (16,903 attacks)**, 
  the ISIL/Taliban era.
- **Concentrated in the Middle East/North Africa and South Asia**. Iraq (24,636), Pakistan (14,368),
  Afghanistan (12,731), India, and Colombia lead.
- **Bombings/explosions are the dominant tactic (~49%)**, then armed assault and assassination.
- **Taliban, ISIL, and Shining Path** are the most active *named* groups (though a large share of
  attacks are attributed to "Unknown").
- **89% of attacks "succeeded"** by the GTD's definition (the attack occurred as intended), a sobering base rate.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
