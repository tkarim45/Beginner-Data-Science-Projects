# Olympics History Analysis

A beginner-level **EDA / visualization** project exploring 120 years of the Olympic Games, 
271,116 athlete-event records (1896 to 2016) with sport, medal, nationality, and body metrics.

## Problem Statement
Which nations dominate the medal table, how has female participation changed, and how have athletes'
bodies evolved over a century? Exploratory analysis; insight + visualizations, no model.

## Dataset
- **Source**: [120 Years of Olympic History](https://github.com/rgriff23/Olympic_history) (`athlete_events.csv`)
- **271,116 rows × 15 columns**: ID, Name, Sex, Age, Height, Weight, Team, NOC, Games, Year, Season,
  City, Sport, Event, Medal. **135,571 unique athletes**, 1896→2016.

## Project Structure
```
Olympics History Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, body-metric distributions
├── 02_analysis.ipynb   # Medal nations, female participation trend, height trend, top sports
├── utils.py · requirements.txt · README.md
└── data/athlete_events.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **271,116 records, 135,571 unique athletes, spanning 1896 to 2016.**
- **Medal powerhouses**: the **USA** leads by a wide margin (5,637 medal-events), then the **USSR (2,503)**,
  Germany, Great Britain, France.
- **Female participation exploded**: **7.4%** of Summer Olympians in 1936 → **45.5%** in 2016, one of
  the clearest gender-equality trends in sport.
- **Athletes trend taller and heavier** across the century (professionalisation + event specialisation).
- Athletics, Gymnastics, and Swimming dominate by event volume.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
