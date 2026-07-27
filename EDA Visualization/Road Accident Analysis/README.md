# Road Accident Analysis

A beginner-level **EDA / visualization** project exploring UK road collisions in 2021 to 101,087 records
from the Department for Transport with severity, vehicles, casualties, timing, speed limit, and
road/weather conditions.

## Problem Statement
When do crashes happen, how severe are they, and what conditions surround them? Exploratory analysis;
insight + visualizations, no model.

## Dataset
- **Source**: [UK DfT Road Safety Data, 2021 collisions](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) (`dft-road-casualty-statistics-collision-2021.csv`)
- **101,087 collisions × 15 columns**: date, day, time, collision_severity, number_of_vehicles,
  number_of_casualties, speed_limit, light/weather/road-surface conditions, urban/rural, road type, lat/lon.

## Project Structure
```
Road Accident Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, distributions
├── 02_analysis.ipynb   # Severity, timing (day/hour), speed limit & conditions, severity-vs-speed
├── utils.py · requirements.txt · README.md
└── data/accidents.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **101,087 collisions in 2021**: **77.5% slight, 21.1% serious, 1.5% fatal** (1,474 fatal crashes).
- **Crashes peak on Fridays and in the 15:00 to 18:00 evening rush** (school-run + commute), timing is
  dominated by traffic exposure.
- **~68% happen in urban areas** (more vehicles, junctions), but…
- **rural / high-speed roads are deadlier per crash**, the fatal share rises sharply from 30mph to
  60mph roads: higher speed → far higher lethality even though urban crashes are more frequent.
- The typical collision involves ~1.8 vehicles and ~1.3 casualties, a low-count urban prang; fatalities
  cluster on fast roads.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
