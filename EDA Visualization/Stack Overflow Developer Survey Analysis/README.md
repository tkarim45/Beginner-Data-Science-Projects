# Stack Overflow Developer Survey Analysis

A beginner-level **EDA / visualization** project exploring who developers are and what they use, 
89,184 responses to the 2023 Stack Overflow Developer Survey (country, experience, languages, remote
work, education, pay).

## Problem Statement
Where are developers, what do they code in, how do they work, and what do they earn? Exploratory
analysis; insight + visualizations, no model.

## Dataset
- **Source**: [Stack Overflow Annual Developer Survey 2023](https://survey.stackoverflow.co/) (via HF
  `yeper/stack-overflow-developer-survey`, `survey_2023.csv`)
- **89,184 responses**, reduced to 15 analysis-relevant columns: Age, Employment, RemoteWork, EdLevel,
  YearsCodePro, DevType, Country, LanguageHaveWorkedWith/WantToWorkWith, ConvertedCompYearly, Industry, OrgSize.

## Project Structure
```
Stack Overflow Developer Survey Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, distributions
├── 02_analysis.ipynb   # Countries, languages, remote work & education, compensation
├── utils.py · requirements.txt · README.md
└── data/survey.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **89,184 responses**; the community skews **US-heavy** (18,647) then Germany, India, UK, Canada.
- **JavaScript is still #1** (used by 55,711), with **HTML/CSS, Python, SQL, TypeScript** close behind, 
  the web stack dominates.
- **Hybrid + fully-remote outnumber in-person ~5:1**, remote/hybrid is now the developer norm post-2020.
- **Bachelor's is the modal education**, followed by Master's, though a meaningful minority are self-taught / no-degree.
- **Median converted annual compensation is ~\$75,000 USD** (heavily right-skewed, dominated by the large US sample).
- **Caveat**: Stack Overflow's audience self-selects (English-speaking, engaged devs), so this describes
  *SO's community*, not all developers worldwide.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
