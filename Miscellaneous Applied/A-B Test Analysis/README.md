# A/B Test Analysis

A beginner-level **applied statistics** project that runs a controlled A/B experiment end-to-end, 
from data to a ship/don't-ship decision, using the two-proportion z-test, chi-square, confidence
intervals, and power analysis.

## Problem Statement
Given conversions from a control and a treatment group, decide whether the treatment genuinely lifts
conversion or the difference is noise. The experiment is **simulated with a known true effect**, so we
can verify the statistics actually recover the ground truth, the honest way to learn A/B analysis.

## Dataset
- **Simulated** (`utils.simulate_experiment`, seeded): 12,000 users/group, control 11.0% vs treatment
  12.8% conversion, a built-in **+0.018 absolute** true effect.

> Note: the checklist's Kaggle A/B dataset is no longer openly hosted; a seeded simulation is used
> instead. It is pedagogically stronger here because ground truth is known, so the test can be
> validated against it.

## Project Structure
```
A-B Test Analysis/
├── 01_eda.ipynb        # The simulated experiment, observed conversion rates
├── 02_analysis.ipynb   # z-test, chi-square, uplift CI, power / sample-size
├── utils.py · requirements.txt · README.md
└── data/  (none — data is generated in-notebook)
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **The result is significant**, z ≈ **5.5**, p < **0.0001**; the treatment lifts conversion
  **+2.3 points (~+21% relative)** in the sample.
- **The 95% CI on the uplift is [0.015, 0.031]**, it excludes 0 (significant) and **contains the true
  +0.018 effect**: the test correctly recovered ground truth.
- **Chi-square agrees** (p < 0.0001), z-test and chi-square are equivalent for a 2×2 table.
- **Power matters**, detecting the true +0.018 effect at 80% power needs ~5,000 users/group; using
  12,000 made the test well-powered. Under-powered tests miss real effects.
- **Decision: ship the treatment**, the lift is real, significant, and practically meaningful.

## Tech Stack
- numpy, pandas, matplotlib, scipy

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
