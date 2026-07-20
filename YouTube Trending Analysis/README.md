# YouTube Trending Analysis

A beginner-level **EDA / visualization** project exploring engagement on trending YouTube videos, 
200,000 daily stat snapshots of ~11,000 trending videos (views, likes, dislikes, comments over time).

## Problem Statement
How do views, likes, and comments relate on trending videos, and what does audience sentiment look
like? Exploratory analysis; insight + visualizations, no model.

## Dataset
- **Source**: [Trending YouTube time-series (HF `jettisonthenet/...`)](https://huggingface.co/datasets/jettisonthenet/timeseries_trending_youtube_videos_2019-04-15_to_2020-04-15), subsampled to 200,000 snapshots.
- **Columns**: videostatsid, ytvideoid, views, comments, likes, dislikes, timestamp.

> Dataset note: the checklist's datasnaek YouTube CSV is no longer hosted; this openly available
> time-series version (stat snapshots per trending video) is used instead. It carries engagement
> metrics over time rather than title/category metadata, so the analysis focuses on engagement dynamics.

## Project Structure
```
YouTube Trending Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, distributions
├── 02_analysis.ipynb   # Engagement distributions, views-vs-likes/comments, like ratio, heatmap
├── utils.py · requirements.txt · README.md
└── data/youtube.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **200,000 snapshots across ~11,160 trending videos** (mean ~2.3M views, ~114k likes per snapshot).
- **Engagement scales together**, views correlate **0.84** with likes and **0.68** with comments.
- **Trending audiences are overwhelmingly positive**, median like ratio **0.976** (dislikes rare on
  trending content).
- All engagement metrics are heavily right-skewed (a few mega-viral videos dominate), hence log scales.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
