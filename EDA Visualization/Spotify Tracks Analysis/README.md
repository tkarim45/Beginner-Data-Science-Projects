# Spotify Tracks Analysis

A beginner-level **EDA / visualization** project exploring 114,000 Spotify tracks across 114 genres
with audio features (danceability, energy, valence…) and popularity, and testing whether the sound
of a track predicts how popular it is.

## Problem Statement
What do audio features look like across a huge track sample, which genres are most popular, and, 
the real question, **do audio features actually predict popularity?** Exploratory analysis; insight
+ visualizations, no model.

## Dataset
- **Source**: [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (HF)
- **114,000 tracks × 20 columns**: popularity, duration, explicit, danceability, energy, key, loudness,
  mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, track_genre.

## Project Structure
```
Spotify Tracks Analysis/
├── 01_eda.ipynb        # Overview: structure, missing, feature distributions
├── 02_analysis.ipynb   # Feature dists, top genres, popularity-vs-features, heatmap
├── utils.py · requirements.txt · README.md
└── data/spotify.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **114,000 tracks across 114 genres**; only **8.6% are explicit**.
- **Energy and danceability skew high; acousticness is bimodal** (a track is either acoustic or not).
- **Most popular genres** (by mean popularity): pop-film, k-pop, chill, sad, grunge.
- **Headline honest result: audio features barely predict popularity.** The strongest correlation is
  loudness at just **0.05**, every audio feature has **|r| ≤ 0.05** with popularity. What makes a track
  popular is artist, marketing, and playlisting, **not** how the audio sounds.
- Strong feature *inter*-correlations exist (energy↔loudness positive, energy↔acousticness negative),
  but none of that structure reaches the listener-popularity outcome.

## Tech Stack
- pandas, numpy, matplotlib, seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
