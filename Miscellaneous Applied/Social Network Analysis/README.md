# Social Network Analysis (Graph-Based)

A beginner-level **applied graph** project that analyses a real friendship network with networkx, 
degree structure, clustering, centrality (bridges vs hubs), and community detection.

## Problem Statement
Given a friendship graph, characterise its structure: how connected is it, who are the influential
nodes, are there communities? This is graph analytics, not tabular ML.

## Dataset
- **Source**: [SNAP Facebook combined ego-networks](https://snap.stanford.edu/data/ego-Facebook.html) (anonymised)
- **`data/facebook_edges.csv`**: **4,039 nodes, 88,234 undirected friendship edges**, one connected component.

## Project Structure
```
Social Network Analysis/
├── 01_eda.ipynb        # Size, density, degree distribution, clustering
├── 02_analysis.ipynb   # Betweenness (bridges) vs degree (hubs), communities, network plot
├── utils.py · requirements.txt · README.md
└── data/facebook_edges.csv
```

## Method
`networkx`: structural stats, betweenness centrality (k-sample approximation for speed), degree
centrality, and greedy-modularity community detection + a spring-layout visualization.

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **4,039 people, 88,234 friendships**; density ~0.011, **average clustering ~0.61**, friends-of-friends
  are usually friends, the hallmark of real social graphs.
- **Heavy-tailed degree distribution**, average degree ~44 but the top hub has **1,045 friends**.
- **Bridges ≠ hubs**: node **107** has the highest **betweenness (0.485)**, it lies on the most shortest
  paths and connects otherwise-separate groups, so removing it would fragment the network even though it
  isn't the highest-degree node.
- **13 communities** emerge from modularity optimisation (largest ~983 members), the ego-networks
  cluster into distinct friend groups.
- **Small-world structure**, high local clustering + short global paths: everyone is a few hops apart yet
  locally cliquey. Bridge nodes are the information/influence chokepoints.

## Tech Stack
- pandas, numpy, matplotlib, networkx

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
