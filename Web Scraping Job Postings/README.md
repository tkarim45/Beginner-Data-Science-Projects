# Web Scraping + Analysis (Job Postings)

A beginner-level **applied** project that scrapes live remote-job postings from a public API and analyses
the in-demand skills, roles, and hiring companies, the full *scrape → structure → analyse* pipeline.

## Problem Statement
Collect current job postings by scraping, turn the raw response into a clean table, and analyse what the
market is hiring for. Demonstrates web scraping + light analysis end to end.

## Dataset
- **Source**: scraped live from the [RemoteOK public API](https://remoteok.com/api), `utils.fetch_jobs()`
  pulls fresh postings; **`data/jobs.csv`** is a saved snapshot (~100 postings) for reproducibility.
- **Columns**: position, company, location, tags (skills/categories), date.

> Note: the checklist's Glassdoor dataset is Kaggle-gated; scraping the RemoteOK API is a cleaner,
> reproducible demonstration of the actual skill (web scraping). RemoteOK does not reliably expose salary,
> so the analysis focuses on roles, skills, and companies.

## Project Structure
```
Web Scraping Job Postings/
├── 01_scrape_and_explore.ipynb  # The scraped data, how fetch_jobs() works
├── 02_analysis.ipynb            # Top tags/skills, companies, role-title words
├── utils.py · requirements.txt · README.md
└── data/jobs.csv
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **The scrape works end-to-end**, a single API call yields ~100 structured postings (role, company,
  location, tags).
- **Tags are the signal**, the most common cluster around business/ops roles (exec, customer support,
  marketing, ops) and functional skills; tags are how boards make listings searchable and are the cleanest
  field to analyse.
- **A few recruiting/agency accounts dominate** the top of the company distribution, a common job-board
  data artifact.
- **Honest limits**: a ~100-row snapshot of one board at one moment, with no salary field, good for
  demonstrating the scraping pipeline, not for market-wide conclusions.

## Tech Stack
- pandas, numpy, matplotlib (stdlib `urllib` for the scrape)

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_scrape_and_explore.ipynb
```
