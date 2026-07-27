"""
Web Scraping + Analysis (Job Postings) — Utility Functions
Scrape live remote-job postings from the RemoteOK public API and analyse the
in-demand skills/tags, roles, and hiring companies. `data/jobs.csv` is a saved
snapshot for reproducibility; `fetch_jobs()` re-scrapes fresh data on demand.
"""
import json, urllib.request
import pandas as pd

def fetch_jobs():
    """Scrape current postings from the RemoteOK API -> DataFrame (live network call)."""
    req = urllib.request.Request("https://remoteok.com/api", headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=60).read())
    jobs = [j for j in data if isinstance(j, dict) and j.get("position")]
    return pd.DataFrame([{
        "position": j.get("position"), "company": j.get("company"),
        "location": j.get("location") or "Remote", "tags": ",".join(j.get("tags", [])),
        "date": j.get("date")} for j in jobs])

def load_jobs(path="data/jobs.csv"):
    """Load the saved snapshot (reproducible)."""
    return pd.read_csv(path)

def top_tags(df, n=15):
    return df["tags"].dropna().str.split(",").explode().str.strip().value_counts().head(n)
