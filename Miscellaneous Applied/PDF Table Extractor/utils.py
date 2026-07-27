"""
PDF Table Extractor + Analyzer — Utility Functions
Extract tables from a PDF with pdfplumber and turn them into pandas DataFrames
for analysis. The sample `data/financial_report.pdf` is a generated financial
report with two tables (revenue by region, expenses by category).
"""
import pandas as pd
import pdfplumber

def extract_tables(path="data/financial_report.pdf"):
    """Return every table in the PDF as a list of raw (list-of-rows) tables."""
    tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    return tables

def to_dataframe(table, numeric_cols=None):
    """Header row + body -> DataFrame; optionally coerce columns to float."""
    df = pd.DataFrame(table[1:], columns=table[0])
    for c in (numeric_cols or []):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("[+%,]", "", regex=True), errors="coerce")
    return df
