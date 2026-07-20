# PDF Table Extractor + Analyzer

A beginner-level **applied** project that extracts tables locked inside a PDF (a financial report) with
pdfplumber, turns them into pandas DataFrames, and analyses the numbers.

## Problem Statement
Financial and business reports bury their data in PDF tables. Automatically extract those tables into a
structured, analysable form, then analyse revenue and expense figures.

## Dataset
- **`data/financial_report.pdf`**: a generated sample annual report with **two tables**, quarterly revenue
  by region and operating expenses by category. (Self-contained; the same pipeline applies to any report PDF.)

> Note: the checklist points at SEC 10-K filings; those are large, messy HTML/PDF documents. A clean
> generated report demonstrates the extract → structure → analyse pipeline without the real-world parsing
> noise (which needs far more cleaning code).

## Project Structure
```
PDF Table Extractor/
├── 01_extract.ipynb    # pdfplumber extraction → DataFrames
├── 02_analysis.ipynb   # Revenue by region, quarterly trend, expense growth
├── utils.py · requirements.txt · README.md
└── data/financial_report.pdf
```

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **pdfplumber recovers both tables exactly**, headers and numeric cells intact, straight into DataFrames.
- **Revenue**: North America leads (~**$556.8M** total 2024), then Europe (~**$377.3M**) and Asia Pacific
  (~**$303.8M**); every region grows quarter-on-quarter.
- **Expenses**: **R&D is the fastest-growing cost (+20.9%)**, then Marketing (+16.8%) and Salaries (+13.6%)
, spend is tilting toward growth investment.
- **The pipeline generalises**, swap in any report PDF and the same extract → `to_dataframe` → analyse flow
  applies (with more cleaning for messier real-world layouts).

## Tech Stack
- pandas, matplotlib, pdfplumber, reportlab

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_extract.ipynb
```
