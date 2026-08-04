# CFPB Complaint Investigation Prediction

Predicting which CFPB consumer complaints escalate to investigation, and testing
whether LLM-extracted narrative content adds predictive signal beyond simply
knowing a narrative exists.

## What is the CFPB?

The Consumer Financial Protection Bureau (CFPB) is a U.S. federal agency that
protects consumers in the financial sector. Consumers can submit complaints
about banks, lenders, credit reporting agencies, and other financial companies
directly to the CFPB. Most complaints get routed to the company for a response,
but a subset are escalated for further investigation — this project predicts
which complaints fall into that smaller, investigated group.

## Overview

This project builds an end-to-end ML pipeline on 14.6M CFPB consumer complaints:

1. **Data Wrangling & EDA** — cleaned and explored 14.6M complaints; found 26%
   (3.76M) include a consumer narrative
2. **Feature Engineering** — built 8 structured features; narrative text
   initially replaced with a binary `has_narrative` flag
3. **Model Training** — XGBoost classifier, ROC-AUC 0.85, PR-AUC 0.26. SHAP
   analysis showed `has_narrative` as the #2 most impactful feature globally
4. **LLM Enrichment** — used Claude's tool-use and Batches APIs to extract 5
   structured features directly from complaint narratives (harm type, severity,
   discrimination flag, resolution requested, sentiment), then measured uplift
   against the baseline model

## Key Finding

An enriched model (baseline + LLM-derived features) outperformed the baseline
on a held-out test set (ROC-AUC 0.946 vs. 0.938, PR-AUC 0.278 vs. 0.268).

SHAP analysis showed why: `has_narrative` — previously the #2 most important
feature — dropped to **zero importance** once the LLM-derived features were
available. The LLM features are extracted directly from the narrative text, so
they capture the same underlying signal `has_narrative` was only proxying for,
with far more precision.

**Caveat:** this result is based on a 10,000-row sample with a small positive
class (~239 train, ~60 test). The improvement is real and mechanistically
well-explained, but modest — a larger sample would strengthen confidence in
the effect size.

## Live Demo

Try it yourself: [[Streamlit app](https://cfpb-complaint-prediction-wjqtvlnw25lydkpkj5cnrg.streamlit.app)]

Paste a real complaint narrative and see the full pipeline run live — Claude
extracts structured features, the enriched XGBoost model predicts investigation
probability.

## Repository Contents

- `Notebook 1 - Data Wrangling and EDA.ipynb`
- `Notebook 2 - Feature Engineering.ipynb`
- `Notebook 3 - Model Training.ipynb`
- `Notebook 4 - LLM Enrichment.ipynb`
- `app.py` — Streamlit demo app

Note: raw data files and trained model artifacts are excluded from this repo
(see `.gitignore`) due to size. The notebooks are runnable against the public
CFPB Consumer Complaint Database.

## Tech Stack

Python, DuckDB, XGBoost, SHAP, Anthropic Claude API (tool use + Batches API),
Streamlit, AWS EC2

## Author

MD A Rahman Ruhel — [LinkedIn](https://linkedin.com/in/rahmanruhel) | [GitHub](https://github.com/RuhelHaq)
