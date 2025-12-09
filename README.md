# Predicting Kalshi NFL Outcome Prices

Charlie Williams & Garrett Kaufmann

Predictive modeling project that collects historical Kalshi NFL market prices and related game data, engineers features, trains models to forecast outcome prices, and evaluates performance for trading and market analysis.

---

## Table of contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Data](#data)
- [Usage](#usage)
- [Model & experiments](#model--experiments)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Authors & contact](#authors--contact)

---

## Overview
This repository collects Kalshi market snapshots and NFL game/context data, constructs time-aware features and market microstructure signals, trains predictive models for outcome prices, and evaluates model performance and simple trading strategies.

Use cases:
- Research price formation in binary event markets
- Build probabilistic pricing models
- Prototype strategy backtests and risk evaluation

## Motivation
- Understand signals that move Kalshi prices before and during games
- Combine sports data with market microstructure features
- Evaluate model utility for forecasting and trading decisions

## Repository structure
- `data/` — raw and processed datasets (raw files should be gitignored)
    - `data/subs_sentiment/` — sentiment data for each NFL subreddit
- `store_models/` - model files created from benchmarking different ML models
- `nfl_kalshi.py` - script to generate granular Kalshi NFL candlestick datasets
- `reddit_extract/reddit_sub_pulls.py` - script to generate reddit text data
- `create_timeseries.py` - script to join kalshi and reddit data into one time series
- `add_vader_sentiment_scores.py` - script to annotate time series with sentiment scores
- `assess_model.py` - script to train, test, and analyze model performance
- `README.md` — this file

## Getting started

Requirements
- Python 3.12+
- Recommended: virtual environment

Quick setup
1. Create and activate a virtual environment:
     - macOS / Linux:
         python -m venv .venv
         source .venv/bin/activate
     - Windows (PowerShell):
         python -m venv .venv
         .venv\Scripts\Activate.ps1
2. Install dependencies:
     pip install -r requirements.txt
3. Create a `.env` (or export env vars) for API keys:
     KALSHI_API_KEY=your_key_here
     KALSHI_API_SECRET=your_secret_here
     REDDIT_CLIENT_ID=your_clinet_id_here
    REDDIT_CLIENT_SECRET=your_reddit_secret_here
    REDDIT_USERNAME=your_reddit_username_here
    REDDIT_PASSWORD=your_reddit_password_here

Notes:
- Do not commit `.env` or API keys. Add them to `.gitignore`.

## Data

Sources
- Kalshi market snapshots / API (requires API access)
- NFL data: schedules, play-by-play, injuries from public providers (e.g., nflfastR, Sportradar, etc.)

## Usage

Collect Data
python nfl_kalshi.py
python reddit_extract/reddit_subs_pull.py
python create_timeseries.py

Sentiment Annotation
python add_vader_sentiment_scores.py

Train & assess model
python assess_model.py

## Authors & contact
- Charlie Williams
- Garrett Kaufmann

For questions or collaboration, open an issue in the repository or contact via repository profiles.

---