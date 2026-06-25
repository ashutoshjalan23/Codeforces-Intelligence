# Codeforces Intelligence

A data-driven analytics and ML platform for competitive programmers. Analyzes Codeforces history, extracts temporal training patterns, and predicts rating changes contest by contest.

---

## Setup

```bash
git clone <repo-url>
cd codeforces-intelligence
pip install -r requirements.txt
```

### Reproduce the dataset and models from scratch

```bash
python scraper.py          # collect data → ml_dataset.csv (takes a while)
python rf_model.py         # train best model → models/rf_rating.pkl
```

### Or train all models

```bash
python xgboost_model.py
python rf_model.py
python lgbm_model.py
python ridge_model.py
python linear_model.py
python bracket_predictor.py   # requires rf_rating.pkl
```

### Run per-user analysis

```bash
python predictor.py <codeforces_username>
```

> **Note:** Model files are not committed to the repository. You must train at least `rf_model.py` before running `predictor.py`.

---

## What it does

- Fetches submission history and contest ratings from the Codeforces API
- Engineers contest-level features from practice patterns (windowed, topic-wise, difficulty-based)
- Builds a structured dataset where each row represents one contest entry for one user
- Trains and compares multiple regression models to predict rating change in the next contest
- Outputs feature importance and per-bracket performance breakdowns
- Runs a per-user analysis report with predicted delta, topic strengths/weaknesses, and peer comparison

---

## Project Structure

```
codeforces-intelligence/
├── intelligence.py         # API fetching, feature engineering, visualization
├── scraper.py              # Batch data collection pipeline with resume support
├── xgboost_model.py        # XGBoost training, evaluation, feature importance
├── linear_model.py         # Linear Regression model
├── ridge_model.py          # Ridge Regression model
├── rf_model.py             # Random Forest model (best MAE)
├── lgbm_model.py           # LightGBM model
├── bracket_predictor.py    # Per-rating-bracket RF models + per-bracket MAE breakdown
├── predictor.py             # Per-user report: predicted delta, topics, peer comparison
├── ml_dataset.csv          # Generated training dataset (one row per user per contest)
├── models/
│   ├── xgboost_rating.json
│   ├── rf_rating.pkl       # Used by predictor.py
│   ├── lgbm_rating.pkl
│   ├── ridge_rating.pkl
│   ├── linear_rating.pkl
│   └── bracket/            # Per-bracket RF models
├── usernames.txt
└── requirements.txt
```

---

## Pipeline

### 1. Collect usernames by division

```bash
python scraper.py
```

Calls `user.ratedList` to fetch all rated CF users, samples up to 1000 per division (Newbie → Grandmaster+), writes `usernames.txt`, then immediately starts building the dataset.

| Division | Rating Range |
|---|---|
| Newbie | < 1200 |
| Pupil | 1200 – 1399 |
| Specialist | 1400 – 1599 |
| Expert | 1600 – 1899 |
| Candidate Master | 1900 – 2099 |
| Master | 2100 – 2299 |
| International Master | 2300 – 2399 |
| Grandmaster+ | ≥ 2400 |

### 2. Dataset building (resumable)

The scraper writes each user's rows to `ml_dataset.csv` immediately after processing. If interrupted, re-running picks up from where it left off — already-processed users are detected from the CSV and skipped.

### 3. Train models

```bash
python xgboost_model.py      # XGBoost
python rf_model.py           # Random Forest — tries 300/500/1000 trees, saves best
python lgbm_model.py         # LightGBM
python ridge_model.py        # Ridge Regression
python linear_model.py       # Linear Regression
python bracket_predictor.py  # Per-bracket RF + breakdown (requires rf_rating.pkl)
```

All predictors split **by user** (80/20) to prevent data leakage, and evaluate against a zero-prediction baseline.

### 4. Run per-user analysis

```bash
python predictor.py <codeforces_username>
```

Fetches live data from the Codeforces API, builds the same feature set used during training, runs the RF model, and prints a report with predicted rating delta, topic strengths/weaknesses, and comparison against peers of similar rating.

---

## Features Engineered

Each row in `ml_dataset.csv` corresponds to one contest a user participated in. All features are computed from practice history **strictly before** that contest date.

### Practice volume (windowed)
- `problems_7d`, `problems_30d`, `problems_90d`

### Practice difficulty (windowed)
- `avg_rating_7d/30d/90d` — average problem rating solved
- `max_rating_30d/90d` — hardest problem solved
- `avg_gap_7d/30d/90d` — average (problem rating − user rating at time of solve)

### Difficulty distribution (last 90 days)
- `frac_sub1200`, `frac_1200_1600`, `frac_1600_2000`, `frac_2000_plus`

### Topic diversity
- `unique_topics_30d`, `unique_topics_90d`

### Per-topic stats (all prior history)
For each of: DP, Greedy, Graphs, Math, Implementation, Binary Search, Sortings, Strings, Trees, Two Pointers:
- `topic_<name>_count` — problems solved in that topic
- `topic_<name>_avg_rating` — average difficulty of solved problems

### Rating momentum
- `rating_before`, `peak_rating_so_far`, `gap_to_peak`
- `trend_last_3`, `trend_last_5` — average rating change over last 3/5 contests
- `rating_volatility` — std of all past rating changes
- `contests_90d` — contest frequency

### Activity cadence
- `days_since_last_contest`
- `days_since_last_practice`

### Target
- `target` — rating change in this contest (positive = gain, negative = loss)

---

## Model Comparison

All models trained on ~366K rows, 46 features, user-based 80/20 split.

| Model | MAE | Notes |
|---|---|---|
| Random Forest | **45** | Best overall; 300–1000 trees auto-tuned |
| LightGBM | 47 | Native null handling; fast training |
| XGBoost | 47 | Strong baseline; early stopping |
| Ridge Regression | — | Regularized linear; correlated features |
| Linear Regression | — | Weakest; no regularization |

`predictor.py` uses the Random Forest model (`models/rf_rating.pkl`).

### Per-bracket MAE (Random Forest)

| Bracket | MAE | R² | Baseline MAE |
|---|---|---|---|
| < 1200 | 40.68 | 0.747 | 82.21 |
| 1200 – 1600 | 43.42 | 0.284 | 51.20 |
| 1600 – 1900 | 49.21 | 0.217 | 56.39 |
| 1900 – 2100 | 50.95 | 0.174 | 56.65 |
| 2100 – 2400 | 47.59 | 0.232 | 55.09 |
| 2400+ | 49.67 | 0.263 | 58.42 |

Prediction is most reliable for users rated below 1200 (R²=0.75). For users above 1600, the model marginally beats the zero-delta baseline — contest performance at higher ratings is dominated by factors not captured in practice history alone.

---

## Data Scale

~9,500 users × ~39 contests average = ~366,000 rows. Rating distribution is naturally skewed: 56% of rows are from users rated below 1600.

Users with fewer than 10 contests are filtered before training via the `min_contests` parameter in `load_data`.

---

## Technologies

Python · Pandas · NumPy · Scikit-Learn · XGBoost · LightGBM · Matplotlib · Codeforces API
