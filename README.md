# Codeforces Intelligence

**Codeforces Intelligence** is a data-driven analytics platform for competitive programmers. It analyzes a user's Codeforces history, extracts meaningful training patterns, identifies strengths and weaknesses, and uses machine learning to model rating growth.

The goal is to answer questions such as:

* Which topics am I strongest in?
* Which topics are holding me back?
* Am I practicing above my current level?
* What factors correlate most with rating growth?
* What rating can I realistically reach in the next few contests?

---

## Features

### User Analytics

* Codeforces API integration
* Contest rating history analysis
* Practice history analysis
* Problem difficulty tracking
* Topic-wise skill profiling

### Performance Metrics

* Current rating and progression
* Average solved problem rating
* Hardest solved problem
* Practice gap analysis
* Topic diversity analysis
* Difficulty progression tracking

### Visualization

* Rating trajectory
* Practice difficulty trends
* Topic skill profiles
* Learning progression heatmaps
* Rating vs practice correlations

### Machine Learning

* Rating growth prediction using XGBoost
* Feature importance analysis
* Training pattern discovery
* Performance forecasting

---

## Motivation
There are tons of resources availabe on the internet for competitive programming, but not of them stand out as personalised and guided. Also, programmers often forget to track their weak topics, focusing on solving only easy or problems in their comfort space. Codeforces intelligence tends to provide a meaningful to way for programmers to upskill and constantly track their imrpovement

---

## Progress Update

I have completed the initial data extraction and exploratory analysis using the Codeforces API.

One of the first questions I investigated was:

## Does solving more problems lead to a higher rating?

Across the users and data samples analyzed so far, there appears to be a positive relationship between problem-solving volume and rating. However, the relationship is not strong enough on its own to serve as a predictive feature. This reinforced the need for more informative metrics beyond simple aggregates such as Average Training Rating (ATR).

Current focus has shifted toward identifying and engineering stronger features, such as:

Practice difficulty relative to user rating
Practice volume over different time windows
Topic diversity
Topic-specific proficiency
Difficulty progression over time
Recent activity trends
Data Collection

The next milestone is building a large-scale scraper.

One challenge is that Codeforces does not expose a public API endpoint for retrieving all user handles or leaderboard rankings. If such an endpoint existed, collecting a representative user sample would be straightforward.

As a result, I am currently exploring alternative approaches for obtaining usernames, including:

Contest standings pages
Public ranking pages
Other publicly available Codeforces data sources
Dataset Goal

The current target is to collect data from approximately 500-800 users distributed across a broad rating spectrum, from Newbie to Grandmaster.

For each user, I plan to generate roughly 200-300 contest-level feature records, resulting in a dataset large enough for meaningful statistical analysis and machine learning experiments.

Next Steps
Build the username collection pipeline.
Implement large-scale data scraping and storage.
Engineer contest-level features.
Construct the training dataset.
Train baseline models before moving to XGBoost.

---

## Architecture

```text
Codeforces API
       │
       ▼
Data Collection Layer
       │
       ▼
Data Cleaning & Feature Engineering
       │
       ▼
Analytics Engine
       │
       ├── Visual Reports
       │
       └── ML Pipeline (XGBoost)
                │
                ▼
        Rating Growth Prediction
```

---

## Project Structure

```text
codeforces-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── feature_engineering.ipynb
│
├── src/
│   ├── collector.py
│   ├── analyzer.py
│   ├── feature_builder.py
│   ├── visualizer.py
│   └── predictor.py
│
├── models/
│   └── xgboost_model.json
│
├── reports/
│
├── requirements.txt
│
└── README.md
```

---

## Core Features Engineered

For every contest, the platform computes:

### Practice Features

* Number of problems solved in last 7 days
* Number of problems solved in last 30 days
* Average problem rating
* Median problem rating
* Maximum problem rating
* Practice gap

where

[
\text{Practice Gap}
===================

## \text{Problem Rating}

\text{Current User Rating}
]

### Topic Features

* Topic frequencies
* Topic diversity
* Topic-specific average ratings
* Topic growth trends

### Contest Features

* Previous rating
* Rating volatility
* Contest activity frequency
* Historical performance trends

---

## Machine Learning Pipeline

### Model

XGBoost Regressor

### Prediction Target

```text
Future Rating
```

or

```text
Rating Change in Next Contest
```

### Example Features

```text
Average Practice Difficulty
Practice Gap
Practice Volume
Topic Diversity
DP Skill Score
Graphs Skill Score
Math Skill Score
Recent Rating Trend
```

### Output

```text
Predicted Rating Gain: +83

Expected Rating:
1324 → 1407
```

---

## Example Insights

```text
Current Rating: 1250

Average Practice Difficulty: 1420

Average Practice Gap: +170

Strongest Topics:
- Graphs
- Math
- Greedy

Weakest Topics:
- DP
- Geometry

Predicted Next Rating:
1335
```

---

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Jupyter Notebook
* Codeforces API

---

## Future Roadmap

### Phase 1

* Data collection
* User reports
* Visualization dashboard

### Phase 2

* Topic skill scoring
* Practice gap analysis
* Progress heatmaps

### Phase 3

* XGBoost rating prediction
* Feature importance analysis

### Phase 4

* Personalized problem recommendations
* Weak-topic detection
* Training plan generation

### Phase 5

* Multi-user benchmarking
* Similar-user discovery
* Rating growth simulation

---

## Example Research Questions

* Do users who consistently solve problems above their rating improve faster?
* Which topics are most predictive of reaching Specialist?
* What practice habits distinguish Experts from Candidates?
* How much does topic diversity affect rating growth?

---

### Author

**Ashutosh Jalan**

Computer Science @ HKU

Competitive Programming • Machine Learning • Data Analytics


