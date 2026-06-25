# Codeforces Intelligence — Research Findings

## Overview

This document summarizes key findings, model comparisons, and limitations observed during development of the Codeforces Intelligence rating prediction system. The goal was to determine whether a competitive programmer's practice history is predictive of their next contest rating change.

---

## 1. Dataset

- **Scale:** ~366,000 rows across ~9,500 users
- **Features:** 46 engineered features per row, covering practice volume, difficulty, topic distribution, rating momentum, and activity cadence
- **Target:** Rating change in a given contest (`target`), ranging from -412 to +1195, mean ≈ +25, std ≈ 87
- **Split:** By user (80/20) to prevent leakage — rows from the same user never appear in both train and test sets
- **Null handling:** Topic `avg_rating` columns are sparse by design; a user who has never solved a DP problem has a null `topic_dp_avg_rating`. XGBoost and LightGBM handle this natively; linear models require median imputation.

### Rating distribution

| Bracket | Rows | % |
|---|---|---|
| < 1200 | 83,792 | 22.8% |
| 1200 – 1600 | 113,597 | 30.9% |
| 1600 – 1900 | 70,667 | 19.2% |
| 1900 – 2100 | 39,874 | 10.8% |
| 2100 – 2400 | 33,537 | 9.1% |
| 2400+ | 26,730 | 7.3% |

The dataset is naturally skewed toward lower-rated users, mirroring the real Codeforces population. Higher-rated users have more rows per user on average (more contest history), but represent a smaller fraction of total rows.

---

## 2. Model Comparison

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (500 trees) | **45** | — | — |
| LightGBM (num_leaves=31) | 47 | — | — |
| XGBoost | 47 | — | — |
| Ridge Regression | — | — | — |
| Linear Regression | — | — | — |

Zero-delta baseline MAE: ~56 (always predicting no change).

**Random Forest achieved the lowest MAE**, outperforming both gradient boosting methods. This is likely because:

1. RF's bagging of independent trees smooths out noise in the target signal better than sequential boosting at this noise level
2. The high inherent randomness in contest outcomes means overfitting to subtle patterns (which boosting is prone to) hurts generalization
3. `min_samples_leaf=5` prevents fitting to sparse rows from lower-rated users with limited history

Gradient boosting methods (XGBoost and LightGBM) tied at MAE=47. Reducing LightGBM's `num_leaves` from 63 to 31 did not improve over XGBoost, suggesting both are hitting a similar noise floor rather than being capacity-limited.

Linear models were weakest, as expected given the non-linear relationships between practice patterns and rating outcomes.

---

## 3. Per-Bracket Analysis

### Finding: Predictability degrades sharply above 1200

| Bracket | MAE | R² | Baseline MAE | MAE reduction |
|---|---|---|---|---|
| < 1200 | 40.68 | **0.747** | 82.21 | 50.5% |
| 1200 – 1600 | 43.42 | 0.284 | 51.20 | 15.2% |
| 1600 – 1900 | 49.21 | 0.217 | 56.39 | 12.7% |
| 1900 – 2100 | 50.95 | 0.174 | 56.65 | 10.0% |
| 2100 – 2400 | 47.59 | 0.232 | 55.09 | 13.6% |
| 2400+ | 49.67 | 0.263 | 58.42 | 15.0% |

The model is highly effective for users rated below 1200 (R²=0.75, 50% reduction in error over baseline) and nearly ineffective above 1600 (R²=0.17–0.26, 10–15% reduction over baseline).

**Why lower-rated users are easier to predict:**
- Improvement is more directly tied to practice quantity and difficulty — solving harder problems reliably leads to rating gains
- Higher variance in outcomes (baseline MAE=82) gives the model more signal to capture
- Rating changes for new/improving players follow predictable patterns

**Why higher-rated users are harder to predict:**
- At 1600+, opponents are more consistent, so individual match variance dominates
- Contest difficulty selection, problem-specific familiarity, and execution on the day contribute significantly
- Practice history is a weaker signal relative to match-day factors that aren't captured in the dataset

### Finding: Bracket-specific models provide no improvement

Training separate RF models for each rating bracket yielded identical MAE to the global model (differences < 0.5 in every bracket). The global RF already captures bracket-specific patterns through `rating_before`, `peak_rating_so_far`, and `gap_to_peak`, making dedicated per-bracket models redundant.

---

## 4. Feature Insights

### Most predictive features (from RF/XGBoost importance)

Based on feature importance across models, the strongest signals are:

- **`rating_before`** — current rating is the strongest single predictor; higher-rated users regress toward the mean after gaining streaks
- **`trend_last_3` / `trend_last_5`** — recent momentum strongly predicts near-term direction
- **`gap_to_peak`** — distance from personal peak; users far below peak tend to recover, users near peak tend to consolidate
- **`avg_gap_30d`** — average difficulty of problems solved relative to rating; practicing above your level is positively correlated with improvement
- **`rating_volatility`** — high volatility users swing more; low volatility users are more stable regardless of direction

### Weaker signals

- **Topic-specific features** — sparse for most users (especially `trees`, `two_pointers`, `sortings`); meaningful only when a user has significant history in that topic
- **`days_since_last_practice`** — small effect; presence/absence of recent practice matters less than difficulty and diversity
- **`frac_*` difficulty distribution** — marginally useful; the raw `avg_rating` windows capture most of the same information

---

## 5. Limitations

### 5.1 Noise floor

Contest rating changes are fundamentally noisy. The randomness in problem selection, contest timing, and opponent pool cannot be predicted from practice history. The achievable MAE for this task likely has a floor around 40–45 rating points regardless of model complexity, representing irreducible variance.

### 5.2 Feature scope

The current feature set captures only what a user practiced, not:
- **Contest-specific difficulty** — a harder contest causes bigger swings for everyone
- **Opponent pool** — rating changes depend on the ratings of other participants
- **Problem-type fit** — whether the problems in a given contest match a user's strengths
- **Fatigue and consistency** — time of day, contest duration, previous contest performance in the same week

### 5.3 Sampling bias

Usernames were sampled up to 1000 per division. This produces a dataset that overrepresents the boundaries of each division (users who were recently promoted or demoted) and may underrepresent users who have been stagnant for a long time.

### 5.4 Temporal leakage risk

Features are computed from practice history strictly before each contest date, which prevents direct leakage. However, the train/test split is by user, not by time. A model trained on users active in 2020–2023 may not generalize to users competing in a different meta (e.g., after Codeforces changed its rating formula).

### 5.5 Sparse topic features

`topic_trees_avg_rating` is only 9.4% populated; `topic_two_pointers_avg_rating` is 20.3% populated. These features are effectively absent for the majority of users and may add noise rather than signal for lower-rated brackets.

### 5.6 Rating formula opacity

Codeforces uses a modified Elo system with unpublished adjustments. The exact mapping from performance to rating change is not fully public, which means the target variable itself contains systematic variance that no model can recover from practice features alone.

---

## 6. Potential Improvements

| Direction | Expected impact | Complexity |
|---|---|---|
| Add contest-level features (avg field rating, num participants) | High for 1600+ bracket | Medium |
| Time-based train/test split instead of user-based | Better temporal generalization | Low |
| Fill sparse topic features with user's `rating_before` as prior | Small improvement for sparse brackets | Low |
| Separate models per bracket with bracket-specific feature engineering | Low — bracket models already match global | High |
| Add recent contest performance (percentile within contest) | High | Medium |
| User embedding (learned representation per user) | Medium | High |

---

## 7. Conclusion

Practice history is a **strong predictor of rating change for users rated below 1200** and a **weak predictor above 1600**. The transition around 1200–1600 reflects the point where competitive programming outcomes shift from being practice-driven to being execution-and-variance-driven.

Random Forest outperformed gradient boosting at this task, which is unusual for structured tabular data but consistent with the high noise level in the target — RF's averaging reduces overfitting to noise patterns that boosting methods tend to exploit.

The primary value of this system is not precise rating prediction but the relative signal it provides: identifying which practice habits correlate with improvement within a user's current bracket, and comparing a user's practice patterns against peers of similar rating.
