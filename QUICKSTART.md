# Quick Start Guide

## 1. Setup Your Environment

```bash
# Navigate to project directory
cd cptracker

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure Your Handle

Open `cpintelligence_clean.ipynb` and change:

```python
USERNAME = "your_codeforces_handle"
```

Replace with your actual Codeforces username (e.g., "jalanashutosh06").

## 3. Run the Notebook

```bash
# Start Jupyter
jupyter notebook

# Then open cpintelligence_clean.ipynb in your browser
```

Or use Jupyter Lab:

```bash
jupyter lab
```

## 4. Execute Cells

- Click on a cell and press `Shift+Enter` to run it
- Run all cells from top to bottom

## 5. Use as a Module (Recommended for Scripts)

The intelligence analysis is now available as a reusable module:

```python
from intelligence import analyze_user, generate_report

# Analyze a single user
result = analyze_user("username")

if result["success"]:
    stats = result["stats"]
    print(f"Problems Solved: {stats['total_problems']}")
    print(f"Current Rating: {stats['current_rating']}")
    
    # Generate visual report
    generate_report(
        "username",
        result["stats"],
        result["df_practice"],
        result["df_contests"]
    )
else:
    print(f"Error: {result['error']}")
```

## 6. Use in Your Scraper

Integrate into [scraper.py](scraper.py):

```python
from intelligence import analyze_user

# Analyze users from usernames.txt
with open("usernames.txt", "r") as f:
    for username in f:
        username = username.strip()
        result = analyze_user(username)
        if result["success"]:
            stats = result["stats"]
            # Process stats as needed
```

## 7. Export Data for ML Training

Use [example_usage.py](example_usage.py) to extract and export data:

```bash
python example_usage.py
```

This creates Excel files with ML-ready data:
- **training_data_long.xlsx** - 1 row per user-topic (recommended for XGBoost)
- **training_data_wide.xlsx** - 1 row per user (alternative format)

### Long Format (Recommended)
Best for XGBoost training. Each row contains:
- User-level metrics (rating, peak_rating, avg_gap, etc.)
- Topic-specific metrics (problems_solved, avg_rating, difficulty_gap, etc.)

**User-Level Columns:**
```
current_rating, peak_rating, gap_to_peak, total_problems, 
avg_problem_rating, avg_gap, contest_count, avg_rating_change, 
rating_volatility, rating_trend, contest_problems_count,
practice_problems_count, avg_practice_problems_before_contests,
avg_rating_diff_practice_vs_contest
```

**Topic-Level Columns:**
```
topic, topic_problems_solved, topic_avg_rating, 
topic_max_rating, topic_avg_gap, topic_difficulty_gap
```

**New Metrics (v2):**
- `contest_problems_count`: Problems solved during contests (rated competitions)
- `practice_problems_count`: Problems solved during practice (between contests)
- `avg_practice_problems_before_contests`: Average practice problems per contest
- `avg_rating_diff_practice_vs_contest`: Practice avg rating - Contest avg rating
  - **Positive** = User practices at higher difficulty than contests → stronger skill development
  - **Negative** = Contest problems are harder → indicates competitive challenge

### Wide Format (Alternative)
1 row per user, topics as separate columns:
```
username, current_rating, peak_rating, ..., 
dp_problems_solved, dp_avg_rating, ..., 
greedy_problems_solved, greedy_avg_rating, ...
```

## 8. Train XGBoost Models

Use [ml_training_guide.py](ml_training_guide.py):

```bash
pip install xgboost scikit-learn openpyxl
python ml_training_guide.py
```

This trains two models:

### Model 1: Rating Predictor
Predicts user's Codeforces rating based on:
- Practice intensity (avg_gap)
- Contest performance (avg_rating_change, rating_trend)
- Topic-specific performance

### Model 2: Topic Recommender
Predicts difficulty gap for each topic:
- **Gap > 0** = User can handle harder problems → recommend advanced topics
- **Gap < 0** = User struggles → recommend fundamentals

## 9. Use Trained Models

```python
import pickle
import pandas as pd
from ml_training_guide import predict_rating_for_user, recommend_topics

# Load trained models
rating_model = pickle.load(open('rating_predictor.pkl', 'rb'))
topic_model = pickle.load(open('topic_recommender.pkl', 'rb'))

# For a new user
result = analyze_user("new_user")
features = extract_ml_features(result["username"], result["stats"], 
                                result["df_practice"], result["df_contests"])

# Predict rating
predicted_rating = rating_model.predict([features_array])

# Get topic recommendations
recommendations = recommend_topics(topic_model, features, all_topics)
for rec in recommendations[:5]:
    print(f"Priority {rec['priority']}: {rec['topic']}")
```

## Data Flow Summary

```
Codeforces API
     ↓
intelligence.py (analyze_user)
     ↓
extract_ml_features()
     ↓
Export to Excel (long/wide format)
     ↓
ml_training_guide.py (train XGBoost)
     ↓
Pickle models → Use for predictions & recommendations
```
- Wait for API calls to complete (usually 5-10 seconds)

## 5. View Results

Once complete, you'll see:
- ✓ Status messages showing data fetched
- ✓ Statistics table with topic breakdown
- ✓ 4-panel dashboard with visualizations

## Troubleshooting

**Issue: "API Error: handle ..."**
- Check your username is spelled correctly (case-sensitive)
- Verify the profile is public on Codeforces

**Issue: "Network error"**
- Check your internet connection
- Try again after a few seconds (API rate limit?)

**Issue: Empty charts**
- Ensure your profile has problems solved
- Check username is correct

## Tips

- The notebook works entirely offline after first run
- Save outputs: `File → Download as → PDF`
- Modify chart colors in matplotlib code
- Add filters for date ranges or problem difficulty

## Next Steps

- Push to GitHub: `git add . && git commit -m "Initial commit" && git push`
- Share your analysis
- Track progress over time by re-running the notebook
