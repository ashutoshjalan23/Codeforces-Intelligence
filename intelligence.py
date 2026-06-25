"""
Codeforces Intelligence Analysis Module

Provides functions to fetch, analyze, and visualize Codeforces user data.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def fetch_user_data(username):
    """
    Fetch submission history and rating data from Codeforces API.
    
    Args:
        username (str): Codeforces user handle
        
    Returns:
        tuple: (submissions list, ratings_data list) or (None, None) if error
    """
    try:
        submissions_url = f"https://codeforces.com/api/user.status?handle={username}&from=1&count=10000"
        ratings_url = f"https://codeforces.com/api/user.rating?handle={username}"
        
        submissions_resp = requests.get(submissions_url, timeout=15).json()
        ratings_resp = requests.get(ratings_url, timeout=15).json()
        
        if submissions_resp["status"] != "OK" or ratings_resp["status"] != "OK":
            raise Exception(f"API Error: {submissions_resp.get('comment', 'Unknown error')}")
        
        submissions = submissions_resp["result"]
        ratings_data = ratings_resp["result"]
        
        return submissions, ratings_data
        
    except requests.exceptions.RequestException as e:
        print(f"Network error for {username}: {e}")
        return None, None
    except Exception as e:
        print(f"Error fetching data for {username}: {e}")
        return None, None


def process_submissions(submissions, ratings_data):
    """
    Clean, deduplicate, and process submission data.
    
    Args:
        submissions (list): Raw submissions from API
        ratings_data (list): Contest ratings from API
        
    Returns:
        tuple: (df_practice DataFrame, df_contests DataFrame)
    """
    seen = set()
    unique_submissions = []
    
    for sub in submissions:
        prob = sub.get("problem", {})
        contest_id = prob.get("contestId")
        index = prob.get("index")
        
        if contest_id and index:
            key = (contest_id, index)
            if key not in seen:
                seen.add(key)
                unique_submissions.append(sub)
    
    practice_data = []
    contests_data = []
    contest_times = set()
    
    for contest in ratings_data:
        date_ts = contest.get("ratingUpdateTimeSeconds", 0)
        try:
            date_obj = datetime.fromtimestamp(date_ts)
            contests_data.append({
                "date": date_obj,
                "rating": contest.get("newRating"),
                "old_rating": contest.get("oldRating")
            })
            contest_times.add(date_ts)
        except:
            pass
    
    for sub in unique_submissions:
        if sub.get("verdict") == "OK":
            date_ts = sub.get("creationTimeSeconds", 0)
            prob = sub.get("problem", {})
            
            try:
                date_obj = datetime.fromtimestamp(date_ts)
                problem_rating = prob.get("rating", 1000)
                tags = prob.get("tags", ["Unknown"])
                primary_tag = tags[0].title() if tags else "Unknown"
                
                is_contest_problem = _is_contest_submission(date_ts, contest_times)
                
                practice_data.append({
                    "date": date_obj,
                    "problem_rating": problem_rating,
                    "tag": primary_tag,
                    "contest_id": prob.get("contestId"),
                    "problem_index": prob.get("index"),
                    "is_contest": is_contest_problem
                })
            except:
                pass
    
    practice_cols = ["date", "problem_rating", "tag", "contest_id", "problem_index", "is_contest"]
    contest_cols  = ["date", "rating", "old_rating"]

    df_practice = (
        pd.DataFrame(practice_data, columns=practice_cols).sort_values("date").reset_index(drop=True)
        if practice_data else pd.DataFrame(columns=practice_cols)
    )
    df_contests = (
        pd.DataFrame(contests_data, columns=contest_cols).sort_values("date").reset_index(drop=True)
        if contests_data else pd.DataFrame(columns=contest_cols)
    )
    
    return df_practice, df_contests


def _is_contest_submission(problem_timestamp, contest_times, window_seconds=14400):
    """
    Determine if a problem was solved during a contest.
    Assumes problems submitted within 4 hours (14400s) before a contest update were part of that contest.
    
    Args:
        problem_timestamp (int): Unix timestamp of problem submission
        contest_times (set): Set of contest end times (ratingUpdateTimeSeconds)
        window_seconds (int): Time window before contest (default 4 hours)
    
    Returns:
        bool: True if submission was likely during a contest
    """
    for contest_time in contest_times:
        if contest_time - window_seconds <= problem_timestamp <= contest_time:
            return True
    return False


def calculate_statistics(df_practice, df_contests):
    """
    Calculate user statistics and performance metrics.
    
    Args:
        df_practice (DataFrame): Processed practice problems data
        df_contests (DataFrame): Processed contest ratings data
        
    Returns:
        dict: Statistics dictionary with ML-friendly metrics
    """
    stats = {}
    
    if not df_practice.empty and not df_contests.empty:
        merged = pd.merge_asof(
            df_practice,
            df_contests,
            on="date",
            direction="backward",
            suffixes=("", "_contest")
        )
        merged = merged.dropna(subset=["rating"])
        merged["gap"] = merged["problem_rating"] - merged["rating"]
        merged["rolling_difficulty"] = merged["problem_rating"].rolling(window=20, min_periods=1).mean()
        
        df_practice = merged
    
    topic_stats = None
    if not df_practice.empty:
        topic_stats = df_practice.groupby("tag").agg({
            "problem_rating": ["count", "mean", "max"],
            "gap": "mean"
        }).round(2)
        
        topic_stats.columns = ["Solves", "Avg Rating", "Max Rating", "Avg Gap"]
        topic_stats = topic_stats.sort_values("Avg Rating", ascending=False)
    
    contest_count = 0
    if not df_contests.empty:
        current_rating = float(df_contests.iloc[-1]["rating"])
        peak_rating = float(df_contests["rating"].max())
        contest_count = len(df_contests)
        
        stats["current_rating"] = current_rating
        stats["peak_rating"] = peak_rating
        stats["contest_count"] = contest_count
        stats["gap_to_peak"] = peak_rating - current_rating
        
        if contest_count >= 2:
            rating_changes = df_contests["rating"].diff().dropna()
            stats["avg_rating_change"] = float(rating_changes.mean())
            stats["rating_volatility"] = float(rating_changes.std())
            stats["rating_trend"] = float(rating_changes.iloc[-5:].mean()) if len(rating_changes) >= 5 else float(rating_changes.mean())
    else:
        stats["current_rating"] = None
        stats["peak_rating"] = None
        stats["contest_count"] = 0
    
    if not df_practice.empty:
        stats["total_problems"] = len(df_practice)
        stats["avg_problem_rating"] = float(df_practice["problem_rating"].mean())
        stats["max_problem_rating"] = float(df_practice["problem_rating"].max())
        stats["avg_gap"] = float(df_practice["gap"].mean()) if "gap" in df_practice.columns else None
        stats["topic_stats"] = topic_stats
        
        if "is_contest" in df_practice.columns:
            contest_problems = df_practice[df_practice["is_contest"] == True]
            practice_problems = df_practice[df_practice["is_contest"] == False]
            
            stats["contest_problems_count"] = len(contest_problems)
            stats["practice_problems_count"] = len(practice_problems)
            stats["avg_practice_problems_before_contests"] = float(
                stats["practice_problems_count"] / max(contest_count, 1)
            )
            
            if not practice_problems.empty and not contest_problems.empty:
                avg_practice_rating = float(practice_problems["problem_rating"].mean())
                avg_contest_rating = float(contest_problems["problem_rating"].mean())
                stats["avg_rating_diff_practice_vs_contest"] = avg_practice_rating - avg_contest_rating
            else:
                stats["avg_rating_diff_practice_vs_contest"] = None
        else:
            stats["contest_problems_count"] = 0
            stats["practice_problems_count"] = stats["total_problems"]
            stats["avg_practice_problems_before_contests"] = 0
            stats["avg_rating_diff_practice_vs_contest"] = None
    else:
        stats["total_problems"] = 0
    
    return stats, df_practice, df_contests


def analyze_user(username):
    """
    Complete analysis pipeline: fetch data, process, and calculate statistics.
    
    Args:
        username (str): Codeforces user handle
        
    Returns:
        dict: Analysis results with keys: stats, df_practice, df_contests, success
    """
    # Fetch data
    submissions, ratings_data = fetch_user_data(username)
    
    if submissions is None:
        return {
            "success": False,
            "username": username,
            "error": "Failed to fetch data"
        }
    
    print(f"✓ Fetched {len(submissions)} submissions")
    print(f"✓ Fetched {len(ratings_data)} contest participations")
    
    # Process data
    df_practice, df_contests = process_submissions(submissions, ratings_data)
    print(f"✓ Processed {len(df_practice)} solved problems")
    print(f"✓ Processed {len(df_contests)} contests")
    
    stats, df_practice, df_contests = calculate_statistics(df_practice, df_contests)
    
    return {
        "success": True,
        "username": username,
        "stats": stats,
        "df_practice": df_practice,
        "df_contests": df_contests
    }


def extract_ml_features(username, stats, df_practice, df_contests):
    """
    Extract ML-ready features for rating prediction and topic recommendation.
    
    Returns a dict with user-level features and topic-level features.
    """
    features = {
        "username": username,
        "current_rating": stats.get("current_rating"),
        "peak_rating": stats.get("peak_rating"),
        "gap_to_peak": stats.get("gap_to_peak"),
        "total_problems": stats.get("total_problems"),
        "avg_problem_rating": stats.get("avg_problem_rating"),
        "avg_gap": stats.get("avg_gap"),
        "contest_count": stats.get("contest_count"),
        "avg_rating_change": stats.get("avg_rating_change"),
        "rating_volatility": stats.get("rating_volatility"),
        "rating_trend": stats.get("rating_trend"),
        "contest_problems_count": stats.get("contest_problems_count"),
        "practice_problems_count": stats.get("practice_problems_count"),
        "avg_practice_problems_before_contests": stats.get("avg_practice_problems_before_contests"),
        "avg_rating_diff_practice_vs_contest": stats.get("avg_rating_diff_practice_vs_contest"),
        "topic_stats": {}
    }
    
    if stats.get("topic_stats") is not None:
        topic_df = stats["topic_stats"].reset_index()
        for _, row in topic_df.iterrows():
            topic_name = row["tag"]
            features["topic_stats"][topic_name] = {
                "problems_solved": int(row["Solves"]),
                "avg_rating": float(row["Avg Rating"]),
                "max_rating": float(row["Max Rating"]),
                "avg_gap": float(row["Avg Gap"]),
                "difficulty_gap": float(row["Avg Rating"] - features["current_rating"]) if features["current_rating"] else None
            }
    
    return features


TOPICS = [
    "Dp", "Greedy", "Graphs", "Math", "Implementation",
    "Binary Search", "Sortings", "Strings", "Trees", "Two Pointers"
]


def extract_contest_level_features(username, df_practice, df_contests):
    rows = []

    if df_contests.empty or df_practice.empty:
        return rows

    has_gap = "gap" in df_practice.columns

    for i, contest in df_contests.iterrows():
        if pd.isna(contest.get("old_rating")):
            continue

        contest_date = contest["date"]
        rating_before = float(contest["old_rating"])
        rating_after = float(contest["rating"])
        target = rating_after - rating_before

        prior = df_practice[df_practice["date"] < contest_date]
        prior_contests = df_contests[df_contests["date"] < contest_date]

        if prior.empty:
            continue

        w7  = prior[prior["date"] >= contest_date - timedelta(days=7)]
        w30 = prior[prior["date"] >= contest_date - timedelta(days=30)]
        w90 = prior[prior["date"] >= contest_date - timedelta(days=90)]

        peak_so_far = float(prior_contests["rating"].max()) if not prior_contests.empty else rating_before

        changes = prior_contests["rating"].diff().dropna()
        trend_last_3 = float(changes.iloc[-3:].mean()) if len(changes) >= 3 else (float(changes.mean()) if len(changes) > 0 else np.nan)
        trend_last_5 = float(changes.iloc[-5:].mean()) if len(changes) >= 5 else (float(changes.mean()) if len(changes) > 0 else np.nan)
        volatility   = float(changes.std()) if len(changes) >= 2 else np.nan

        contests_90d = len(prior_contests[prior_contests["date"] >= contest_date - timedelta(days=90)])
        days_since_last_contest  = int((contest_date - prior_contests["date"].max()).days) if not prior_contests.empty else np.nan
        days_since_last_practice = int((contest_date - prior["date"].max()).days) if not prior.empty else np.nan

        total_90 = max(len(w90), 1)
        frac_sub1200   = len(w90[w90["problem_rating"] < 1200]) / total_90
        frac_1200_1600 = len(w90[(w90["problem_rating"] >= 1200) & (w90["problem_rating"] < 1600)]) / total_90
        frac_1600_2000 = len(w90[(w90["problem_rating"] >= 1600) & (w90["problem_rating"] < 2000)]) / total_90
        frac_2000_plus = len(w90[w90["problem_rating"] >= 2000]) / total_90

        row = {
            "username":           username,
            "contest_index":      i,
            "rating_before":      rating_before,
            "peak_rating_so_far": peak_so_far,
            "gap_to_peak":        peak_so_far - rating_before,

            "problems_7d":  len(w7),
            "problems_30d": len(w30),
            "problems_90d": len(w90),

            "avg_rating_7d":  float(w7["problem_rating"].mean())  if not w7.empty  else np.nan,
            "avg_rating_30d": float(w30["problem_rating"].mean()) if not w30.empty else np.nan,
            "avg_rating_90d": float(w90["problem_rating"].mean()) if not w90.empty else np.nan,

            "max_rating_30d": float(w30["problem_rating"].max()) if not w30.empty else np.nan,
            "max_rating_90d": float(w90["problem_rating"].max()) if not w90.empty else np.nan,

            "avg_gap_7d":  float(w7["gap"].mean())  if (has_gap and not w7.empty)  else np.nan,
            "avg_gap_30d": float(w30["gap"].mean()) if (has_gap and not w30.empty) else np.nan,
            "avg_gap_90d": float(w90["gap"].mean()) if (has_gap and not w90.empty) else np.nan,

            "unique_topics_30d": int(w30["tag"].nunique()) if not w30.empty else 0,
            "unique_topics_90d": int(w90["tag"].nunique()) if not w90.empty else 0,

            "trend_last_3":    trend_last_3,
            "trend_last_5":    trend_last_5,
            "rating_volatility": volatility,
            "contests_90d":    contests_90d,

            "days_since_last_contest":  days_since_last_contest,
            "days_since_last_practice": days_since_last_practice,

            "frac_sub1200":   frac_sub1200,
            "frac_1200_1600": frac_1200_1600,
            "frac_1600_2000": frac_1600_2000,
            "frac_2000_plus": frac_2000_plus,

            "target": target,
        }

        for topic in TOPICS:
            t = prior[prior["tag"] == topic]
            col = topic.lower().replace(" ", "_")
            row[f"topic_{col}_count"]      = len(t)
            row[f"topic_{col}_avg_rating"] = float(t["problem_rating"].mean()) if not t.empty else np.nan

        rows.append(row)

    return rows


def generate_report(username, stats, df_practice, df_contests, show_plot=True):
    """
    Generate comprehensive intelligence report with visualizations.
    
    Args:
        username (str): Codeforces user handle
        stats (dict): Statistics from calculate_statistics
        df_practice (DataFrame): Practice data
        df_contests (DataFrame): Contest data
        show_plot (bool): Whether to display the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel 1: Rating Progression
    if not df_contests.empty:
        axes[0, 0].plot(df_contests["date"], df_contests["rating"], 
                        marker="o", linewidth=2.5, color="crimson", markersize=6)
        axes[0, 0].axhline(y=1200, color="gray", linestyle="--", alpha=0.3, label="Pupil (1200)")
        axes[0, 0].axhline(y=1400, color="cyan", linestyle="--", alpha=0.3, label="Specialist (1400)")
        axes[0, 0].set_title("Codeforces Rating Progression", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Rating")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Panel 2: Practice Difficulty Trend
    if not df_practice.empty:
        axes[0, 1].scatter(df_practice["date"], df_practice["problem_rating"], 
                           alpha=0.3, s=50, color="skyblue")
        if "rolling_difficulty" in df_practice.columns:
            axes[0, 1].plot(df_practice["date"], df_practice["rolling_difficulty"], 
                           linewidth=3, color="navy", label="20-Problem MA")
        axes[0, 1].set_title("Practice Difficulty Trend", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Problem Rating")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Panel 3: Performance Gap
    if not df_practice.empty and "gap" in df_practice.columns:
        axes[1, 0].scatter(df_practice["date"], df_practice["gap"], alpha=0.6, color="green")
        axes[1, 0].axhline(y=0, linestyle="--", color="red", linewidth=2)
        axes[1, 0].set_title("Practice Difficulty Gap", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Problem Rating - User Rating")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Topic Skill Distribution
    if not df_practice.empty:
        topic_avg = df_practice.groupby("tag")["problem_rating"].mean().sort_values(ascending=True).tail(10)
        axes[1, 1].barh(range(len(topic_avg)), topic_avg.values, color="steelblue")
        axes[1, 1].set_yticks(range(len(topic_avg)))
        axes[1, 1].set_yticklabels(topic_avg.index)
        axes[1, 1].set_title("Top 10 Topic Skills", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Average Solved Rating")
        axes[1, 1].grid(True, alpha=0.3, axis="x")
    
    plt.suptitle(f"Codeforces Intelligence Report: {username}", 
                 fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig
