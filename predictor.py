import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from intelligence import analyze_user, TOPICS

MODEL_PATH = "models/rf_rating.pkl"
DATASET    = "ml_dataset.csv"

FEATURE_COLS = [
    "rating_before", "peak_rating_so_far", "gap_to_peak",
    "problems_7d", "problems_30d", "problems_90d",
    "avg_rating_7d", "avg_rating_30d", "avg_rating_90d",
    "max_rating_30d", "max_rating_90d",
    "avg_gap_7d", "avg_gap_30d", "avg_gap_90d",
    "unique_topics_30d", "unique_topics_90d",
    "trend_last_3", "trend_last_5", "rating_volatility", "contests_90d",
    "days_since_last_contest", "days_since_last_practice",
    "frac_sub1200", "frac_1200_1600", "frac_1600_2000", "frac_2000_plus",
    "topic_dp_count", "topic_dp_avg_rating",
    "topic_greedy_count", "topic_greedy_avg_rating",
    "topic_graphs_count", "topic_graphs_avg_rating",
    "topic_math_count", "topic_math_avg_rating",
    "topic_implementation_count", "topic_implementation_avg_rating",
    "topic_binary_search_count", "topic_binary_search_avg_rating",
    "topic_sortings_count", "topic_sortings_avg_rating",
    "topic_strings_count", "topic_strings_avg_rating",
    "topic_trees_count", "topic_trees_avg_rating",
    "topic_two_pointers_count", "topic_two_pointers_avg_rating",
]


def build_current_features(df_practice, df_contests):
    if df_contests.empty:
        return None

    now          = datetime.now()
    rating       = float(df_contests.iloc[-1]["rating"])
    peak         = float(df_contests["rating"].max())
    has_gap      = "gap" in df_practice.columns

    w7  = df_practice[df_practice["date"] >= now - timedelta(days=7)]
    w30 = df_practice[df_practice["date"] >= now - timedelta(days=30)]
    w90 = df_practice[df_practice["date"] >= now - timedelta(days=90)]

    changes      = df_contests["rating"].diff().dropna()
    trend_last_3 = float(changes.iloc[-3:].mean()) if len(changes) >= 3 else (float(changes.mean()) if len(changes) > 0 else np.nan)
    trend_last_5 = float(changes.iloc[-5:].mean()) if len(changes) >= 5 else (float(changes.mean()) if len(changes) > 0 else np.nan)
    volatility   = float(changes.std()) if len(changes) >= 2 else np.nan
    total_90     = max(len(w90), 1)

    features = {
        "rating_before":            rating,
        "peak_rating_so_far":       peak,
        "gap_to_peak":              peak - rating,

        "problems_7d":              len(w7),
        "problems_30d":             len(w30),
        "problems_90d":             len(w90),

        "avg_rating_7d":            float(w7["problem_rating"].mean())  if not w7.empty  else np.nan,
        "avg_rating_30d":           float(w30["problem_rating"].mean()) if not w30.empty else np.nan,
        "avg_rating_90d":           float(w90["problem_rating"].mean()) if not w90.empty else np.nan,

        "max_rating_30d":           float(w30["problem_rating"].max()) if not w30.empty else np.nan,
        "max_rating_90d":           float(w90["problem_rating"].max()) if not w90.empty else np.nan,

        "avg_gap_7d":               float(w7["gap"].mean())  if (has_gap and not w7.empty)  else np.nan,
        "avg_gap_30d":              float(w30["gap"].mean()) if (has_gap and not w30.empty) else np.nan,
        "avg_gap_90d":              float(w90["gap"].mean()) if (has_gap and not w90.empty) else np.nan,

        "unique_topics_30d":        int(w30["tag"].nunique()) if not w30.empty else 0,
        "unique_topics_90d":        int(w90["tag"].nunique()) if not w90.empty else 0,

        "trend_last_3":             trend_last_3,
        "trend_last_5":             trend_last_5,
        "rating_volatility":        volatility,
        "contests_90d":             len(df_contests[df_contests["date"] >= now - timedelta(days=90)]),

        "days_since_last_contest":  int((now - df_contests["date"].max()).days),
        "days_since_last_practice": int((now - df_practice["date"].max()).days) if not df_practice.empty else np.nan,

        "frac_sub1200":             len(w90[w90["problem_rating"] < 1200]) / total_90,
        "frac_1200_1600":           len(w90[(w90["problem_rating"] >= 1200) & (w90["problem_rating"] < 1600)]) / total_90,
        "frac_1600_2000":           len(w90[(w90["problem_rating"] >= 1600) & (w90["problem_rating"] < 2000)]) / total_90,
        "frac_2000_plus":           len(w90[w90["problem_rating"] >= 2000]) / total_90,
    }

    for topic in TOPICS:
        col = topic.lower().replace(" ", "_")
        t   = df_practice[df_practice["tag"] == topic]
        features[f"topic_{col}_count"]      = len(t)
        features[f"topic_{col}_avg_rating"] = float(t["problem_rating"].mean()) if not t.empty else np.nan

    return features


def get_topic_insights(features, rating):
    weak, strong, untouched = [], [], []
    for topic in TOPICS:
        col   = topic.lower().replace(" ", "_")
        count = features.get(f"topic_{col}_count", 0)
        avg_r = features.get(f"topic_{col}_avg_rating", np.nan)

        if count == 0:
            untouched.append(topic)
        elif not np.isnan(avg_r):
            gap = avg_r - rating
            if gap < -150:
                weak.append((topic, int(avg_r), int(gap)))
            elif gap >= 0:
                strong.append((topic, int(avg_r), int(gap)))

    weak.sort(key=lambda x: x[2])
    strong.sort(key=lambda x: -x[2])
    return weak, strong, untouched


def get_peer_comparison(features, rating):
    peer_metrics = ["rating_before", "avg_gap_30d", "problems_30d", "unique_topics_30d", "max_rating_30d"]
    try:
        df = pd.read_csv(DATASET, usecols=peer_metrics)
    except FileNotFoundError:
        return None

    for band in [100, 200, 300]:
        peers = df[(df["rating_before"] >= rating - band) & (df["rating_before"] <= rating + band)]
        if len(peers) >= 50:
            break
    else:
        return None

    result = {}
    labels = {
        "avg_gap_30d":       "Practice difficulty gap",
        "problems_30d":      "Problems solved / month",
        "unique_topics_30d": "Topic diversity / month",
        "max_rating_30d":    "Hardest problem (30d)",
    }
    for metric in labels:
        peer_vals = peers[metric].dropna()
        user_val  = features.get(metric)
        if user_val is None or np.isnan(user_val) or peer_vals.empty:
            continue
        result[metric] = {
            "label":      labels[metric],
            "user":       round(user_val, 1),
            "peer_mean":  round(float(peer_vals.mean()), 1),
            "peer_median":round(float(peer_vals.median()), 1),
            "percentile": round((peer_vals < user_val).mean() * 100, 1),
        }

    return result, len(peers), band


def print_report(username, features, predicted_delta, topic_insights, peer_data):
    rating        = features["rating_before"]
    weak, strong, untouched = topic_insights
    sign          = "+" if predicted_delta >= 0 else ""

    print("\n" + "=" * 56)
    print(f"  Codeforces Intelligence  —  {username}")
    print("=" * 56)

    print(f"\n  Current Rating  :  {int(rating)}")
    print(f"  Peak Rating     :  {int(features['peak_rating_so_far'])}")
    print(f"\n  Predicted Delta :  {sign}{predicted_delta:.0f}")
    print(f"  Predicted Next  :  {int(rating + predicted_delta)}")

    print("\n── Practice (last 30 days) ──────────────────────────")
    print(f"  Problems solved  :  {int(features['problems_30d'])}")
    avg_r = features.get("avg_rating_30d")
    avg_g = features.get("avg_gap_30d")
    if avg_r and not np.isnan(avg_r):
        gap_str = f"  ({avg_g:+.0f} vs your rating)" if avg_g and not np.isnan(avg_g) else ""
        print(f"  Avg difficulty   :  {avg_r:.0f}{gap_str}")
    print(f"  Topic diversity  :  {int(features['unique_topics_30d'])} topics")

    print("\n── Topics ───────────────────────────────────────────")
    if strong:
        print("  Strong:")
        for t, r, g in strong:
            print(f"    {t:<22}  avg {r}  ({g:+d})")
    if weak:
        print("  Needs work (solving well below your rating):")
        for t, r, g in weak:
            print(f"    {t:<22}  avg {r}  ({g:+d})")
    if untouched:
        print(f"  Never practiced:  {', '.join(untouched)}")

    if peer_data:
        comparison, n_peers, band = peer_data
        print(f"\n── vs. Similar Users (±{band} rating, n={n_peers}) ─────────")
        for data in comparison.values():
            pct  = data["percentile"]
            fill = int(pct / 5)
            bar  = "▓" * fill + "░" * (20 - fill)
            print(f"  {data['label']:<28} [{bar}] {pct:.0f}th pct")
            print(f"    You: {data['user']}  |  Peer avg: {data['peer_mean']}  |  Peer median: {data['peer_median']}")

    print("\n" + "=" * 56)


def analyze(username):
    print(f"Fetching data for {username}...")
    result = analyze_user(username)

    if not result["success"]:
        print(f"Failed: {result.get('error')}")
        return

    features = build_current_features(result["df_practice"], result["df_contests"])
    if features is None:
        print("Not enough contest history.")
        return

    model = joblib.load(MODEL_PATH)

    X               = pd.DataFrame([features])[FEATURE_COLS]
    predicted_delta = float(model.predict(X)[0])
    topic_insights  = get_topic_insights(features, features["rating_before"])
    peer_data       = get_peer_comparison(features, features["rating_before"])

    print_report(username, features, predicted_delta, topic_insights, peer_data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python insights.py <codeforces_username>")
        sys.exit(1)
    analyze(sys.argv[1])
