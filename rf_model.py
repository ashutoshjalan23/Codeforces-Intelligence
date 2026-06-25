import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATASET   = "ml_dataset.csv"
MODEL_OUT = "models/rf_rating.pkl"
DROP_COLS = ["username", "contest_index"]
TARGET    = "target"


def load_data(path=DATASET, min_contests=10):
    df = pd.read_csv(path)
    counts = df.groupby("username")["contest_index"].count()
    valid_users = counts[counts >= min_contests].index
    df = df[df["username"].isin(valid_users)]
    users = df["username"]
    y = df[TARGET]
    X = df.drop(columns=DROP_COLS + [TARGET])
    return X, y, users


def split_by_user(X, y, users, test_size=0.2, seed=42):
    train_users, test_users = train_test_split(
        users.unique(), test_size=test_size, random_state=seed
    )
    train_mask = users.isin(train_users)
    test_mask  = users.isin(test_users)
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def train(X_train, y_train, n_estimators=100):
    # no scaler needed — trees are invariant to feature scale
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def tune_n_estimators(X_train, y_train, X_test, y_test, candidates=(100, 200, 300)):
    print("\nTuning n_estimators...")
    results = {}
    for n in candidates:
        print(f"  Training with n_estimators={n}...")
        pipeline = train(X_train, y_train, n_estimators=n)
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        results[n] = (mae, pipeline)
        print(f"    MAE: {mae:.2f}")
    best_n = min(results, key=lambda k: results[k][0])
    print(f"\n  Best: n_estimators={best_n}  MAE={results[best_n][0]:.2f}")
    return results[best_n][1], best_n


def evaluate(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))

    print(f"  MAE:          {mae:.2f} rating points")
    print(f"  RMSE:         {rmse:.2f}")
    print(f"  R²:           {r2:.4f}")
    print(f"  Baseline MAE: {baseline_mae:.2f}  (always predict 0)")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_feature_importance(pipeline, feature_names, top_n=20):
    importance = pipeline.named_steps["model"].feature_importances_
    idx = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(idx)), importance[idx], color="steelblue")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Feature Importance (mean decrease impurity)")
    plt.title(f"Top {top_n} Features — Random Forest Rating Predictor")
    plt.tight_layout()
    plt.savefig("feature_importance_rf.png", dpi=150)
    plt.show()
    print("Saved feature_importance_rf.png")


if __name__ == "__main__":
    print("Loading data...")
    X, y, users = load_data()
    print(f"  {len(X)} rows | {X.shape[1]} features | {users.nunique()} users")
    print(f"  Target — mean: {y.mean():.1f}  std: {y.std():.1f}  range: [{y.min():.0f}, {y.max():.0f}]")

    print("\nSplitting by user (80/20)...")
    X_train, X_test, y_train, y_test = split_by_user(X, y, users)
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    pipeline, best_n = tune_n_estimators(X_train, y_train, X_test, y_test)

    print(f"\nFinal evaluation (n_estimators={best_n}):")
    evaluate(pipeline, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

    plot_feature_importance(pipeline, list(X.columns))
