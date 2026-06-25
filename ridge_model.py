import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATASET   = "ml_dataset.csv"
MODEL_OUT = "models/ridge_rating.pkl"
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


def train(X_train, y_train, alpha=1.0):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=alpha)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


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


def plot_coefficients(pipeline, feature_names, top_n=20):
    coefs = pipeline.named_steps["model"].coef_
    idx = np.argsort(np.abs(coefs))[-top_n:]
    values = coefs[idx]
    colors = ["steelblue" if v >= 0 else "tomato" for v in values]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(idx)), values, color=colors)
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient (scaled)")
    plt.title(f"Top {top_n} Features by Magnitude — Ridge Regression Rating Predictor")
    plt.tight_layout()
    plt.savefig("feature_importance_ridge.png", dpi=150)
    plt.show()
    print("Saved feature_importance_ridge.png")


if __name__ == "__main__":
    print("Loading data...")
    X, y, users = load_data()
    print(f"  {len(X)} rows | {X.shape[1]} features | {users.nunique()} users")
    print(f"  Target — mean: {y.mean():.1f}  std: {y.std():.1f}  range: [{y.min():.0f}, {y.max():.0f}]")

    print("\nSplitting by user (80/20)...")
    X_train, X_test, y_train, y_test = split_by_user(X, y, users)
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("\nTraining Ridge Regression (alpha=1.0)...")
    pipeline = train(X_train, y_train, alpha=1.0)

    print("\nTest set evaluation:")
    evaluate(pipeline, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

    plot_coefficients(pipeline, list(X.columns))
