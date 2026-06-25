import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET   = "ml_dataset.csv"
MODEL_OUT = "models/xgboost_rating.json"
DROP_COLS = ["username", "contest_index"]
TARGET    = "target"


def load_data(path=DATASET):
    df = pd.read_csv(path)
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


def train(X_train, y_train):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))

    print(f"  MAE:          {mae:.2f} rating points")
    print(f"  RMSE:         {rmse:.2f}")
    print(f"  R²:           {r2:.4f}")
    print(f"  Baseline MAE: {baseline_mae:.2f}  (always predict 0)")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_feature_importance(model, feature_names, top_n=20):
    importance = model.feature_importances_
    idx = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(idx)), importance[idx], color="steelblue")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Feature Importance (gain)")
    plt.title(f"Top {top_n} Features — XGBoost Rating Predictor")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("✓ Saved feature_importance.png")


if __name__ == "__main__":
    print("Loading data...")
    X, y, users = load_data()
    print(f"  {len(X)} rows | {X.shape[1]} features | {users.nunique()} users")
    print(f"  Target — mean: {y.mean():.1f}  std: {y.std():.1f}  range: [{y.min():.0f}, {y.max():.0f}]")

    print("\nSplitting by user (80/20)...")
    X_train, X_test, y_train, y_test = split_by_user(X, y, users)
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("\nTraining XGBoost...")
    model = train(X_train, y_train)
    print(f"  Best iteration: {model.best_iteration}")

    print("\nTest set evaluation:")
    evaluate(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_OUT)
    print(f"\n✓ Model saved to {MODEL_OUT}")

    plot_feature_importance(model, list(X.columns))
