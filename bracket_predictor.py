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
MODELS_DIR = "models/bracket"
DROP_COLS = ["username", "contest_index"]
TARGET    = "target"

BRACKETS = [
    ("<1200",     0,    1200),
    ("1200-1600", 1200, 1600),
    ("1600-1900", 1600, 1900),
    ("1900-2100", 1900, 2100),
    ("2100-2400", 2100, 2400),
    ("2400+",     2400, 9999),
]


def load_data(path=DATASET, min_contests=10):
    df = pd.read_csv(path)
    counts = df.groupby("username")["contest_index"].count()
    valid_users = counts[counts >= min_contests].index
    df = df[df["username"].isin(valid_users)]
    return df


def assign_bracket(df):
    labels = [b[0] for b in BRACKETS]
    bins   = [b[1] for b in BRACKETS] + [BRACKETS[-1][2]]
    df = df.copy()
    df["bracket"] = pd.cut(df["rating_before"], bins=bins, labels=labels, right=False)
    return df


def split_by_user(df, test_size=0.2, seed=42):
    users = df["username"].unique()
    train_users, test_users = train_test_split(users, test_size=test_size, random_state=seed)
    train_mask = df["username"].isin(train_users)
    test_mask  = df["username"].isin(test_users)
    return df[train_mask], df[test_mask]


def get_xy(df):
    users = df["username"]
    y = df[TARGET]
    X = df.drop(columns=DROP_COLS + [TARGET, "bracket"])
    return X, y, users


def train_rf(X_train, y_train, n_estimators=500):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_bracket_mae(preds, y_test, brackets):
    print(f"\n  {'Bracket':<14} {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  {'Rows':>6}  {'Baseline MAE':>13}")
    print(f"  {'-'*65}")
    summary = {}
    for label in [b[0] for b in BRACKETS]:
        mask = brackets == label
        if mask.sum() < 10:
            continue
        mae  = mean_absolute_error(y_test[mask], preds[mask])
        rmse = np.sqrt(mean_squared_error(y_test[mask], preds[mask]))
        r2   = r2_score(y_test[mask], preds[mask])
        base = mean_absolute_error(y_test[mask], np.zeros(mask.sum()))
        print(f"  {label:<14} {mae:>8.2f}  {rmse:>8.2f}  {r2:>8.4f}  {mask.sum():>6}  {base:>13.2f}")
        summary[label] = {"mae": mae, "rmse": rmse, "r2": r2, "n": int(mask.sum())}
    return summary


# --- Global RF model per-bracket breakdown ---

def evaluate_global_model(df_test):
    global_model_path = "models/rf_rating.pkl"
    if not os.path.exists(global_model_path):
        print("Global RF model not found — run rf_predictor.py first.")
        return

    print("\n=== Global RF model — per-bracket MAE ===")
    pipeline = joblib.load(global_model_path)
    X_test, y_test, _ = get_xy(df_test)
    preds = pipeline.predict(X_test)
    evaluate_bracket_mae(preds, y_test.values, df_test["bracket"].values)


# --- Bracket-specific models ---

def train_bracket_models(df_train, df_test):
    os.makedirs(MODELS_DIR, exist_ok=True)
    bracket_models = {}
    all_preds  = np.full(len(df_test), np.nan)
    all_y      = df_test[TARGET].values
    all_brackets = df_test["bracket"].values

    for label, low, high in BRACKETS:
        train_b = df_train[df_train["bracket"] == label]
        test_b  = df_test[df_test["bracket"] == label]

        if train_b["username"].nunique() < 20:
            print(f"  [{label}] skipping — only {train_b['username'].nunique()} users in train")
            continue

        X_tr, y_tr, _ = get_xy(train_b)
        X_te, y_te, _ = get_xy(test_b)

        print(f"  [{label}] training on {len(X_tr)} rows ({train_b['username'].nunique()} users)...")
        model = train_rf(X_tr, y_tr)
        mae = mean_absolute_error(y_te, model.predict(X_te))
        print(f"    MAE: {mae:.2f}")

        test_idx = df_test[df_test["bracket"] == label].index
        all_preds[df_test.index.get_indexer(test_idx)] = model.predict(X_te)

        bracket_models[label] = model
        joblib.dump(model, f"{MODELS_DIR}/rf_{label.replace(' ', '_').replace('+','plus').replace('<','lt')}.pkl")

    return bracket_models, all_preds, all_y, all_brackets


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    df = assign_bracket(df)
    print(f"  {len(df)} rows | {df['username'].nunique()} users")

    print("\nBracket distribution:")
    counts = df["bracket"].value_counts().sort_index()
    for label, cnt in counts.items():
        print(f"  {label:<14} {cnt:>7} rows  ({cnt/len(df)*100:.1f}%)")

    print("\nSplitting by user (80/20)...")
    df_train, df_test = split_by_user(df)
    print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")

    # per-bracket breakdown of the global RF model (if it exists)
    evaluate_global_model(df_test)

    # train one RF per bracket
    print("\n=== Training bracket-specific RF models ===")
    bracket_models, all_preds, all_y, all_brackets = train_bracket_models(df_train, df_test)

    # only evaluate rows where we have a bracket-specific prediction
    valid = ~np.isnan(all_preds)
    print("\n=== Bracket-specific models — per-bracket MAE ===")
    evaluate_bracket_mae(all_preds[valid], all_y[valid], all_brackets[valid])

    overall_mae = mean_absolute_error(all_y[valid], all_preds[valid])
    print(f"\n  Overall MAE (bracket models): {overall_mae:.2f}")

    # feature importance for each bracket model
    sample_bracket = list(bracket_models.keys())[0]
    feature_names = list(get_xy(df_train[df_train["bracket"] == sample_bracket])[0].columns)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for ax, (label, model) in zip(axes, bracket_models.items()):
        importance = model.named_steps["model"].feature_importances_
        idx = np.argsort(importance)[-10:]
        ax.barh(range(len(idx)), importance[idx], color="steelblue")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        ax.set_title(f"Bracket: {label}", fontsize=10)
    for ax in axes[len(bracket_models):]:
        ax.set_visible(False)
    plt.suptitle("Top 10 Features per Rating Bracket — RF", fontsize=13)
    plt.tight_layout()
    plt.savefig("feature_importance_brackets.png", dpi=150)
    plt.show()
    print("\nSaved feature_importance_brackets.png")
