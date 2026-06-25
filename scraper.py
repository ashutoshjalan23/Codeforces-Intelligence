from intelligence import analyze_user, extract_contest_level_features
import os
import random
import requests
import pandas as pd


DIVISIONS = [
    ("Newbie",               0,    1199),
    ("Pupil",             1200,    1399),
    ("Specialist",        1400,    1599),
    ("Expert",            1600,    1899),
    ("Candidate Master",  1900,    2099),
    ("Master",            2100,    2299),
    ("International Master", 2300, 2399),
    ("Grandmaster+",      2400,    float("inf")),
]


def sample_usernames_by_division(per_division=1000, output_file="usernames.txt"):
    print("Fetching rated user list from Codeforces...")
    resp = requests.get(
        "https://codeforces.com/api/user.ratedList?activeOnly=false",
        timeout=60,
    ).json()

    if resp["status"] != "OK":
        raise Exception(f"API error: {resp.get('comment')}")

    all_users = resp["result"]
    print(f"Total rated users fetched: {len(all_users)}")

    buckets = {name: [] for name, _, _ in DIVISIONS}
    for user in all_users:
        rating = user.get("rating", 0)
        for name, low, high in DIVISIONS:
            if low <= rating <= high:
                buckets[name].append(user["handle"])
                break

    already_processed = set()
    if os.path.exists("ml_dataset.csv"):
        try:
            already_processed = set(
                pd.read_csv("ml_dataset.csv", usecols=["username"])["username"].unique()
            )
            print(f"Found {len(already_processed)} already-processed users — they will be kept at the top")
        except Exception:
            pass

    selected = list(already_processed)
    for name, _, _ in DIVISIONS:
        pool = [u for u in buckets[name] if u not in already_processed]
        sample = random.sample(pool, min(per_division, len(pool)))
        selected.extend(sample)
        print(f"  {name:25s}: {len(sample):4d} sampled  (pool: {len(pool)})")

    with open(output_file, "w") as f:
        f.write("\n".join(selected))

    print(f"\n✓ {len(selected)} usernames written to {output_file}")
    return selected


def build_ml_dataset(usernames, output_file="ml_dataset.csv"):
    processed = set()
    total_rows = 0

    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file, usecols=["username"])
            processed = set(existing_df["username"].unique())
            total_rows = len(existing_df)
            print(f"Resuming: {len(processed)} users already processed ({total_rows} rows)")
        except Exception:
            pass

    remaining = [u for u in usernames if u not in processed]
    print(f"{len(remaining)} users left to process\n")

    for i, username in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] Processing {username}...")
        result = analyze_user(username)

        if not result["success"]:
            print(f"  ✗ Failed: {result.get('error')}")
            continue

        rows = extract_contest_level_features(
            username,
            result["df_practice"],
            result["df_contests"],
        )

        if not rows:
            print(f"  ✗ No rows for {username} (not enough data)")
            continue

        df_user = pd.DataFrame(rows)
        write_header = not os.path.exists(output_file)
        df_user.to_csv(output_file, mode="a", header=write_header, index=False)

        total_rows += len(rows)
        processed.add(username)
        print(f"  ✓ {len(rows)} rows written — {total_rows} total so far")

    print(f"\n✓ Done. Dataset: {output_file} ({total_rows} rows, {len(processed)} users)")
    return output_file


if __name__ == "__main__":
    if not os.path.exists("usernames.txt"):
        print("Sampling usernames by division...")
        usernames = sample_usernames_by_division(per_division=1000)
    else:
        with open("usernames.txt", "r") as f:
            usernames = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(usernames)} usernames from usernames.txt")

    build_ml_dataset(usernames)
