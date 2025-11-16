import os
import sys
import pandas as pd

# Resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

RESULT_DIR = os.path.join(PROJECT_ROOT, "metadata", "results")

BASELINE_CSV = os.path.join(
    RESULT_DIR, "hf_fairface_gender_frontish_validation.csv"
)
NORM_CSV = os.path.join(
    RESULT_DIR, "hf_fairface_gender_frontish_validation_norm.csv"
)

# FairFace convention (this matches the dataset docs):
#   gender: 0 = Male, 1 = Female
GENDER_MAP = {
    0: "male",
    1: "female",
}

# race (just for nicer printing; not needed for correctness logic)
RACE_MAP = {
    0: "East Asian",
    1: "Indian",
    2: "Black",
    3: "White",
    4: "Middle Eastern",
    5: "Latino_Hispanic",
    6: "Southeast Asian",
}


def load_results():
    print("Loading baseline:", BASELINE_CSV)
    df_base = pd.read_csv(BASELINE_CSV)

    print("Loading normalized:", NORM_CSV)
    df_norm = pd.read_csv(NORM_CSV)

    df_base["condition"] = "baseline"
    df_norm["condition"] = "normalized"

    df = pd.concat([df_base, df_norm], ignore_index=True)

    print("\nColumns in merged dataframe:")
    print(list(df.columns))

    # ----- Decode true labels -----

    # true_gender is 0/1 → map to "male"/"female"
    # (handles both int and string "0"/"1")
    df["true_gender_id"] = df["true_gender"].astype(int)
    df["gender_name"] = df["true_gender_id"].map(GENDER_MAP)

    # true_race is 0–6 → map to race names (for display)
    df["true_race_id"] = df["true_race"].astype(int)
    df["race_name"] = df["true_race_id"].map(RACE_MAP)

    # ----- Normalize predictions -----

    # api_pred is "male"/"female"
    df["pred_label_norm"] = df["api_pred"].astype(str).str.strip().str.lower()
    df["true_gender_norm"] = df["gender_name"].astype(str).str.strip().str.lower()

    # correctness as numeric 0/1
    df["correct"] = (df["pred_label_norm"] == df["true_gender_norm"]).astype("int64")

    print("\nSanity check: overall mean(correct) =", df["correct"].mean())

    return df


def compute_group_stats(df, group_cols):
    """
    group_cols = ["race_name"], ["gender_name"], ["race_name", "gender_name"]
    """
    stats = (
        df.groupby(group_cols + ["condition"])
        .agg(correct_rate=("correct", "mean"), count=("correct", "size"))
        .reset_index()
    )

    stats["correct_rate"] = stats["correct_rate"].astype(float)

    pivot = stats.pivot_table(
        index=group_cols,
        columns="condition",
        values="correct_rate",
    ).reset_index()

    # Ensure both columns exist
    if "baseline" not in pivot.columns:
        pivot["baseline"] = float("nan")
    if "normalized" not in pivot.columns:
        pivot["normalized"] = float("nan")

    pivot["baseline"] = pivot["baseline"].astype(float)
    pivot["normalized"] = pivot["normalized"].astype(float)
    pivot["delta"] = pivot["normalized"] - pivot["baseline"]

    ordered_cols = list(group_cols) + ["baseline", "normalized", "delta"]
    pivot = pivot[ordered_cols]

    return pivot


def main():
    df = load_results()

    print("\n==========================")
    print(" OVERALL ACCURACY ")
    print("==========================")
    overall = (
        df.groupby("condition")
        .agg(acc=("correct", "mean"), count=("correct", "size"))
        .reset_index()
    )
    overall["acc"] = overall["acc"].astype(float)
    print(overall)

    print("\n==========================")
    print(" BY RACE ")
    print("==========================")
    race_stats = compute_group_stats(df, ["race_name"])
    print(race_stats)

    print("\n==========================")
    print(" BY GENDER ")
    print("==========================")
    gender_stats = compute_group_stats(df, ["gender_name"])
    print(gender_stats)

    print("\n==========================")
    print(" BY RACE × GENDER ")
    print("==========================")
    joint_stats = compute_group_stats(df, ["race_name", "gender_name"])
    print(joint_stats)

    # Save CSVs
    os.makedirs(RESULT_DIR, exist_ok=True)
    overall.to_csv(os.path.join(RESULT_DIR, "overall.csv"), index=False)
    race_stats.to_csv(os.path.join(RESULT_DIR, "by_race.csv"), index=False)
    gender_stats.to_csv(os.path.join(RESULT_DIR, "by_gender.csv"), index=False)
    joint_stats.to_csv(
        os.path.join(RESULT_DIR, "by_race_gender.csv"), index=False
    )

    print(f"\nSaved summary CSVs to {RESULT_DIR}")


if __name__ == "__main__":
    main()