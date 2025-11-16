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

    # We know your schema:
    #   true_gender, true_race, api_pred
    # Make nice readable columns
    df["race_name"] = df["true_race"].astype(str).str.strip()
    df["gender_name"] = df["true_gender"].astype(str).str.strip()

    # Normalize prediction + gender to lowercase for comparison
    df["pred_label_norm"] = df["api_pred"].astype(str).str.strip().str.lower()
    df["true_gender_norm"] = df["gender_name"].astype(str).str.strip().str.lower()

    # Correctness flag
    df["correct"] = df["pred_label_norm"] == df["true_gender_norm"]

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

    pivot = stats.pivot_table(
        index=group_cols,
        columns="condition",
        values="correct_rate",
    ).reset_index()

    if "baseline" not in pivot.columns:
        pivot["baseline"] = None
    if "normalized" not in pivot.columns:
        pivot["normalized"] = None

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
    joint_stats.to_csv(os.path.join(RESULT_DIR, "by_race_gender.csv"), index=False)

    print(f"\nSaved summary CSVs to {RESULT_DIR}")


if __name__ == "__main__":
    main()