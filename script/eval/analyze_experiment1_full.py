import os
import sys
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------
# Path setup
# --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "metadata", "results")

# --------------------
# Mappings
# --------------------
RACE_MAP = {
    0: "Black",
    1: "East Asian",
    2: "Indian",
    3: "Latino_Hispanic",
    4: "Middle Eastern",
    5: "Southeast Asian",
    6: "White",
}
GENDER_MAP = {0: "female", 1: "male"}


# --------------------
# Helper functions
# --------------------
def pred_to_int(x):
    x = str(x).lower()
    if x in ["0", "female", "woman"]:
        return 0
    if x in ["1", "male", "man"]:
        return 1
    return -1


def binomial_ci(p, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    se = np.sqrt(p * (1 - p) / n)
    return (p - z * se, p + z * se)


# --------------------
# Load ONE system’s CSV
# --------------------
def load_condition(csv_path, condition_name):
    df = pd.read_csv(csv_path)
    df["condition"] = condition_name
    df["pred_int"] = df["api_pred"].apply(pred_to_int)
    df["true_gender"] = df["true_gender"].astype(int)
    df["true_race"] = df["true_race"].astype(int)
    df["race_name"] = df["true_race"].map(RACE_MAP)
    df["gender_name"] = df["true_gender"].map(GENDER_MAP)
    df["correct"] = df["pred_int"] == df["true_gender"]
    return df


# --------------------
# Aggregation
# --------------------
def summarize(df, group_cols):
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        acc = sub["correct"].mean()
        lo, hi = binomial_ci(acc, n)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "n": n,
            "acc": acc,
            "ci_low": lo,
            "ci_high": hi
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


# --------------------
# Plotting
# --------------------
def plot_overall(overall_df, out_path):
    plt.figure(figsize=(6, 4))
    plt.bar(overall_df["condition"], overall_df["acc"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy by Condition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)
    
def plot_group(df, category, out_file):
    plt.figure(figsize=(10, 4))

    # Use pivot_table instead of pivot to avoid duplicate-column crash
    pivot = df.pivot_table(
        index=category,
        columns="condition",
        values="acc",
        aggfunc="mean"     # safe even if duplicates exist
    )

    pivot.plot(kind="bar", figsize=(10, 4))
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by {category}")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print("Saved:", out_file)


# --------------------
# Main
# --------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------
    # File paths (update if your filenames differ)
    # ---------------------------------------------
    base_rgb_csv = os.path.join(RESULTS_DIR, "hf_fairface_gender_frontish_validation.csv")
    base_norm_csv = os.path.join(RESULTS_DIR, "hf_fairface_gender_frontish_validation_norm.csv")
    normft_rgb_csv = os.path.join(RESULTS_DIR, "ff_norm_ft_full_eval_on_rgb.csv")
    normft_norm_csv = os.path.join(RESULTS_DIR, "ff_norm_ft_full_eval_on_norm.csv")

    # ---------------------------------------------
    # Load all 4 conditions
    # ---------------------------------------------
    df_list = [
        load_condition(base_rgb_csv, "base_rgb"),
        load_condition(base_norm_csv, "base_norm"),
        load_condition(normft_rgb_csv, "normft_rgb"),
        load_condition(normft_norm_csv, "normft_norm"),
    ]

    df = pd.concat(df_list, ignore_index=True)

    print(df["condition"].value_counts())

    # ---------------------------------------------
    # Overall accuracy
    # ---------------------------------------------
    overall = summarize(df, ["condition"])
    print("\n=== OVERALL ===")
    print(overall)
    overall.to_csv(os.path.join(args.out_dir, "exp1_full_overall.csv"), index=False)

    # ---------------------------------------------
    # By race
    # ---------------------------------------------
    by_race = summarize(df, ["condition", "race_name"])
    by_race.to_csv(os.path.join(args.out_dir, "exp1_full_by_race.csv"), index=False)

    # ---------------------------------------------
    # By gender
    # ---------------------------------------------
    by_gender = summarize(df, ["condition", "gender_name"])
    by_gender.to_csv(os.path.join(args.out_dir, "exp1_full_by_gender.csv"), index=False)

    # ---------------------------------------------
    # By race × gender
    # ---------------------------------------------
    by_race_gender = summarize(df, ["condition", "race_name", "gender_name"])
    by_race_gender.to_csv(os.path.join(args.out_dir, "exp1_full_by_race_gender.csv"), index=False)

    # ---------------------------------------------
    # Plots
    # ---------------------------------------------
    plot_overall(overall, os.path.join(args.out_dir, "exp1_full_plot_overall.png"))

    plot_group(by_race, "race_name",
               os.path.join(args.out_dir, "exp1_full_plot_race.png"))

    plot_group(by_gender, "gender_name",
               os.path.join(args.out_dir, "exp1_full_plot_gender.png"))

    print("\nAll done! Full Experiment 1A + 1B analysis generated.")


if __name__ == "__main__":
    main()