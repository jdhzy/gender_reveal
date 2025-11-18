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

# Convention we’re using everywhere now: 0 = female, 1 = male
GENDER_MAP = {
    0: "male",
    1: "female",
}


# --------------------
# Helpers
# --------------------
def pred_to_int(x: str) -> int:
    """
    Map string predictions to integer labels consistent with FairFace:
      0 = male, 1 = female
    """
    x = str(x).strip().lower()
    if x in ["male", "man", "0"]:
        return 0
    elif x in ["female", "woman", "1"]:
        return 1
    else:
        return -1


def binomial_ci(p: float, n: int, z: float = 1.96):
    """
    95% CI for a binomial proportion using normal approximation.
    """
    if n == 0:
        return (np.nan, np.nan)
    se = np.sqrt(p * (1.0 - p) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def load_condition(csv_path: str, condition_name: str) -> pd.DataFrame:
    """
    Load one result CSV and add:
      - condition
      - pred_int
      - race_name, gender_name
      - correct (bool)
    """
    df = pd.read_csv(csv_path)

    required = ["filename", "true_gender", "true_race", "api_pred"]
    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df["condition"] = condition_name

    df["true_gender"] = df["true_gender"].astype(int)
    df["true_race"] = df["true_race"].astype(int)

    df["pred_int"] = df["api_pred"].apply(pred_to_int)

    df["race_name"] = df["true_race"].map(RACE_MAP)
    df["gender_name"] = df["true_gender"].map(GENDER_MAP)

    df["correct"] = df["pred_int"] == df["true_gender"]
    return df


# --------------------
# Aggregation
# --------------------
def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    For groups defined by group_cols:
      - n
      - acc
      - ci_low, ci_high (95% CI)
    """
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        acc = sub["correct"].mean()
        ci_low, ci_high = binomial_ci(acc, n)

        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update(
            {
                "n": n,
                "acc": acc,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    return out


def compute_worst_group(by_race_gender: pd.DataFrame) -> pd.DataFrame:
    """
    Given baseline summary by [condition, race_name, gender_name],
    compute worst-group accuracy for each condition:

      For each condition:
        - pick the subgroup (race, gender) with the minimum acc.
    """
    rows = []
    for cond, sub in by_race_gender.groupby("condition"):
        # row with smallest accuracy
        worst = sub.loc[sub["acc"].idxmin()]
        rows.append(
            {
                "condition": cond,
                "race_name": worst["race_name"],
                "gender_name": worst["gender_name"],
                "n": int(worst["n"]),
                "worst_group_acc": float(worst["acc"]),
                "ci_low": float(worst["ci_low"]),
                "ci_high": float(worst["ci_high"]),
            }
        )
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


# --------------------
# Plot helpers
# --------------------
def plot_overall(overall_df: pd.DataFrame, out_path: str):
    """
    Bar chart with 95% CI for overall baseline comparison.
    """
    plt.figure(figsize=(6, 4))
    conds = overall_df["condition"].tolist()
    accs = overall_df["acc"].values
    ci_low = overall_df["ci_low"].values
    ci_high = overall_df["ci_high"].values

    x = np.arange(len(conds))
    yerr = np.vstack([accs - ci_low, ci_high - accs])

    plt.bar(x, accs, yerr=yerr, capsize=4)
    plt.xticks(x, conds)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Baseline Accuracy (RGB vs Normalized)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)


def plot_group(summary_df: pd.DataFrame, category: str, out_path: str):
    """
    Grouped bar chart by category (race_name or gender_name) with CI.
    """
    plt.figure(figsize=(10, 4))

    categories = sorted(summary_df[category].unique())
    conditions = sorted(summary_df["condition"].unique())

    x = np.arange(len(categories))
    width = 0.8 / max(len(conditions), 1)

    for i, cond in enumerate(conditions):
        sub = summary_df[summary_df["condition"] == cond].set_index(category).reindex(
            categories
        )
        accs = sub["acc"].values
        ci_low = sub["ci_low"].values
        ci_high = sub["ci_high"].values
        yerr = np.vstack([accs - ci_low, ci_high - accs])

        offset = (i - (len(conditions) - 1) / 2.0) * width
        plt.bar(x + offset, accs, width, label=cond, yerr=yerr, capsize=3)

    plt.xticks(x, categories, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(f"Baseline Accuracy by {category}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)


# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_rgb_csv",
        type=str,
        default=os.path.join(
            RESULTS_DIR, "hf_fairface_gender_frontish_validation.csv"
        ),
        help="Baseline RGB evaluation CSV",
    )
    parser.add_argument(
        "--base_norm_csv",
        type=str,
        default=os.path.join(
            RESULTS_DIR, "hf_fairface_gender_frontish_validation_norm.csv"
        ),
        help="Baseline evaluation CSV with on-the-fly skin normalization",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory to save baseline-only summaries and plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ----------------------------
    # Load both baseline conditions
    # ----------------------------
    base_rgb = load_condition(args.base_rgb_csv, "base_rgb")
    base_norm = load_condition(args.base_norm_csv, "base_norm")

    df = pd.concat([base_rgb, base_norm], ignore_index=True)

    print("Counts by condition:")
    print(df["condition"].value_counts())

    # ----------------------------
    # Overall accuracy
    # ----------------------------
    overall = summarize(df, ["condition"])
    print("\n=== OVERALL ===")
    print(overall)
    overall.to_csv(os.path.join(args.out_dir, "baseline_overall.csv"), index=False)

    # ----------------------------
    # By race
    # ----------------------------
    by_race = summarize(df, ["condition", "race_name"])
    print("\n=== BY RACE ===")
    print(by_race[["condition", "race_name", "acc"]])
    by_race.to_csv(os.path.join(args.out_dir, "baseline_by_race.csv"), index=False)

    # ----------------------------
    # By gender
    # ----------------------------
    by_gender = summarize(df, ["condition", "gender_name"])
    print("\n=== BY GENDER ===")
    print(by_gender[["condition", "gender_name", "acc"]])
    by_gender.to_csv(
        os.path.join(args.out_dir, "baseline_by_gender.csv"), index=False
    )

    # ----------------------------
    # By race × gender
    # ----------------------------
    by_race_gender = summarize(df, ["condition", "race_name", "gender_name"])
    print("\n=== BY RACE × GENDER ===")
    print(by_race_gender[["condition", "race_name", "gender_name", "acc"]])
    by_race_gender.to_csv(
        os.path.join(args.out_dir, "baseline_by_race_gender.csv"), index=False
    )

    # ----------------------------
    # Worst-group accuracy
    # ----------------------------
    worst = compute_worst_group(by_race_gender)
    print("\n=== WORST-GROUP ACCURACY (by race × gender) ===")
    print(worst)
    worst.to_csv(
        os.path.join(args.out_dir, "baseline_worst_group.csv"), index=False
    )

    # ----------------------------
    # Plots (with 95% CI)
    # ----------------------------
    plot_overall(
        overall,
        os.path.join(args.out_dir, "baseline_plot_overall.png"),
    )

    plot_group(
        by_race,
        "race_name",
        os.path.join(args.out_dir, "baseline_plot_by_race.png"),
    )

    plot_group(
        by_gender,
        "gender_name",
        os.path.join(args.out_dir, "baseline_plot_by_gender.png"),
    )

    print("\nBaseline-only analysis complete. Files saved to:", args.out_dir)


if __name__ == "__main__":
    main()