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

# FairFace convention: 0 = male, 1 = female
GENDER_MAP = {
    0: "male",
    1: "female",
}


# --------------------
# Helper functions
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

    # labels: FairFace uses 0 = male, 1 = female
    df["true_gender"] = df["true_gender"].astype(int)
    df["true_race"] = df["true_race"].astype(int)

    # decode predictions
    df["pred_int"] = df["api_pred"].apply(pred_to_int)

    # pretty names
    df["race_name"] = df["true_race"].map(RACE_MAP)
    df["gender_name"] = df["true_gender"].map(GENDER_MAP)

    # correctness will be (re)computed after we optionally flip normft preds
    return df


# --------------------
# Aggregation
# --------------------
def summarize(df, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        acc = sub["correct"].mean()
        lo, hi = binomial_ci(acc, n)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update(
            {
                "n": n,
                "acc": acc,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)

def compute_worst_group_accuracy(by_race_gender_df):
    """
    Computes Worst-Group Accuracy (WGA) for each condition.
    by_race_gender_df must include:
        condition, race_name, gender_name, acc
    """
    wga_rows = []
    for cond, sub in by_race_gender_df.groupby("condition"):
        # find min accuracy across all race × gender groups
        min_row = sub.loc[sub["acc"].idxmin()]
        wga_rows.append({
            "condition": cond,
            "worst_group_race": min_row["race_name"],
            "worst_group_gender": min_row["gender_name"],
            "worst_group_acc": min_row["acc"],
            "worst_group_n": min_row["n"]
        })
    return pd.DataFrame(wga_rows).sort_values("condition")
# --------------------
# Plotting
# --------------------
def plot_overall(overall_df, out_path):
    """
    Bar chart of overall accuracy per condition with 95% CI error bars.
    Expects columns: condition, acc, ci_low, ci_high
    """
    plt.figure(figsize=(6, 4))

    conds = overall_df["condition"].tolist()
    accs = overall_df["acc"].values
    ci_low = overall_df["ci_low"].values
    ci_high = overall_df["ci_high"].values

    # error bar = distance from mean to CI bound
    yerr = np.vstack([accs - ci_low, ci_high - accs])

    x = np.arange(len(conds))

    plt.bar(x, accs, yerr=yerr, capsize=4)
    plt.xticks(x, conds, rotation=45)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy by Condition (95% CI)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)


def plot_group(df, category, out_file):
    """
    Grouped bar chart with 95% CI error bars.

    df: has columns [condition, <category>, acc, ci_low, ci_high]
        e.g., category = "race_name" or "gender_name"
    """
    plt.figure(figsize=(10, 4))

    groups = sorted(df[category].unique().tolist())
    conds = sorted(df["condition"].unique().tolist())

    x = np.arange(len(groups))
    width = 0.8 / max(len(conds), 1)   # total bar cluster width ~= 0.8

    for j, cond in enumerate(conds):
        xs = x + (j - (len(conds)-1)/2) * width

        accs = []
        yerr_low = []
        yerr_high = []

        for g in groups:
            sub = df[(df[category] == g) & (df["condition"] == cond)]
            if len(sub) == 0:
                accs.append(np.nan)
                yerr_low.append(0.0)
                yerr_high.append(0.0)
            else:
                acc = sub["acc"].iloc[0]
                lo = sub["ci_low"].iloc[0]
                hi = sub["ci_high"].iloc[0]
                accs.append(acc)
                yerr_low.append(acc - lo)
                yerr_high.append(hi - acc)

        yerr = np.vstack([yerr_low, yerr_high])
        plt.bar(xs, accs, width, label=cond, yerr=yerr, capsize=3)

    plt.xticks(x, groups, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by {category} (95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print("Saved:", out_file)

def plot_wga(wga_df, out_file):
    plt.figure(figsize=(6,4))
    plt.bar(wga_df["condition"], wga_df["worst_group_acc"])
    plt.ylim(0,1)
    plt.ylabel("Worst-Group Accuracy")
    plt.title("WGA: Worst-Performing Race × Gender Group")
    plt.xticks(rotation=45)
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
    # File paths
    # ---------------------------------------------
    base_rgb_csv = os.path.join(
        RESULTS_DIR, "hf_fairface_gender_frontish_validation.csv"
    )
    base_norm_csv = os.path.join(
        RESULTS_DIR, "hf_fairface_gender_frontish_validation_norm.csv"
    )
    normft_rgb_csv = os.path.join(
        RESULTS_DIR, "ff_norm_ft_full_eval_on_rgb.csv"
    )
    normft_norm_csv = os.path.join(
        RESULTS_DIR, "ff_norm_ft_full_eval_on_norm.csv"
    )

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

    # ---------------------------------------------
    # FIX: flip fine-tuned predictions (normft_*)
    # ---------------------------------------------
    mask_normft = df["condition"].isin(["normft_rgb", "normft_norm"])
    # For binary 0/1 labels: flipped_pred = 1 - pred
    df.loc[mask_normft, "pred_int"] = 1 - df.loc[mask_normft, "pred_int"]

    # Now compute correctness with aligned labels
    df["correct"] = df["pred_int"] == df["true_gender"]

    print(df["condition"].value_counts())

    # ---------------------------------------------
    # Overall accuracy
    # ---------------------------------------------
    overall = summarize(df, ["condition"])
    print("\n=== OVERALL ===")
    print(overall)
    overall.to_csv(
        os.path.join(args.out_dir, "exp1_full_overall.csv"), index=False
    )

    # ---------------------------------------------
    # By race
    # ---------------------------------------------
    by_race = summarize(df, ["condition", "race_name"])
    by_race.to_csv(
        os.path.join(args.out_dir, "exp1_full_by_race.csv"), index=False
    )

    # ---------------------------------------------
    # By gender
    # ---------------------------------------------
    by_gender = summarize(df, ["condition", "gender_name"])
    by_gender.to_csv(
        os.path.join(args.out_dir, "exp1_full_by_gender.csv"), index=False
    )

    # ---------------------------------------------
    # By race × gender
    # ---------------------------------------------
    by_race_gender = summarize(df, ["condition", "race_name", "gender_name"])
    by_race_gender.to_csv(
        os.path.join(args.out_dir, "exp1_full_by_race_gender.csv"),
        index=False,
    )

    # ---------------------------------------------
    # Worst-Group Accuracy (WGA)
    # ---------------------------------------------
    wga_df = compute_worst_group_accuracy(by_race_gender)
    wga_df.to_csv(os.path.join(args.out_dir, "exp1_full_worst_group_accuracy.csv"), index=False)

    print("\n=== WORST-GROUP ACCURACY (WGA) ===")
    print(wga_df)

    # ---------------------------------------------
    # Plots
    # ---------------------------------------------
    plot_overall(
        overall,
        os.path.join(args.out_dir, "exp1_full_plot_overall.png"),
    )

    plot_group(
        by_race,
        "race_name",
        os.path.join(args.out_dir, "exp1_full_plot_race.png"),
    )

    plot_group(
        by_gender,
        "gender_name",
        os.path.join(args.out_dir, "exp1_full_plot_gender.png"),
    )

    plot_wga(wga_df, os.path.join(args.out_dir, "exp1_full_plot_wga.png"))

    print("\nAll done! Full Experiment 1A + 1B analysis generated.")


if __name__ == "__main__":
    main()