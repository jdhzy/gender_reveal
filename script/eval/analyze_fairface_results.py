import os
import sys
import argparse
from typing import Dict, Tuple, List

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
# Constants / mappings
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

GENDER_MAP = {
    0: "female",
    1: "male",
}


# --------------------
# Stats helpers
# --------------------
def binomial_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Simple normal-approx CI for a binomial proportion.
    Returns (lower, upper).
    """
    if n == 0:
        return np.nan, np.nan
    se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def mcnemar_test(
    correct_base: np.ndarray, correct_norm: np.ndarray
) -> Tuple[float, float, int, int]:
    """
    McNemar's test (chi-square with continuity correction).

    correct_base, correct_norm: boolean arrays for the SAME samples.

    Returns:
        (chi2_stat, p_value, b, c)
        where:
          b = correct in baseline, wrong in normalized
          c = wrong in baseline, correct in normalized
    """
    from math import erf, sqrt

    # b: 1,0   c: 0,1
    b = int(((correct_base == True) & (correct_norm == False)).sum())
    c = int(((correct_base == False) & (correct_norm == True)).sum())

    if b + c == 0:
        # No disagreement — test is undefined; return zeros / 1.0
        return 0.0, 1.0, b, c

    # chi-square with 1 df, continuity correction
    stat = (abs(b - c) - 1) ** 2 / float(b + c)

    # Approximate p-value using chi-square(1) ~ square of N(0,1)
    # So P(Chi2 > x) = 2 * (1 - Phi(sqrt(x)))
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    z = np.sqrt(stat)
    Phi = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    p_value = 2.0 * (1.0 - Phi)

    return stat, p_value, b, c


# --------------------
# Core loading
# --------------------
def load_results(
    baseline_csv: str, normalized_csv: str
) -> pd.DataFrame:
    """
    Load baseline + normalized CSVs and merge into a single dataframe with:
      - filename, true_gender, true_race
      - api_pred_baseline, api_pred_normalized
      - correct_baseline, correct_normalized
      - race_name, gender_name
    """
    print(f"Loading baseline:   {baseline_csv}")
    print(f"Loading normalized: {normalized_csv}")

    df_base = pd.read_csv(baseline_csv)
    df_norm = pd.read_csv(normalized_csv)

    # Make sure columns exist
    for df, name in [(df_base, "baseline"), (df_norm, "normalized")]:
        for col in ["filename", "true_gender", "true_race", "api_pred"]:
            if col not in df.columns:
                raise ValueError(f"{name} CSV missing column: {col}")

    # Merge on filename (and optionally labels to be safe)
    df = pd.merge(
        df_base[
            ["filename", "true_gender", "true_race", "api_pred"]
        ].rename(columns={"api_pred": "api_pred_baseline"}),
        df_norm[
            ["filename", "true_gender", "true_race", "api_pred"]
        ].rename(columns={"api_pred": "api_pred_normalized"}),
        on=["filename", "true_gender", "true_race"],
        suffixes=("_base", "_norm"),
        how="inner",
    )

    print("\nColumns in merged dataframe:")
    print(df.columns.tolist())

    # Normalize true_gender to {0,1} ints if needed
    # FairFace labels: 0=female, 1=male
    if df["true_gender"].dtype == "O":
        # Try to convert from strings like "0"/"1"
        df["true_gender"] = df["true_gender"].astype(int)

    # Normalize true_race to int
    if df["true_race"].dtype == "O":
        df["true_race"] = df["true_race"].astype(int)

    # Map predictions to 0/1 using string labels
    def pred_to_int(x: str) -> int:
        x = str(x).strip().lower()
        if x in ["female", "woman", "0"]:
            return 0
        elif x in ["male", "man", "1"]:
            return 1
        else:
            # Unknown / error
            return -1

    df["pred_base_int"] = df["api_pred_baseline"].apply(pred_to_int)
    df["pred_norm_int"] = df["api_pred_normalized"].apply(pred_to_int)

    df["correct_baseline"] = (
        df["pred_base_int"] == df["true_gender"]
    )
    df["correct_normalized"] = (
        df["pred_norm_int"] == df["true_gender"]
    )

    # Add human-readable names
    df["race_name"] = df["true_race"].map(RACE_MAP)
    df["gender_name"] = df["true_gender"].map(GENDER_MAP)

    # Sanity check
    overall_mean = np.mean(
        np.concatenate(
            [
                df["correct_baseline"].values,
                df["correct_normalized"].values,
            ]
        )
    )
    print(f"\nSanity check: overall mean(correct) = {overall_mean}")

    return df


# --------------------
# Aggregation helpers
# --------------------
def summarize_group(
    df: pd.DataFrame, group_cols: List[str]
) -> pd.DataFrame:
    """
    For groups defined by group_cols, compute:
      - baseline acc + CI
      - normalized acc + CI
      - delta
      - McNemar stats
    """
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n = len(sub)
        base_acc = sub["correct_baseline"].mean()
        norm_acc = sub["correct_normalized"].mean()
        delta = norm_acc - base_acc

        base_ci = binomial_ci(base_acc, n)
        norm_ci = binomial_ci(norm_acc, n)

        chi2, p_val, b, c = mcnemar_test(
            sub["correct_baseline"].values,
            sub["correct_normalized"].values,
        )

        row = {
            "n": n,
            "baseline": base_acc,
            "baseline_ci_low": base_ci[0],
            "baseline_ci_high": base_ci[1],
            "normalized": norm_acc,
            "normalized_ci_low": norm_ci[0],
            "normalized_ci_high": norm_ci[1],
            "delta": delta,
            "mcnemar_chi2": chi2,
            "mcnemar_p": p_val,
            "b_correct_base_wrong_norm": b,
            "c_wrong_base_correct_norm": c,
        }

        # Attach group keys
        for col, val in zip(group_cols, keys):
            row[col] = val

        rows.append(row)

    out_df = pd.DataFrame(rows)
    # Ensure group_cols appear first
    out_df = out_df[group_cols + [c for c in out_df.columns if c not in group_cols]]
    return out_df.sort_values(group_cols).reset_index(drop=True)


def compute_bias_amplification(
    race_gender_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given race × gender summary (with baseline/normalized/delta),
    compute bias amplification per race:

      BA_race = delta_female - delta_male

    Negative BA means women are hurt more than men by normalization.
    """
    rows = []
    for race, sub in race_gender_df.groupby("race_name"):
        fem_row = sub[sub["gender_name"] == "female"]
        male_row = sub[sub["gender_name"] == "male"]

        if fem_row.empty or male_row.empty:
            continue

        fem_delta = fem_row["delta"].iloc[0]
        male_delta = male_row["delta"].iloc[0]
        ba = fem_delta - male_delta

        rows.append(
            {
                "race_name": race,
                "delta_female": fem_delta,
                "delta_male": male_delta,
                "bias_amplification": ba,
            }
        )

    return pd.DataFrame(rows).sort_values("race_name").reset_index(drop=True)


# --------------------
# Plotting
# --------------------
def plot_overall(overall_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(5, 4))
    conds = overall_df["condition"].tolist()
    accs = overall_df["acc"].tolist()
    x = np.arange(len(conds))

    plt.bar(x, accs)
    plt.xticks(x, conds)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy: Baseline vs Normalized")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved overall plot to {out_path}")


def plot_by_race(race_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 4))
    races = race_df["race_name"].tolist()
    base = race_df["baseline"].tolist()
    norm = race_df["normalized"].tolist()

    x = np.arange(len(races))
    width = 0.35

    plt.bar(x - width / 2, base, width, label="Baseline")
    plt.bar(x + width / 2, norm, width, label="Normalized")

    plt.xticks(x, races, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Race (Baseline vs Normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved race plot to {out_path}")


def plot_bias_amplification(ba_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 4))
    races = ba_df["race_name"].tolist()
    ba_vals = ba_df["bias_amplification"].tolist()
    x = np.arange(len(races))

    plt.bar(x, ba_vals)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, races, rotation=45, ha="right")
    plt.ylabel("Δ_female - Δ_male")
    plt.title("Bias Amplification per Race\n(Negative = women hurt more)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved bias amplification plot to {out_path}")


# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default=os.path.join(
            RESULTS_DIR, "hf_fairface_gender_frontish_validation.csv"
        ),
    )
    parser.add_argument(
        "--normalized_csv",
        type=str,
        default=os.path.join(
            RESULTS_DIR, "hf_fairface_gender_frontish_validation_norm.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=RESULTS_DIR,
        help="Where to save summary CSVs and plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_results(args.baseline_csv, args.normalized_csv)

    # ------------------------
    # Overall accuracy
    # ------------------------
    overall = []
    for condition, col in [
        ("baseline", "correct_baseline"),
        ("normalized", "correct_normalized"),
    ]:
        acc = df[col].mean()
        n = len(df)
        overall.append({"condition": condition, "acc": acc, "count": n})
    overall_df = pd.DataFrame(overall)
    print("\n==========================\n OVERALL ACCURACY \n==========================")
    print(overall_df)

    # ------------------------
    # Grouped stats
    # ------------------------
    race_stats = summarize_group(df, ["race_name"])
    gender_stats = summarize_group(df, ["gender_name"])
    race_gender_stats = summarize_group(df, ["race_name", "gender_name"])

    print("\n==========================\n BY RACE \n==========================")
    print(race_stats[["race_name", "baseline", "normalized", "delta"]])

    print("\n==========================\n BY GENDER \n==========================")
    print(gender_stats[["gender_name", "baseline", "normalized", "delta"]])

    print("\n==========================\n BY RACE × GENDER \n==========================")
    print(
        race_gender_stats[
            ["race_name", "gender_name", "baseline", "normalized", "delta"]
        ]
    )

    # ------------------------
    # Bias amplification
    # ------------------------
    ba_df = compute_bias_amplification(race_gender_stats)
    print("\n==========================\n BIAS AMPLIFICATION (Δ_female - Δ_male) \n==========================")
    print(ba_df)

    # ------------------------
    # Save CSVs
    # ------------------------
    overall_df.to_csv(
        os.path.join(args.out_dir, "summary_overall.csv"),
        index=False,
    )
    race_stats.to_csv(
        os.path.join(args.out_dir, "summary_by_race.csv"),
        index=False,
    )
    gender_stats.to_csv(
        os.path.join(args.out_dir, "summary_by_gender.csv"),
        index=False,
    )
    race_gender_stats.to_csv(
        os.path.join(args.out_dir, "summary_by_race_gender.csv"),
        index=False,
    )
    ba_df.to_csv(
        os.path.join(args.out_dir, "summary_bias_amplification.csv"),
        index=False,
    )

    # ------------------------
    # Plots
    # ------------------------
    plot_overall(
        overall_df,
        os.path.join(args.out_dir, "plot_overall_acc.png"),
    )
    plot_by_race(
        race_stats,
        os.path.join(args.out_dir, "plot_by_race_acc.png"),
    )
    plot_bias_amplification(
        ba_df,
        os.path.join(args.out_dir, "plot_bias_amplification.png"),
    )

    print(f"\nSaved summary CSVs and plots to {args.out_dir}")


if __name__ == "__main__":
    main()