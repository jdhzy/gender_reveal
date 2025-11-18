import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "metadata", "results")

RACE_MAP = {
    0: "Black", 1: "East Asian", 2: "Indian",
    3: "Latino_Hispanic", 4: "Middle Eastern",
    5: "Southeast Asian", 6: "White",
}
GENDER_MAP = {0: "female", 1: "male"}


def pred_to_int(x):
    x = str(x).strip().lower()
    if x in ["female","woman","0"]: return 0
    if x in ["male","man","1"]: return 1
    return -1


def load_csv(path, name):
    df = pd.read_csv(path)
    df["condition"] = name
    df["pred_int"] = df["api_pred"].apply(pred_to_int)
    df["true_gender"] = df["true_gender"].astype(int)
    df["true_race"] = df["true_race"].astype(int)
    df["correct"] = df["pred_int"] == df["true_gender"]
    df["race_name"] = df["true_race"].map(RACE_MAP)
    df["gender_name"] = df["true_gender"].map(GENDER_MAP)
    return df


def summarize(df, group_cols):
    rows=[]
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple): keys = (keys,)
        n=len(sub)
        acc=sub["correct"].mean()
        se=np.sqrt(acc*(1-acc)/n)
        lo,hi=acc-1.96*se, acc+1.96*se
        row={col:keys[i] for i,col in enumerate(group_cols)}
        row.update({"n":n,"acc":acc,"ci_low":lo,"ci_high":hi})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_group(df, category, out_file):
    pivot = df.pivot(index=category, columns="condition", values="acc")
    pivot.plot(kind="bar")
    plt.ylabel("Accuracy")
    plt.title(f"Baseline Accuracy by {category}")
    plt.tight_layout()
    plt.savefig(out_file)
    print("Saved", out_file)


def main():

    base_rgb_csv  = os.path.join(RESULTS_DIR, "base_eval_on_rgb.csv")
    base_norm_csv = os.path.join(RESULTS_DIR, "base_eval_on_norm.csv")

    df = pd.concat([
        load_csv(base_rgb_csv, "base_rgb"),
        load_csv(base_norm_csv, "base_norm"),
    ])

    # OVERALL
    overall = summarize(df, ["condition"])
    print("\n=== OVERALL ===")
    print(overall)
    plot_group(overall, "condition",
               os.path.join(RESULTS_DIR, "baseline_overall.png"))

    # RACE
    by_race = summarize(df, ["race_name", "condition"])
    plot_group(by_race, "race_name",
               os.path.join(RESULTS_DIR, "baseline_race.png"))

    # GENDER
    by_gender = summarize(df, ["gender_name", "condition"])
    plot_group(by_gender, "gender_name",
               os.path.join(RESULTS_DIR, "baseline_gender.png"))

    # RACE × GENDER
    by_rg = summarize(df, ["race_name","gender_name","condition"])
    plot_group(by_rg, "race_name",
               os.path.join(RESULTS_DIR, "baseline_race_gender.png"))

    # Worst group
    worst = by_rg.sort_values("acc").head(5)
    print("\n=== WORST GROUPS ===")
    print(worst)


if __name__ == "__main__":
    main()