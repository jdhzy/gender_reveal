import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt


# --------------------
# Path setup
# --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

DATA_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, "data", "cleaned", "frontish")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "metadata", "results")


# --------------------
# Mappings (same as analysis scripts)
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


def load_labels(data_root: str, split: str) -> pd.DataFrame:
    """
    Load labels.csv for a given split under the cleaned data root.
    """
    split_dir = os.path.join(data_root, split)
    csv_path = os.path.join(split_dir, "labels.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labels.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"filename", "gender", "race"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    return df


def plot_pies(df: pd.DataFrame, split: str, out_dir: str):
    """
    Create pie charts for race distribution and gender distribution.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Race distribution
    race_counts = df["race"].value_counts().sort_index()
    race_labels = [RACE_MAP.get(int(i), str(i)) for i in race_counts.index]

    plt.figure(figsize=(6, 6))
    plt.pie(
        race_counts.values,
        labels=race_labels,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title(f"Race distribution ({split})")
    plt.axis("equal")
    race_out = os.path.join(out_dir, f"cleaned_{split}_race_pie.png")
    plt.savefig(race_out, bbox_inches="tight")
    plt.close()
    print("Saved:", race_out)

    # Gender distribution
    gender_counts = df["gender"].value_counts().sort_index()
    gender_labels = [GENDER_MAP.get(int(i), str(i)) for i in gender_counts.index]

    plt.figure(figsize=(6, 6))
    plt.pie(
        gender_counts.values,
        labels=gender_labels,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title(f"Gender distribution ({split})")
    plt.axis("equal")
    gender_out = os.path.join(out_dir, f"cleaned_{split}_gender_pie.png")
    plt.savefig(gender_out, bbox_inches="tight")
    plt.close()
    print("Saved:", gender_out)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize race and gender distributions in cleaned frontish data."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT_DEFAULT,
        help="Root of cleaned/frontish data (with train/ and validation/ subdirs).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory to save pie charts.",
    )
    args = parser.parse_args()

    df = load_labels(args.data_root, args.split)
    print(f"Loaded {len(df)} rows from {args.split} split.")

    plot_pies(df, args.split, args.out_dir)


if __name__ == "__main__":
    main()

