import os
import argparse
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict

# Figure out project root (two levels up from this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def make_mini_eval(split: str, per_group: int, out_root: str, seed: int = 42):
    """
    Build a small, balanced evaluation subset from
    data/cleaned/frontish/<split>/.

    Balances across (race, gender) pairs, sampling up to `per_group`
    images per pair.

    Writes:
        <out_root>/<split>/labels.csv
        <out_root>/<split>/<images...>
    """
    src_dir = os.path.join(PROJECT_ROOT, "data", "cleaned", "frontish", split)
    src_csv = os.path.join(src_dir, "labels.csv")

    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Could not find labels.csv at {src_csv}")

    # Output dir for this split
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading labels from: {src_csv}")
    df = pd.read_csv(src_csv)

    # Normalize column names a bit just in case
    if "filename" not in df.columns:
        raise ValueError("Expected a 'filename' column in labels.csv")
    if "gender" not in df.columns or "race" not in df.columns:
        raise ValueError("Expected 'gender' and 'race' columns in labels.csv")

    # Group by (race, gender)
    grouped = defaultdict(list)
    for idx, row in df.iterrows():
        race = row["race"]
        gender = row["gender"]
        grouped[(race, gender)].append(idx)

    print("Available groups (race, gender) and counts:")
    for (race, gender), idxs in grouped.items():
        print(f"  ({race}, {gender}): {len(idxs)}")

    # Sample per group
    rng = np.random.RandomState(seed)
    selected_indices = []

    for key, idxs in grouped.items():
        race, gender = key
        available = len(idxs)
        n = min(per_group, available)
        if n == 0:
            continue
        chosen = rng.choice(idxs, size=n, replace=False)
        selected_indices.extend(chosen)
        print(f"Selected {n} of {available} for ({race}, {gender})")

    mini_df = df.loc[selected_indices].reset_index(drop=True)

    # Copy images
    print(f"Copying {len(mini_df)} images to {out_dir}")
    for _, row in mini_df.iterrows():
        fname = row["filename"]
        src_img = os.path.join(src_dir, fname)
        dst_img = os.path.join(out_dir, fname)

        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(src_img, dst_img)

    # Write mini labels.csv
    out_csv = os.path.join(out_dir, "labels.csv")
    mini_df.to_csv(out_csv, index=False)
    print(f"Wrote mini-eval labels to: {out_csv}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Build a small balanced mini-eval subset from frontish.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"],
                        help="Which split in data/cleaned/frontish to sample from.")
    parser.add_argument("--per_group", type=int, default=20,
                        help="Number of images per (race, gender) group (max; uses min(count, per_group)).")
    parser.add_argument("--out_root", default=None,
                        help="Root directory for mini-eval output (default: data/mini_eval).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling.")
    args = parser.parse_args()

    if args.out_root is None:
        args.out_root = os.path.join(PROJECT_ROOT, "data", "mini_eval")

    print("Building mini-eval subset with:")
    print("  split      :", args.split)
    print("  per_group  :", args.per_group)
    print("  out_root   :", args.out_root)
    print("  seed       :", args.seed)

    make_mini_eval(args.split, args.per_group, args.out_root, args.seed)


if __name__ == "__main__":
    main()