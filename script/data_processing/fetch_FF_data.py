import argparse
import os
import csv

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download FairFace via HuggingFace")
    parser.add_argument(
        "--version",
        default="1.25",
        choices=["0.25", "1.25"],
        help="FairFace config to use",
    )
    parser.add_argument(
        "--max_per_split",
        type=int,
        default=None,
        help="Optional cap on number of samples per split.",
    )
    args = parser.parse_args()

    # -----------------------------------------
    # Compute project root and target data dir
    # -----------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))          # script/data_processing/
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))  # gender_reveal/
    out_dir = os.path.join(project_root, "data", "fairface")         # gender_reveal/data/fairface
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving FairFace to: {out_dir}")

    # -----------------------------------------
    # Load the dataset
    # -----------------------------------------
    print(f"Loading FairFace {args.version} from HuggingFace…")
    ds = load_dataset("HuggingFaceM4/FairFace", args.version)

    # -----------------------------------------
    # Process all splits
    # -----------------------------------------
    for split_name, split_ds in ds.items():

        if args.max_per_split is not None:
            n = min(args.max_per_split, len(split_ds))
            split_ds = split_ds.select(range(n))

        # Where to save this split
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        csv_path = os.path.join(split_dir, "labels.csv")
        print(f"Saving {split_name} split to {split_dir}")

        # Write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "gender", "race", "age"])

            for idx, example in enumerate(tqdm(split_ds, desc=f"{split_name}")):
                img = example["image"]
                filename = f"{idx:07d}.jpg"
                img.save(os.path.join(split_dir, filename))

                writer.writerow([
                    filename,
                    example.get("gender", ""),
                    example.get("race", ""),
                    example.get("age", ""),
                ])

    print("Done downloading FairFace.")


if __name__ == "__main__":
    main()