# script/eval/run_ms_face.py

import os
import sys
import csv
import argparse

from PIL import Image
from tqdm import tqdm

# Make repo root importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from script.apis.ms_face_api import MicrosoftFaceClient
from script.data_processing.transforms import normalize_skintone


def load_frontish_split(split: str):
    """
    Yield (image_path, filename, row_dict) for the given split.
    """
    base_dir = os.path.join(PROJECT_ROOT, "data", "cleaned", "frontish", split)
    csv_path = os.path.join(base_dir, "labels.csv")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        filename = row["filename"]
        img_path = os.path.join(base_dir, filename)
        yield img_path, filename, row


def main():
    parser = argparse.ArgumentParser(description="Evaluate Microsoft Face API on FairFace frontish subset.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"],
                        help="Which split to evaluate.")
    parser.add_argument("--use_norm", action="store_true",
                        help="Apply skin-tone normalization before sending to API.")
    parser.add_argument("--out_csv", default=None,
                        help="Output CSV path (default: metadata/results/ms_face_<split>[_norm].csv).")
    args = parser.parse_args()

    client = MicrosoftFaceClient()

    if args.out_csv is None:
        out_dir = os.path.join(PROJECT_ROOT, "metadata", "results")
        os.makedirs(out_dir, exist_ok=True)
        suffix = "_norm" if args.use_norm else ""
        args.out_csv = os.path.join(out_dir, f"{client.name}_{args.split}{suffix}.csv")

    print("API:", client.name)
    print("Split:", args.split)
    print("Skin-tone normalization:", args.use_norm)
    print("Saving to:", args.out_csv)

    results = []

    for img_path, filename, row in tqdm(load_frontish_split(args.split)):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            results.append({
                "filename": filename,
                "true_gender": row.get("gender", ""),
                "true_race": row.get("race", ""),
                "api_pred": "error",
                "error": str(e),
            })
            continue

        if args.use_norm:
            img = normalize_skintone(img)

        pred_dict = client.predict_gender(img)
        pred_label = pred_dict.get("pred_label", "unknown")

        results.append({
            "filename": filename,
            "true_gender": row.get("gender", ""),
            "true_race": row.get("race", ""),
            "api_pred": pred_label,
            "raw": pred_dict.get("raw"),
        })

    fieldnames = ["filename", "true_gender", "true_race", "api_pred", "raw"]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("Done. Wrote", len(results), "rows.")


if __name__ == "__main__":
    main()