import os
import sys
import csv
import argparse

from PIL import Image
from tqdm import tqdm

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from script.apis.fairface_model import FairFaceGenderModel
from script.data_processing.transforms import normalize_skintone


def load_split(data_root: str, split: str):
    """
    Yield (image_path, filename, row_dict) for the given split
    under data_root.

    Expects:
        <data_root>/<split>/labels.csv
        <data_root>/<split>/<image files...>
    """
    base_dir = os.path.join(data_root, split)
    csv_path = os.path.join(base_dir, "labels.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labels.csv not found at {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        filename = row["filename"]
        img_path = os.path.join(base_dir, filename)
        yield img_path, filename, row


def main():
    parser = argparse.ArgumentParser(description="Evaluate FairFace gender model on a subset.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"],
                        help="Which split to evaluate.")
    parser.add_argument("--use_norm", action="store_true",
                        help="Apply skin-tone normalization before feeding the model.")
    parser.add_argument("--out_csv", default=None,
                        help="Output CSV path (default: metadata/results/fairface_<split>[_norm].csv).")
    parser.add_argument("--data_root", default=None,
                        help="Root directory containing <split>/labels.csv. "
                             "Default: data/cleaned/frontish")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional maximum number of images to process (for smoke tests).")
    parser.add_argument("--weights", default=None,
                        help="Path to FairFace gender model weights (.pth). "
                             "Default: metadata/models/fairface_gender_resnet18.pth")
    args = parser.parse_args()

    # Default data root: full cleaned frontish
    if args.data_root is None:
        args.data_root = os.path.join(PROJECT_ROOT, "data", "cleaned", "frontish")

    model = FairFaceGenderModel(weights_path=args.weights or None)

    # Default output path
    if args.out_csv is None:
        out_dir = os.path.join(PROJECT_ROOT, "metadata", "results")
        os.makedirs(out_dir, exist_ok=True)
        suffix = "_norm" if args.use_norm else ""
        root_tag = "mini" if "mini_eval" in args.data_root else "frontish"
        args.out_csv = os.path.join(out_dir, f"{model.name}_{root_tag}_{args.split}{suffix}.csv")

    print("Model         :", model.name)
    print("Split         :", args.split)
    print("Data root     :", args.data_root)
    print("Skin norm     :", args.use_norm)
    print("Max images    :", args.max_images)
    print("Output CSV    :", args.out_csv)

    results = []
    count = 0

    for img_path, filename, row in tqdm(load_split(args.data_root, args.split)):
        if args.max_images is not None and count >= args.max_images:
            break

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

        pred_dict = model.predict_gender(img)
        pred_label = pred_dict.get("pred_label", "unknown")

        results.append({
            "filename": filename,
            "true_gender": row.get("gender", ""),
            "true_race": row.get("race", ""),
            "api_pred": pred_label,
            "raw": pred_dict.get("raw"),
        })

        count += 1

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