import os
import sys
import argparse
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from script.data_processing.transforms import normalize_skintone
from script.apis.fairface_hf_model import DEFAULT_MODEL_DIR
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


# -----------------------------
# Dataset: frontish FairFace
# -----------------------------
class FrontishFairFaceDataset(Dataset):
    """
    Expects:
      data/cleaned/frontish/<split>/
        - labels.csv      (with columns: filename, gender, race)
        - <images...>     (filenames in labels.csv)

    We do NOT store normalized images on disk.
    We apply normalize_skintone() on the fly.
    """

    def __init__(self, root: str, split: str = "train", use_skin_norm: bool = True):
        self.root = root
        self.split = split
        self.use_skin_norm = use_skin_norm

        self.split_dir = os.path.join(self.root, split)
        self.labels_path = os.path.join(self.split_dir, "labels.csv")

        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"labels.csv not found at {self.labels_path}")

        self.df = pd.read_csv(self.labels_path)

        required_cols = {"filename", "gender"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"labels.csv is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = int(row["gender"])  # 0/1

        img_path = os.path.join(self.split_dir, filename)
        img = Image.open(img_path).convert("RGB")

        if self.use_skin_norm:
            img = normalize_skintone(img)

        return img, label


# -----------------------------
# Helper: freeze backbone
# -----------------------------
def freeze_backbone(model: nn.Module):
    """
    Freeze all parameters except the classification head.

    For ViTForImageClassification, the classification head is `model.classifier`.
    If this breaks, print(model) and adjust.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        print("[WARN] Model has no `.classifier` attribute; check architecture.")


# -----------------------------
# Train / eval loops
# -----------------------------
def train_epoch(model, feature_extractor, loader, optimizer, device: str = "cpu"):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        labels = labels.to(device)

        inputs = feature_extractor(images=list(imgs), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def eval_epoch(model, feature_extractor, loader, device: str = "cpu"):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        labels = labels.to(device)

        inputs = feature_extractor(images=list(imgs), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits

        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "cleaned", "frontish"),
        help="Root of cleaned/frontish data (with train/ and validation/ subdirs)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Base HF FairFace model dir (pytorch_model.bin + config.json)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            "metadata",
            "models",
            "fairface_gender_image_detection_norm_ft",
        ),
        help="Where to save fine-tuned model (HF format).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Fine-tuning FairFace on SKIN-NORMALIZED TRAIN images ===")
    print("Data root :", args.data_root)
    print("Model dir :", args.model_dir)
    print("Out dir   :", args.out_dir)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Device    :", device)
    # Force CPU for now (GPU build is too old for A40)
    device = "cpu"
    print("Device    :", device)

    # Datasets
    # 👉 TRAIN: normalized faces (Option A)
    train_ds = FrontishFairFaceDataset(
        args.data_root, split="train", use_skin_norm=True
    )

    # 👉 VAL for early stopping: also normalized (same distribution as train)
    val_ds = FrontishFairFaceDataset(
        args.data_root, split="validation", use_skin_norm=True
    )

    def collate_fn(batch):
        imgs = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return imgs, labels

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Model + processor
    print("Loading base model from:", args.model_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_dir)

    # --- PATCH: make sure size is int / tuple of ints, not strings ---
    sz = feature_extractor.size
    if isinstance(sz, dict):
        # e.g. {"height": "224", "width": "224"}
        new_sz = {}
        for k, v in sz.items():
            try:
                new_sz[k] = int(v)
            except (TypeError, ValueError):
                new_sz[k] = v
        feature_extractor.size = new_sz
    elif isinstance(sz, (list, tuple)):
        feature_extractor.size = tuple(int(v) for v in sz)
    elif isinstance(sz, str):
        # e.g. "224"
        try:
            feature_extractor.size = int(sz)
        except ValueError:
            pass
    # --- END PATCH ---

    model = AutoModelForImageClassification.from_pretrained(args.model_dir)
    model.to(device)

    # Only train classifier head
    freeze_backbone(model)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss, train_acc = train_epoch(
            model, feature_extractor, train_loader, optimizer, device
        )
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        val_loss, val_acc = eval_epoch(
            model, feature_extractor, val_loader, device
        )
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best val acc = {best_val_acc:.4f}. Saving to {args.out_dir} ...")
            model.save_pretrained(args.out_dir)
            feature_extractor.save_pretrained(args.out_dir)

    print("Done. Best val acc:", best_val_acc)
    print("Fine-tuned model saved in:", args.out_dir)


if __name__ == "__main__":
    main()