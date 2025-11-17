import os
import sys
import argparse
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
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
        - labels.csv      (columns: filename, gender, race)
        - <images...>     (filenames in labels.csv)

    We DO NOT store normalized images on disk.
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

    def __getitem__(self, idx: int):
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
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        print("[WARN] Model has no `.classifier` attribute; check architecture.")


# -----------------------------
# Simple preprocessing
# -----------------------------
class SimplePreprocessor:
    """
    Re-implements the essential part of the HF feature extractor for training:
      - resize to expected size
      - convert to tensor
      - normalize by image_mean / image_std
    """

    def __init__(self, feat_extractor):
        # --- size handling ---
        sz = getattr(feat_extractor, "size", 224)

        # Normalize size into (H, W) ints
        if isinstance(sz, dict):
            h = sz.get("height", sz.get("shortest_edge", 224))
            w = sz.get("width", sz.get("shortest_edge", 224))
        elif isinstance(sz, (list, tuple)):
            if len(sz) == 2:
                h, w = sz
            else:
                h = w = sz[0]
        else:
            h = w = sz

        self.height = int(h)
        self.width = int(w)

        # --- mean/std ---
        self.mean = getattr(feat_extractor, "image_mean", [0.5, 0.5, 0.5])
        self.std = getattr(feat_extractor, "image_std", [0.5, 0.5, 0.5])

        # make them tensors (for broadcasting)
        self.mean_tensor = torch.tensor(self.mean).view(3, 1, 1)
        self.std_tensor = torch.tensor(self.std).view(3, 1, 1)

        print("Preprocessor size (HxW):", self.height, self.width)
        print("Preprocessor mean:", self.mean)
        print("Preprocessor std:", self.std)

    def __call__(self, imgs: List[Image.Image]) -> torch.Tensor:
        """
        imgs: list of PIL RGB images
        returns: float32 tensor of shape (B, 3, H, W)
        """
        tensor_list = []
        for img in imgs:
            # resize with PIL
            img_resized = img.resize((self.width, self.height), resample=Image.BILINEAR)

            # (H, W, C) in [0,1]
            arr = np.array(img_resized, dtype="float32") / 255.0   # (H, W, C)
            arr = torch.from_numpy(arr).permute(2, 0, 1)           # (C, H, W)

            # normalize
            arr = (arr - self.mean_tensor) / self.std_tensor
            tensor_list.append(arr)

        batch = torch.stack(tensor_list, dim=0)  # (B, 3, H, W)
        return batch


# -----------------------------
# Train / eval loops
# -----------------------------
def train_epoch(model, preprocessor, loader, optimizer, device: str = "cpu"):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        labels = labels.to(device)

        pixel_values = preprocessor(imgs).to(device)  # (B, 3, H, W)
        outputs = model(pixel_values=pixel_values)
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
def eval_epoch(model, preprocessor, loader, device: str = "cpu"):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        labels = labels.to(device)

        pixel_values = preprocessor(imgs).to(device)
        outputs = model(pixel_values=pixel_values)
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_train",
        type=int,
        default=2000,
        help="Optional cap on number of training examples (for speed).",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=1000,
        help="Optional cap on number of validation examples.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Fine-tuning FairFace on SKIN-NORMALIZED TRAIN images ===")
    print("Data root :", args.data_root)
    print("Model dir :", args.model_dir)
    print("Out dir   :", args.out_dir)

    # 🔒 Force CPU to avoid CUDA/A40 compatibility hell
    device = "cpu"
    print("Device    :", device)

    # Datasets
    full_train_ds = FrontishFairFaceDataset(
        args.data_root, split="train", use_skin_norm=True
    )
    full_val_ds = FrontishFairFaceDataset(
        args.data_root, split="validation", use_skin_norm=True
    )

    # Optional subsampling for speed
    if args.max_train is not None and args.max_train < len(full_train_ds):
        train_indices = list(range(args.max_train))
        train_ds = Subset(full_train_ds, train_indices)
        print(f"Using subset of TRAIN: {len(train_ds)} examples (max_train={args.max_train})")
    else:
        train_ds = full_train_ds
        print(f"Using FULL TRAIN: {len(train_ds)} examples")

    if args.max_val is not None and args.max_val < len(full_val_ds):
        val_indices = list(range(args.max_val))
        val_ds = Subset(full_val_ds, val_indices)
        print(f"Using subset of VAL: {len(val_ds)} examples (max_val={args.max_val})")
    else:
        val_ds = full_val_ds
        print(f"Using FULL VAL: {len(val_ds)} examples")

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

    # Model + feature extractor config (but we don't use HF FE in the loop)
    print("Loading base model from:", args.model_dir)
    hf_feat = AutoFeatureExtractor.from_pretrained(args.model_dir)
    preprocessor = SimplePreprocessor(hf_feat)

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
            model, preprocessor, train_loader, optimizer, device
        )
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        val_loss, val_acc = eval_epoch(
            model, preprocessor, val_loader, device
        )
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(
                f"New best val acc = {best_val_acc:.4f}. Saving to {args.out_dir} ..."
            )
            model.save_pretrained(args.out_dir)
            hf_feat.save_pretrained(args.out_dir)

    print("Done. Best val acc:", best_val_acc)
    print("Fine-tuned model saved in:", args.out_dir)


if __name__ == "__main__":
    main()