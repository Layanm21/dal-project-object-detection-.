#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "positive", "pos"}:
            return 1
        if s in {"false", "0", "no", "negative", "neg"}:
            return 0
    try:
        return int(v)
    except Exception:
        raise ValueError(f"Unsupported label value: {v}")


class ImgClsDataset(Dataset):
    def __init__(self, df, image_col, label_col, image_root=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = Path(row[self.image_col])
        if self.image_root and not p.is_absolute():
            p = self.image_root / p

        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)

        y = float(normalize_label(row[self.label_col]))
        return img, torch.tensor(y, dtype=torch.float32)


def build_model(model_name="resnet18", pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 1)
        return model
    if model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 1)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits)
        probs.extend(p.detach().cpu().numpy().tolist())
        ys.extend(y.detach().cpu().numpy().tolist())

    probs = np.array(probs, dtype=float)
    ys = np.array(ys, dtype=int)
    return probs, ys


def best_threshold_by_f1(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser("Imbalanced Binary Classifier Trainer")
    parser.add_argument("--csv", required=True, help="CSV path containing image+label columns")
    parser.add_argument("--image-col", default="image_path", help="Image path column in CSV")
    parser.add_argument("--label-col", default="label", help="Label column in CSV (true/false or 1/0)")
    parser.add_argument("--image-root", default="", help="Optional root to prepend to image paths")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--outdir", default="runs/imbalance_cls")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv).dropna(subset=[args.image_col, args.label_col]).copy()
    df[args.label_col] = df[args.label_col].apply(normalize_label).astype(int)
    pos = int((df[args.label_col] == 1).sum())
    neg = int((df[args.label_col] == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError(f"Need both classes. found pos={pos}, neg={neg}")

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df[args.label_col],
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ImgClsDataset(train_df, args.image_col, args.label_col, args.image_root, train_tf)
    val_ds = ImgClsDataset(val_df, args.image_col, args.label_col, args.image_root, val_tf)

    # Oversample minority class via sampler
    train_labels = train_df[args.label_col].values.astype(int)
    class_count = np.bincount(train_labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_count, 1)
    sample_weights = np.array([class_weights[y] for y in train_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, pretrained=True).to(device)

    # Penalize missing positives more strongly
    train_pos = int((train_df[args.label_col] == 1).sum())
    train_neg = int((train_df[args.label_col] == 0).sum())
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    best_path = outdir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / max(len(train_ds), 1)
        probs, y_true = evaluate(model, val_loader, device)
        th, val_f1 = best_threshold_by_f1(y_true, probs)
        pred = (probs >= th).astype(int)
        val_prec = precision_score(y_true, pred, zero_division=0)
        val_rec = recall_score(y_true, pred, zero_division=0)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_f1": float(val_f1),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "best_threshold": float(th),
        }
        history.append(row)
        print(row)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": th,
                    "model_name": args.model,
                },
                best_path,
            )

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "csv": args.csv,
                "image_col": args.image_col,
                "label_col": args.label_col,
                "total": len(df),
                "positive": pos,
                "negative": neg,
                "best_f1": best_f1,
                "best_ckpt": str(best_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved best model to: {best_path}")
    print(f"Training artifacts: {outdir}")


if __name__ == "__main__":
    main()
