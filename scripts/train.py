from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    PATIENCE,
    SEED,
    SPLIT_DIR,
    WEIGHT_DECAY,
    FREEZE_BACKBONE
)
from src.data import HAM10000Dataset, build_transforms, load_split
from src.model import build_model
from src.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DenseNet121 HAM10000 classifier.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--output", type=Path, default=CHECKPOINT_PATH)
    return parser.parse_args()


def make_loader(frame, transform, batch_size, workers, shuffle):
    dataset = HAM10000Dataset(frame, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, y_true, y_pred = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item() * labels.size(0))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    loss = sum(losses) / len(loader.dataset)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return loss, macro_f1


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader.dataset)


def main():
    seed_everything(SEED)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_frame = load_split("train", SPLIT_DIR)
    val_frame = load_split("val", SPLIT_DIR)

    train_loader = make_loader(train_frame, build_transforms(True), args.batch_size, args.workers, True)
    val_loader = make_loader(val_frame, build_transforms(False), args.batch_size, args.workers, False)

    model = build_model(pretrained=True).to(device)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_frame["label_idx"].to_numpy(),
    )
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Class weights:")
    for name, weight in zip(CLASS_NAMES, weights):
        print(f"  {name:>5}: {weight:.3f}")


    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    
    optimizer = AdamW(trainable_parameters, lr=args.lr, weight_decay=WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)


    best_f1 = -np.inf
    stalled = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.output.parent / "training_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1", "lr"])

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macro_f1={val_f1:.4f} | lr={current_lr:.2e}")

            writer.writerow([epoch, train_loss, val_loss, val_f1, current_lr])
            history_file.flush()
            scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                stalled = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "class_weights": weights.tolist(),
                    "best_val_macro_f1": float(best_f1),
                }, args.output)
                print(f"Saved new best checkpoint → {args.output}")
            else:
                stalled += 1
                if stalled >= args.patience:
                    print("Early stopping: validation macro-F1 has stopped improving.")
                    break

    print(f"Best validation macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
