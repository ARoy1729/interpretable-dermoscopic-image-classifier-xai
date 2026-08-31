from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_PATH, CLASS_NAMES, OUTPUT_DIR, SPLIT_DIR, BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE
from src.data import HAM10000Dataset, build_transforms, load_split
from src.explainability import GradCAM, overlay_heatmap, preprocess_pil, predict
from src.metrics import multiclass_metrics
from src.model import build_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained HAM10000 classifier.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--failure-count", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def collect_predictions(model, loader, device):
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(probabilities.argmax(axis=1).tolist())
            probs.append(probabilities)
    return np.asarray(y_true), np.asarray(y_pred), np.concatenate(probs, axis=0)


def save_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("HAM10000 test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_roc(y_true, probabilities, class_names, path):
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_failure_examples(frame, y_true, y_pred, probabilities, model, device, out_dir, count):
    failed = np.where(y_true != y_pred)[0]
    if len(failed) == 0:
        return pd.DataFrame()

    ranked = failed[np.argsort(probabilities[failed, y_pred[failed]])[::-1]][:count]
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model)
    records = []
    try:
        for rank, idx in enumerate(ranked, start=1):
            row = frame.iloc[int(idx)]
            image = Image.open(row["image_path"]).convert("RGB")
            tensor = preprocess_pil(image).to(device)
            pred = predict(model, tensor, CLASS_NAMES, device)
            heatmap = cam(tensor, pred.index)
            overlay = overlay_heatmap(image, heatmap)
            out_path = out_dir / f"failure_{rank}_{row['image_id']}.png"

            canvas = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE + 56), "white")
            canvas.paste(image.resize((IMAGE_SIZE, IMAGE_SIZE)), (0, 0))
            canvas.paste(overlay.resize((IMAGE_SIZE, IMAGE_SIZE)), (IMAGE_SIZE, 0))
            drawer = ImageDraw.Draw(canvas)
            drawer.text(
                (8, 235),
                f"Actual: {row['dx']}  |  Predicted: {pred.label} ({pred.confidence:.1%})",
                fill="black",
            )
            drawer.text((8, 253), "Left: original  |  Right: Grad-CAM", fill="black")
            canvas.save(out_path)

            records.append({
                "image_id": row["image_id"],
                "actual": row["dx"],
                "predicted": pred.label,
                "confidence": pred.confidence,
                "gradcam_file": out_path.name,
            })
    finally:
        cam.close()

    return pd.DataFrame(records)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    model = build_model_from_checkpoint(checkpoint, device)

    test_frame = load_split("test", SPLIT_DIR)
    test_dataset = HAM10000Dataset(test_frame, transform=build_transforms(False))
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    y_true, y_pred, probabilities = collect_predictions(model, test_loader, device)
    metrics, cm = multiclass_metrics(y_true, y_pred, probabilities, class_names)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)

    prediction_frame = test_frame.copy()
    prediction_frame["predicted_idx"] = y_pred
    prediction_frame["predicted"] = [class_names[i] for i in y_pred]
    prediction_frame["confidence"] = probabilities.max(axis=1)
    prediction_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)

    save_confusion_matrix(cm, class_names, args.output_dir / "confusion_matrix.png")
    save_roc(y_true, probabilities, class_names, args.output_dir / "roc_curves.png")

    failure_frame = save_failure_examples(
        test_frame, y_true, y_pred, probabilities, model, device,
        args.output_dir / "failure_examples", args.failure_count
    )
    failure_dir = args.output_dir / "failure_examples"
    failure_dir.mkdir(parents=True, exist_ok=True)
    if not failure_frame.empty:
        failure_frame.to_csv(failure_dir / "failure_examples.csv", index=False)

    print("\nTest metrics")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved evaluation artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
