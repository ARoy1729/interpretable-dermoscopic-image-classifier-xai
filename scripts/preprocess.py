from __future__ import annotations

# Allow this script to be run directly with:
#     python scripts/preprocess.py
# without installing the project as a package first.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
#from torchvision import transforms
from PIL import Image


from src.config import (
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_DIR_1,
    IMAGE_DIR_2,
    METADATA_PATH,
    MODEL_DIR,
    SPLIT_DIR,
    SEED,
)
from src.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare HAM10000 metadata and lesion-aware splits.")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--image-dir-1", type=Path, default=IMAGE_DIR_1)
    parser.add_argument("--image-dir-2", type=Path, default=IMAGE_DIR_2)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    return parser.parse_args()


def attach_image_paths(frame: pd.DataFrame, image_dirs: list[Path]) -> pd.DataFrame:
    lookup = {}
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for path in image_dir.glob("*.jpg"):
            lookup[path.stem] = str(path.resolve())

    result = frame.copy()
    result["image_path"] = result["image_id"].map(lookup)
    missing = result["image_path"].isna()
    if missing.any():
        missing_ids = result.loc[missing, "image_id"].head(10).tolist()
        raise FileNotFoundError(
            f"Could not locate {missing.sum()} images. First missing IDs: {missing_ids}"
        )
    return result


def lesion_aware_splits(frame: pd.DataFrame):
    # HAM10000 contains multiple observations from some patients. Keeping a patient's
    # images together avoids an easy but misleading validation signal.
    groups = frame["lesion_id"].fillna(frame["image_id"])
    y = frame["dx"]

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(frame, y, groups))

    train_val = frame.iloc[train_val_idx].reset_index(drop=True)
    train_groups = train_val["lesion_id"].fillna(train_val["image_id"])
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    train_idx, val_idx = next(inner.split(train_val, train_val["dx"], train_groups))

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    return train, val, test


def main():
    seed_everything(SEED)
    args = parse_args()
    args.split_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {args.metadata}. Put HAM10000_metadata.csv under data/."
        )

    frame = pd.read_csv(args.metadata)
    frame = frame[frame["dx"].isin(CLASS_NAMES)].copy()
    frame = attach_image_paths(frame, [args.image_dir_1, args.image_dir_2])

    train, val, test = lesion_aware_splits(frame)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_csv(args.split_dir / f"{name}.csv", index=False)
        print(f"{name:>5}: {len(split):5d} images | {split['lesion_id'].nunique():4d} lesions")
        print(split["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
        print()

    summary = pd.DataFrame({
        "split": ["train", "val", "test"],
        "images": [len(train), len(val), len(test)],
        "lesions": [train.lesion_id.nunique(), val.lesion_id.nunique(), test.lesion_id.nunique()],
    })
    summary.to_csv(args.split_dir / "split_summary.csv", index=False)

    # Keep one sample image available for a quick preprocessing sanity check in notebooks.
    sample_path = Path(train.iloc[0]["image_path"])
    with Image.open(sample_path) as image:
        preview = image.convert("RGB").resize((224, 224))
        preview.save(args.split_dir / "preprocessing_preview.jpg", quality=92)

    print(f"Saved split metadata to {args.split_dir.resolve()}")


if __name__ == "__main__":
    main()
