from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class HAM10000Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        required = {"image_path", "label_idx"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label_idx"])


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_split(name: str, split_dir: Path):
    path = split_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}. Run scripts/preprocess.py first."
        )
    frame = pd.read_csv(path)
    frame["label_idx"] = frame["dx"].map({label: i for i, label in enumerate(CLASS_NAMES)})
    if frame["label_idx"].isna().any():
        bad = frame.loc[frame["label_idx"].isna(), "dx"].unique().tolist()
        raise ValueError(f"Unknown class labels in {path}: {bad}")
    frame["image_path"] = frame["image_path"].map(lambda p: str(Path(p).resolve()))
    return frame
