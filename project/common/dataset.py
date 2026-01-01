"""
dataset.py
==========

Dataset definition for Chest X-Ray (CheXpert) binary classification.

Responsibilities:
- Represent a single X-ray image + label
- Load images from absolute paths
- Apply standard preprocessing transforms

This module is intentionally minimal and stateless.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class XRTDataset(Dataset):
    """
    Chest X-Ray Dataset (CheXpert-based)

    Expected DataFrame columns:
    - image_path : absolute path to image file
    - label      : string label ("NORMAL" | "PNEUMONIA")
    """

    LABEL_MAP = {
        "NORMAL": 0,
        "PNEUMONIA": 1,
    }

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label_str = row["label"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image as grayscale
        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = self.LABEL_MAP[label_str]

        return image, torch.tensor(label, dtype=torch.long)
