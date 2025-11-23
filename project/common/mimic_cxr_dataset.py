import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch


class MIMICCXRDataset(Dataset):
    """
    Dataset for MIMIC-CXR with:
        - image loading
        - CheXpert labels
        - demographics (age, gender, ethnicity)
    """

    def __init__(
        self,
        csv_path,
        images_root,
        target_label="Pneumonia",
        transform=None,
        select_positive=None,
        select_negative=None,
        max_samples=None,
    ):
        self.images_root = images_root
        self.target_label = target_label
        self.transform = transform

        df = pd.read_csv(csv_path)

        # Keep only rows that have the target label (0/1)
        df = df[df[target_label].notna()]

        # Convert CheXpert uncertainty "−1" → positive (optional)
        df[target_label] = df[target_label].replace(-1, 1)

        # Filter Positives
        if select_positive is not None:
            pos_df = df[df[target_label] == 1].sample(select_positive, replace=False)
        else:
            pos_df = df[df[target_label] == 1]

        # Filter Negatives
        if select_negative is not None:
            neg_df = df[df[target_label] == 0].sample(select_negative, replace=False)
        else:
            neg_df = df[df[target_label] == 0]

        df = pd.concat([pos_df, neg_df])

        if max_samples is not None:
            df = df.sample(max_samples, replace=False)

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.images_root, row["Path"])

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(row[self.target_label])

        # demographic info
        gender = row.get("gender", None)
        ethnicity = row.get("ethnicity", None)
        age = row.get("age", None)

        return (
            img,
            label,
            {
                "gender": gender,
                "ethnicity": ethnicity,
                "age": age,
            },
        )


def get_mimic_transforms(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_balanced_mimic(csv_path, images_root, target_label, per_class, image_size=224):
    transform = get_mimic_transforms(image_size)
    return MIMICCXRDataset(
        csv_path=csv_path,
        images_root=images_root,
        target_label=target_label,
        transform=transform,
        select_positive=per_class,
        select_negative=per_class,
    )


def load_imbalanced_mimic(
    csv_path, images_root, target_label, majority_count, minority_count, image_size=224
):

    transform = get_mimic_transforms(image_size)
    return MIMICCXRDataset(
        csv_path=csv_path,
        images_root=images_root,
        target_label=target_label,
        transform=transform,
        select_positive=minority_count,
        select_negative=majority_count,
    )
