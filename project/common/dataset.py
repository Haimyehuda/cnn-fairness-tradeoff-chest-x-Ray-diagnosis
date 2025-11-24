# project/common/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# ============================================================
# 1. Core transforms for Chest X-Ray
# ============================================================


def get_chest_transforms(
    image_size: int = 224,
    in_channels: int = 3,
    augment: bool = False,
):
    """
    Transforms for chest X-ray images.

    Supports:
      - in_channels = 1 (grayscale)
      - in_channels = 3 (RGB for pretrained ResNet)

    augment=True -> light, medically-safe augmentations.
    """

    if in_channels not in (1, 3):
        raise ValueError("in_channels must be 1 or 3")

    # grayscale output channels
    t = [transforms.Grayscale(num_output_channels=in_channels)]

    # safe augmentations:
    if augment:
        t.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        t.append(transforms.Resize((image_size, image_size)))

    t.append(transforms.ToTensor())

    # normalize
    if in_channels == 1:
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    t.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(t)


# ============================================================
# 2. Dataset wrapper to ensure classes map to {0,1}
# ============================================================


class TwoClassDataset(Dataset):
    """
    Wraps an ImageFolder, keeping only two classes:
      class0_name -> label 0
      class1_name -> label 1
    """

    def __init__(self, base_dataset, indices, class0_label, class1_label):
        self.base = base_dataset
        self.indices = np.array(indices)
        self.map = {class0_label: 0, class1_label: 1}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, lbl = self.base[int(self.indices[i])]
        return img, torch.tensor(self.map[int(lbl)], dtype=torch.long)


# ============================================================
# 3. Balanced Chest X-Ray loader
# ============================================================


def load_balanced_chestxray(
    root: str,
    split: str = "train",
    class0_name: str = "NORMAL",
    class1_name: str = "PNEUMONIA",
    per_class: int | None = None,
    image_size: int = 224,
    in_channels: int = 3,
    augment: bool = False,
    seed: int = 42,
):
    """
    Loads a *balanced* subset from Chest X-Ray using ImageFolder.

    Folder structure:
        root/train/NORMAL/*.jpg
        root/train/PNEUMONIA/*.jpg

    per_class=None → take the min(#class0, #class1)
    """
    np.random.seed(seed)

    split_root = os.path.join(root, split)
    transform = get_chest_transforms(image_size, in_channels, augment)
    base = datasets.ImageFolder(split_root, transform=transform)

    if class0_name not in base.class_to_idx:
        raise ValueError(f"{class0_name} not found in dataset.")
    if class1_name not in base.class_to_idx:
        raise ValueError(f"{class1_name} not found in dataset.")

    class0_label = base.class_to_idx[class0_name]
    class1_label = base.class_to_idx[class1_name]

    targets = np.array(base.targets)
    idx0_all = np.where(targets == class0_label)[0]
    idx1_all = np.where(targets == class1_label)[0]

    if per_class is None:
        per_class = min(len(idx0_all), len(idx1_all))

    if per_class > len(idx0_all) or per_class > len(idx1_all):
        raise ValueError("per_class is larger than available samples.")

    idx0 = np.random.choice(idx0_all, per_class, replace=False)
    idx1 = np.random.choice(idx1_all, per_class, replace=False)

    final_idx = np.concatenate([idx0, idx1])

    return TwoClassDataset(base, final_idx, class0_label, class1_label)


# ============================================================
# 4. Imbalanced Chest X-Ray loader
# ============================================================


def load_imbalanced_chestxray(
    root: str,
    split: str = "train",
    class0_name: str = "NORMAL",
    class1_name: str = "PNEUMONIA",
    count_class0: int = 8000,  # majority
    count_class1: int = 500,  # minority
    image_size: int = 224,
    in_channels: int = 3,
    augment: bool = False,
    seed: int = 42,
):
    """
    Loads an *imbalanced* subset from Chest X-Ray using ImageFolder.

    For research:
      - class0_name = majority (label 0)
      - class1_name = minority (label 1)
    """
    np.random.seed(seed)

    split_root = os.path.join(root, split)
    transform = get_chest_transforms(image_size, in_channels, augment)
    base = datasets.ImageFolder(split_root, transform=transform)

    if class0_name not in base.class_to_idx or class1_name not in base.class_to_idx:
        raise ValueError("Class names not found in dataset.")

    class0_label = base.class_to_idx[class0_name]
    class1_label = base.class_to_idx[class1_name]

    targets = np.array(base.targets)
    idx0_all = np.where(targets == class0_label)[0]
    idx1_all = np.where(targets == class1_label)[0]

    if count_class0 > len(idx0_all):
        raise ValueError(f"count_class0={count_class0} > available {len(idx0_all)}")
    if count_class1 > len(idx1_all):
        raise ValueError(f"count_class1={count_class1} > available {len(idx1_all)}")

    idx0 = np.random.choice(idx0_all, count_class0, replace=False)
    idx1 = np.random.choice(idx1_all, count_class1, replace=False)

    final_idx = np.concatenate([idx0, idx1])

    return TwoClassDataset(base, final_idx, class0_label, class1_label)


# ============================================================
# 5. XRT (CheXpert) – NORMAL vs PNEUMONIA via train.csv
# ============================================================

CHEXPERT_ROOT = "/content/chexpert"  # לעדכן אם ב-Init יש נתיב אחר


def clean_path(p: str) -> str:
    """
    מסיר את הפריפיקס 'CheXpert-v1.0-small/' אם קיים.
    """
    prefix = "CheXpert-v1.0-small/"
    if isinstance(p, str) and p.startswith(prefix):
        return p[len(prefix) :]
    return p


class ChestDataset(Dataset):
    """
    Dataset פשוט ל-XRT:
    מקבל DataFrame עם עמודות:
        - abs  (נתיב מלא לתמונה)
        - label (0 = NORMAL, 1 = PNEUMONIA)
    """

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["abs"]).convert("L")  # grayscale
        return self.transform(img), int(row["label"])


def build_xrt_dataset(
    n_pneumonia: int | None = None,
    n_normal: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Build XRT dataset (NORMAL vs PNEUMONIA) מתוך CheXpert train.csv.

    n_pneumonia / n_normal:
        - None → משתמשים בכל הדגימות הזמינות למחלקה.
        - מספר גדול מהכמות הזמינה → min(requested, available).
    """
    print("\n=== XRT – Build Chest X-Ray dataset (NORMAL vs PNEUMONIA) ===")

    train_csv = os.path.join(CHEXPERT_ROOT, "train.csv")
    df = pd.read_csv(train_csv)

    # Filter NORMAL / PNEUMONIA
    df["Pneumonia"] = df["Pneumonia"].astype("float32")
    df["No Finding"] = df["No Finding"].astype("float32")

    df_norm = df[(df["Pneumonia"] == 0) & (df["No Finding"] == 1)].copy()
    df_pneu = df[df["Pneumonia"] == 1].copy()

    print(f"NORMAL candidates   : {len(df_norm)}")
    print(f"PNEUMONIA candidates: {len(df_pneu)}")

    # Clean paths + build absolute path
    for sub in (df_norm, df_pneu):
        sub["rel"] = sub["Path"].apply(clean_path)
        sub["abs"] = sub["rel"].apply(lambda p: os.path.join(CHEXPERT_ROOT, p))

    df_norm = df_norm[df_norm["abs"].apply(os.path.exists)]
    df_pneu = df_pneu[df_pneu["abs"].apply(os.path.exists)]

    print(f"NORMAL available    : {len(df_norm)}")
    print(f"PNEUMONIA available : {len(df_pneu)}")

    # אם לא נשלח ערך → משתמשים בכל הכמות הזמינה
    if n_normal is None:
        target_norm = len(df_norm)
    else:
        target_norm = min(n_normal, len(df_norm))

    if n_pneumonia is None:
        target_pneu = len(df_pneu)
    else:
        target_pneu = min(n_pneumonia, len(df_pneu))

    print(f"Requested NORMAL    : {n_normal if n_normal is not None else 'ALL'}")
    print(f"Requested PNEUMONIA : {n_pneumonia if n_pneumonia is not None else 'ALL'}")
    print(f"Using NORMAL        : {target_norm}")
    print(f"Using PNEUMONIA     : {target_pneu}")

    # דגימה לפי הכמויות הסופיות
    df_norm_sel = df_norm.sample(n=target_norm, random_state=random_state).assign(
        label=0
    )
    df_pneu_sel = df_pneu.sample(n=target_pneu, random_state=random_state).assign(
        label=1
    )

    full_df = pd.concat([df_norm_sel, df_pneu_sel], axis=0)
    full_df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"Total labeled samples (all): {len(full_df)}")

    # Train / test split
    train_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=full_df["label"],
    )

    # טרנספורם בסיסי כמו באימון
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_ds = ChestDataset(train_df, transform)
    test_ds = ChestDataset(test_df, transform)

    print("✔ XRT dataset built successfully.")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples : {len(test_ds)}")

    return train_ds, test_ds
