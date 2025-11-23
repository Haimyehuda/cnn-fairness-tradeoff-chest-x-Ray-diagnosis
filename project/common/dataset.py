# project/common/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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
