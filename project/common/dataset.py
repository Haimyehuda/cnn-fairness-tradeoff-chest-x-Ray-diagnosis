# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def load_imbalanced_cifar(cat_count=30, dog_count=500, seed=42):
    """Load CIFAR-10 and create an imbalanced dataset of cats (minority) and dogs (majority)."""
    np.random.seed(seed)
    transform = transforms.Compose([transforms.ToTensor()])

    full = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    classes = full.classes
    CAT, DOG = classes.index("cat"), classes.index("dog")

    targets = np.array(full.targets)
    idx_cat = np.where(targets == CAT)[0]
    idx_dog = np.where(targets == DOG)[0]

    cat_idx = np.random.choice(idx_cat, cat_count, replace=False)
    dog_idx = np.random.choice(idx_dog, dog_count, replace=False)

    final_idx = np.concatenate([cat_idx, dog_idx])
    return TwoClassSubset(full, final_idx, CAT, DOG)


class TwoClassSubset(Dataset):
    """Dataset wrapper that keeps only cats and dogs and maps labels: cat=0, dog=1."""

    def __init__(self, base, indices, cat_label, dog_label):
        self.base = base
        self.indices = indices
        self.map = {cat_label: 0, dog_label: 1}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, lbl = self.base[self.indices[i]]
        return img, torch.tensor(self.map[int(lbl)])
