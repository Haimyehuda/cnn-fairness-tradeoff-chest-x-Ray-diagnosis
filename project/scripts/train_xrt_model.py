# train_xrt_model.py
# Standalone training script for XRT (Chest X-Ray) models:
# - balanced model
# - imbalanced model
#
# Usage from Colab (after Init):
#   %cd /content/cnn-fairness-tradeoff-chest-x-Ray-diagnosis/project
#   !python train_xrt_model.py --scenario balanced
#   !python train_xrt_model.py --scenario imbalanced

import os
import sys
import argparse
import time

import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image


# -----------------------------
# Global config (for this script)
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # עולה רמה אחת
COMMON_PATH = os.path.join(PROJECT_ROOT, "common")

# נוסיף את common ל-PYTHONPATH כדי שנוכל לייבא את model/train/utils
if COMMON_PATH not in sys.path:
    sys.path.append(COMMON_PATH)

from model import get_model  # מתוך common/model.py
from pipeline.train import train_model  # מתוך common/pipeline/train.py
from utils import get_device  # מתוך common/utils.py

CHEXPERT_ROOT = "/content/chexpert"  # כמו ב-Init
CKPT_DIR = "/content/drive/MyDrive/XRT_Models"
MODEL_ARCH = "densenet121"
BATCH_SIZE = 16
EPOCHS = 2
TRANNING_N = 25
LR = 1e-4


# ===============================================================
# Dataset helpers (XRT: NORMAL vs PNEUMONIA)
# ===============================================================
class ChestDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["abs"]).convert("L")  # grayscale
        return self.transform(img), int(row["label"])


def clean_path(p):
    prefix = "CheXpert-v1.0-small/"
    if isinstance(p, str) and p.startswith(prefix):
        return p[len(prefix) :]
    return p


def build_xrt_dataset(n_pneumonia=None, n_normal=None, test_size=0.2, random_state=42):
    """
    Build XRT dataset (NORMAL vs PNEUMONIA).

    n_pneumonia / n_normal:
        - אם None → משתמשים בכל הדגימות הזמינות למחלקה.
        - אם המספר גדול מהכמות הזמינה → נעשה min(requested, available).
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

    # Clean paths
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


# ===============================================================
# Training logic
# ===============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XRT model (balanced / imbalanced / custom)."
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="balanced",
        choices=["balanced", "imbalanced", "custom"],
        help="Training scenario: balanced / imbalanced / custom",
    )
    parser.add_argument(
        "--n_pneumonia",
        type=int,
        default=None,
        help="Number of PNEUMONIA samples (custom).",
    )
    parser.add_argument(
        "--n_normal", type=int, default=None, help="Number of NORMAL samples (custom)."
    )

    return parser.parse_args()


def train_xrt_model():
    args = parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    # בוחרים כמה דגימות לפי התרחיש
    if args.scenario == "balanced":
        # ברירת מחדל למודל מאוזן – אפשר לשנות בהמשך
        n_pneu = TRANNING_N * 2  # for GPU - > 1100
        n_norm = TRANNING_N * 2  # for GPU - > 1100
        ckpt_path = os.path.join(CKPT_DIR, "chexpert_resnet18_balanced.pth")
    elif args.scenario == "imbalanced":
        # כל הדאטה – לא שולחים גבולות
        n_pneu = TRANNING_N * 5  # for GPU - > None
        n_norm = TRANNING_N  # for GPU - > None
        ckpt_path = os.path.join(CKPT_DIR, "chexpert_resnet18_imbalanced.pth")
    else:  # custom
        n_pneu = args.n_pneumonia
        n_norm = args.n_normal
        ckpt_path = os.path.join(CKPT_DIR, "chexpert_resnet18_custom.pth")

    print("\n==============================")
    print(f"Scenario   : {args.scenario}")
    print(f"n_pneumonia: {n_pneu}")
    print(f"n_normal   : {n_norm}")
    print(f"Checkpoint : {ckpt_path}")
    print("==============================")

    # בניית הדאטאסט
    train_ds, test_ds = build_xrt_dataset(
        n_pneumonia=n_pneu,
        n_normal=n_norm,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # שליטה פשוטה: GPU או CPU

    if torch.cuda.is_available():
        print("✔ Using GPU (cuda)")
    device = get_device()
    print(f"✔ Using {device}")

    # בינתיים – מתחילים מאפס בכל פעם (בלי המשך מאותו checkpoint)
    model = get_model(
        arch=MODEL_ARCH,
        num_classes=2,
        in_channels=1,
        pretrained=(MODEL_ARCH == "resnet18"),
    ).to(device)

    print("\n🚀 Starting training...")
    t0 = time.time()

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        lr=LR,
        epochs=EPOCHS,
    )

    total_time = time.time() - t0
    print(f"✔ Training finished in {total_time:.2f} seconds")

    # שמירת המודל
    checkpoint = {
        "arch": MODEL_ARCH,
        "epochs": EPOCHS,
        "state_dict": model.state_dict(),
        "history": history,
    }

    torch.save(checkpoint, ckpt_path)
    print(f"✔ Checkpoint saved to: {ckpt_path}")


# ===============================================================
# Entry point
# ===============================================================
if __name__ == "__main__":
    train_xrt_model()
