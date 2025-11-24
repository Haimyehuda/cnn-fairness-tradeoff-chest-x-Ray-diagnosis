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
from torch.utils.data import DataLoader

# -----------------------------
# Global config (paths)
# -----------------------------

# תיקייה של הסקריפט: .../project/scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# תיקיית הפרויקט: .../project
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# נוסיף את project ל-PYTHONPATH → ואז אפשר לעשות import common.
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# עכשיו ה-imports מתוך common
from common.dataset import ChestDataset, clean_path, build_xrt_dataset, CHEXPERT_ROOT
from model import get_model
from train import train_model
from utils import get_device

CKPT_DIR = "/content/drive/MyDrive/XRT_Models"
MODEL_ARCH = "densenet121"
BATCH_SIZE = 16
EPOCHS = 2
TRANNING_N = 25
LR = 1e-4


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
        ckpt_path = os.path.join(CKPT_DIR, f"chexpert_{MODEL_ARCH}_balanced.pth")
    elif args.scenario == "imbalanced":
        # כל הדאטה – לא שולחים גבולות
        n_pneu = TRANNING_N * 5  # for GPU - > None
        n_norm = TRANNING_N  # for GPU - > None
        ckpt_path = os.path.join(CKPT_DIR, f"chexpert_{MODEL_ARCH}_imbalanced.pth")
    else:  # custom
        n_pneu = args.n_pneumonia
        n_norm = args.n_normal
        ckpt_path = os.path.join(CKPT_DIR, f"chexpert_{MODEL_ARCH}_custom.pth")

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
        pretrained=(MODEL_ARCH == "resnet18" or MODEL_ARCH == "densenet121"),
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

    print("\n=== TRAINING SUMMARY ===")
    print(f"Model architecture : {MODEL_ARCH}")
    print(f"Pretrained         : True")  # כי אנחנו תמיד עובדים עם DenseNet121
    print(f"Epochs trained     : {EPOCHS}")
    print(f"Train samples      : {len(train_ds)}")
    print(f"Test samples       : {len(test_ds)}")
    if "loss" in history and history["loss"]:
        last_loss = history["loss"][-1]
    else:
        last_loss = "N/A"

    print(f"Final loss         : {last_loss}")
    print(f"Checkpoint saved   : {ckpt_path}")
    print("==================================")


# ===============================================================
# Entry point
# ===============================================================
if __name__ == "__main__":
    train_xrt_model()
