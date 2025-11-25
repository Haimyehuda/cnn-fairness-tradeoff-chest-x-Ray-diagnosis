# ===============================================================
# XRT Evaluation v2 – כולל יצירת גריד תמונות נוסף למחקר
# ===============================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# סגנון כללי "מאמר רפואי"
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("seaborn-whitegrid")

MED_BLUE = "#1f77b4"
DARK_BLUE = "#004c8c"

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMMON_PATH = os.path.join(PROJECT_ROOT, "common")

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if COMMON_PATH not in sys.path:
    sys.path.append(COMMON_PATH)

from model import get_model
from utils import get_device
from train_xrt_model import (
    ChestDataset,
    clean_path,
    CHEXPERT_ROOT,
    CKPT_DIR,
    MODEL_ARCH,
)

from torchvision import transforms
from torch.utils.data import DataLoader

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
RANDOM_STATE = 42


# ---------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XRT model + sample grid")

    parser.add_argument(
        "--scenario",
        type=str,
        default="balanced",
        choices=["balanced", "imbalanced"],
        help="Which checkpoint to evaluate",
    )

    parser.add_argument(
        "--n_normal_test",
        type=int,
        default=300,
        help="Number of NORMAL samples",
    )

    parser.add_argument(
        "--n_pneumonia_test",
        type=int,
        default=300,
        help="Number of PNEUMONIA samples",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots inline in Colab",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Eval batch size",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Folder to save plots",
    )

    return parser.parse_args()


# ---------------------------------------------------------------
# Build evaluation subset
# ---------------------------------------------------------------
def build_eval_subset(n_normal_test: int, n_pneumonia_test: int):
    print("\n=== Step 1: Building evaluation subset from CheXpert ===")

    train_csv = os.path.join(CHEXPERT_ROOT, "train.csv")
    df = pd.read_csv(train_csv)

    df["Pneumonia"] = df["Pneumonia"].astype("float32")
    df["No Finding"] = df["No Finding"].astype("float32")

    df_norm = df[(df["Pneumonia"] == 0) & (df["No Finding"] == 1)].copy()
    df_pneu = df[df["Pneumonia"] == 1].copy()

    print(f"NORMAL candidates   : {len(df_norm)}")
    print(f"PNEUMONIA candidates: {len(df_pneu)}")

    for sub in (df_norm, df_pneu):
        sub["rel"] = sub["Path"].apply(clean_path)
        sub["abs"] = sub["rel"].apply(lambda p: os.path.join(CHEXPERT_ROOT, p))

    df_norm = df_norm[df_norm["abs"].apply(os.path.exists)]
    df_pneu = df_pneu[df_pneu["abs"].apply(os.path.exists)]

    print(f"NORMAL available    : {len(df_norm)}")
    print(f"PNEUMONIA available : {len(df_pneu)}")

    n_norm_eval = min(n_normal_test, len(df_norm))
    n_pneu_eval = min(n_pneumonia_test, len(df_pneu))

    print(f"Requested NORMAL    : {n_normal_test} → Using {n_norm_eval}")
    print(f"Requested PNEUMONIA : {n_pneumonia_test} → Using {n_pneu_eval}")

    df_norm_sel = df_norm.sample(n=n_norm_eval, random_state=RANDOM_STATE).assign(
        label=0
    )
    df_pneu_sel = df_pneu.sample(n=n_pneu_eval, random_state=RANDOM_STATE).assign(
        label=1
    )

    eval_df = pd.concat([df_norm_sel, df_pneu_sel], axis=0)
    eval_df = eval_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Total EVAL samples: {len(eval_df)}")

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    eval_ds = ChestDataset(eval_df, eval_transform)
    return eval_df, eval_ds


# ---------------------------------------------------------------
# NEW — save grid of sample images (no display)
# ---------------------------------------------------------------
def save_sample_grid(eval_df, scenario: str, out_dir: str, max_samples: int = 12):
    print(f"\n=== Saving sample grid ({max_samples} images) ===")
    os.makedirs(out_dir, exist_ok=True)

    if len(eval_df) == 0:
        print("No samples.")
        return

    n = min(max_samples, len(eval_df))
    samples = eval_df.sample(n=n, random_state=RANDOM_STATE)

    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, samples.iterrows()):
        try:
            img = plt.imread(row["abs"])
            ax.imshow(img, cmap="gray")
            ax.set_title(CLASS_NAMES[int(row["label"])])
            ax.axis("off")
        except:
            ax.set_title("Error loading")
            ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Sample Images – {scenario} model", fontsize=15, y=0.98)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"sample_grid_{scenario}.png")
    plt.savefig(out_path, dpi=250)
    plt.close(fig)

    print(f"✔ Sample grid saved to: {out_path}")


# ---------------------------------------------------------------
# Load model & checkpoint
# ---------------------------------------------------------------
def load_model_and_ckpt(scenario: str):
    ckpt_paths = {
        "balanced": os.path.join(CKPT_DIR, f"chexpert_{MODEL_ARCH}_balanced.pth"),
        "imbalanced": os.path.join(CKPT_DIR, f"chexpert_{MODEL_ARCH}_imbalanced.pth"),
    }

    ckpt_path = ckpt_paths[scenario]

    print("\n=== XRT Evaluation ===")
    print(f"Scenario   : {scenario}")
    print(f"Model arch : {MODEL_ARCH}")
    print(f"Checkpoint : {ckpt_path}")

    device = get_device()
    print(f"Using device: {device}")

    model = get_model(
        arch=MODEL_ARCH,
        num_classes=2,
        in_channels=1,
        pretrained=(MODEL_ARCH in ["resnet18", "densenet121"]),
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print("✔ Model loaded")
    return model, device, ckpt_path


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------
def run_inference(model, device, eval_loader):
    print("\n=== Step 3: Running inference ===")

    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    print(f"Done. {len(all_targets)} predictions.")
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def compute_metrics(all_targets, all_preds, all_probs):
    print("\n=== Step 4: Computing metrics ===")

    cm = confusion_matrix(all_targets, all_preds)
    overall_acc = (all_targets == all_preds).mean()

    report = classification_report(
        all_targets,
        all_preds,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = float("nan")

    print(f"Accuracy: {overall_acc * 100:.2f}%")
    print(f"AUC: {auc:.3f}")

    return cm, auc, report, overall_acc


# ---------------------------------------------------------------
# Plot CM / ROC / TPR
# ---------------------------------------------------------------
def plot_confusion_matrix(cm, scenario, out_dir, show=False):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
    ax.set_yticklabels(["NORMAL", "PNEUMONIA"])

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.title(f"Confusion Matrix – {scenario}")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"confusion_matrix_{scenario}.png")
    plt.savefig(out_path, dpi=250)
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_curve(all_targets, all_probs, auc, scenario, out_dir, show=False):
    os.makedirs(out_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(all_targets, all_probs)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color=MED_BLUE, linewidth=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")

    plt.grid(axis="both", visible=True, linestyle=":", alpha=0.7)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve – {scenario}")
    plt.legend()

    out_path = os.path.join(out_dir, f"roc_{scenario}.png")
    plt.savefig(out_path, dpi=250)
    if show:
        plt.show()
    plt.close(fig)


def plot_tpr_bars(report, scenario, out_dir, show=False):
    os.makedirs(out_dir, exist_ok=True)

    tprs = [
        report["NORMAL"]["recall"],
        report["PNEUMONIA"]["recall"],
    ]

    fig = plt.figure(figsize=(5, 4))
    plt.bar(["NORMAL", "PNEUMONIA"], tprs, color=MED_BLUE)

    for i, v in enumerate(tprs):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

    plt.ylim(0, 1.05)
    plt.title(f"TPR per class – {scenario}")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"tpr_{scenario}.png")
    plt.savefig(out_path, dpi=250)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    args = parse_args()

    # Build evaluation dataset
    eval_df, eval_ds = build_eval_subset(
        n_normal_test=args.n_normal_test,
        n_pneumonia_test=args.n_pneumonia_test,
    )

    out_dir = os.path.join(args.out_dir, f"xrt_eval_{args.scenario}")
    os.makedirs(out_dir, exist_ok=True)

    # Save extended sample grid (new feature)
    save_sample_grid(eval_df, args.scenario, out_dir, max_samples=12)

    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    model, device, _ = load_model_and_ckpt(args.scenario)

    all_targets, all_preds, all_probs = run_inference(model, device, eval_loader)

    cm, auc, report, _ = compute_metrics(all_targets, all_preds, all_probs)

    plot_confusion_matrix(cm, args.scenario, out_dir, show=args.show)
    plot_roc_curve(all_targets, all_probs, auc, args.scenario, out_dir, show=args.show)
    plot_tpr_bars(report, args.scenario, out_dir, show=args.show)

    print("\n✔ Evaluation completed.")


if __name__ == "__main__":
    main()
