# ===============================================================
# eval_xrt_augmentation.py
# ===============================================================
# Augmentation & Oversampling experiment for XRT (CheXpert)
#
# Usage from Colab (after Init):
#   %cd /content/cnn-fairness-tradeoff-chest-x-Ray-diagnosis/project
#   !python scripts/eval_xrt_augmentation.py --scenario imbalanced --show
#
# - מאמן מודל חדש (ללא שמירה ל-CKPT)
# - אימון על דאטה לא-מאוזן + Oversampling + Augmentation
# - הערכה על סט מאוזן
# - שומר גרפים ותמונות ל:
#   project/results/augmentation/xrt_aug_<scenario>/
# ===============================================================

import os
import sys
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# סגנון כללי "מאמר רפואי"
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("seaborn-whitegrid")

MED_BLUE = "#1f77b4"
DARK_BLUE = "#004c8c"

# ---------------------------------------------------------------
# 1. Paths & imports from project
# ---------------------------------------------------------------

# .../project/experiments/pre_processing/augmentation_xrt.py
SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # .../project/experiments/pre_processing
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)  # .../project/experiments
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)  # .../project

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.model import get_model  # מתוך common/model.py
from common.utils import get_device, set_seed  # מתוך common/utils.py
from common.pipeline.train import train_model  # מתוך common/pipeline/train.py

# נייבא מהסקריפט הקיים את עזרי ה-XRT
from scripts.train_xrt_model import (  # מתוך project/scripts/train_xrt_model.py
    ChestDataset,
    clean_path,
    CHEXPERT_ROOT,
    MODEL_ARCH,
    TRANNING_N,
    BATCH_SIZE,
    LR,
)

from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------
# 2. קבועים
# ---------------------------------------------------------------
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
RANDOM_STATE = 42

# מספר אפוקים לניסוי (אפשר להתאים לפי כוח המחשב)
EPOCHS_EXP = 5

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "augmentation")


# ---------------------------------------------------------------
# 3. ארגומנטים משורת הפקודה
# ---------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="XRT Augmentation+Oversampling experiment (no checkpoint save)."
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="imbalanced",
        choices=["balanced", "imbalanced"],
        help="Scenario configuration (uses same counts as train_xrt_model).",
    )

    parser.add_argument(
        "--eval_per_class",
        type=int,
        default=300,
        help="Number of NORMAL & PNEUMONIA samples in balanced eval subset.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots inline in Colab in addition to saving files.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training/evaluation.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------
# 4. עזרה: טרנספורמציות
# ---------------------------------------------------------------
def get_basic_transform():
    """Transform בסיסי – ללא Augmentation (להערכה)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def get_augmented_transform():
    """Transform לאימון – עם Augmentation עדין."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


# ---------------------------------------------------------------
# 5. בניית דאטה אימון לא-מאוזן (לניסוי)
# ---------------------------------------------------------------
def build_train_df_for_scenario(scenario: str):
    """
    בונה DataFrame אימון (לא-מאוזן) לפי ה-scenario,
    מאותו CSV של CheXpert כמו באימון המקורי.
    """
    print("\n=== Step 1: Building TRAIN set for augmentation experiment ===")

    train_csv = os.path.join(CHEXPERT_ROOT, "train.csv")
    df = pd.read_csv(train_csv)

    # אותו פילטר כמו ב-train_xrt_model.py
    df["Pneumonia"] = df["Pneumonia"].astype("float32")
    df["No Finding"] = df["No Finding"].astype("float32")

    df_norm = df[(df["Pneumonia"] == 0) & (df["No Finding"] == 1)].copy()
    df_pneu = df[df["Pneumonia"] == 1].copy()

    print(f"NORMAL candidates   : {len(df_norm)}")
    print(f"PNEUMONIA candidates: {len(df_pneu)}")

    # ניקוי path + יצירת נתיב מלא
    for sub in (df_norm, df_pneu):
        sub["rel"] = sub["Path"].apply(clean_path)
        sub["abs"] = sub["rel"].apply(lambda p: os.path.join(CHEXPERT_ROOT, p))

    df_norm = df_norm[df_norm["abs"].apply(os.path.exists)]
    df_pneu = df_pneu[df_pneu["abs"].apply(os.path.exists)]

    print(f"NORMAL available    : {len(df_norm)}")
    print(f"PNEUMONIA available : {len(df_pneu)}")

    # כמו ב-train_xrt_model: קובעים גודל לכל מחלקה
    if scenario == "balanced":
        n_norm = TRANNING_N * 2
        n_pneu = TRANNING_N * 2
    else:  # "imbalanced" – כמו המודל הלא-מאוזן
        n_norm = TRANNING_N  # מיעוט
        n_pneu = TRANNING_N * 5  # רוב

    n_norm = min(n_norm, len(df_norm))
    n_pneu = min(n_pneu, len(df_pneu))

    print(f"Using NORMAL (train): {n_norm}")
    print(f"Using PNEUMONIA (train): {n_pneu}")

    df_norm_sel = df_norm.sample(n=n_norm, random_state=RANDOM_STATE).assign(label=0)
    df_pneu_sel = df_pneu.sample(n=n_pneu, random_state=RANDOM_STATE).assign(label=1)

    train_df = pd.concat([df_norm_sel, df_pneu_sel], axis=0)
    train_df = train_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    print(f"Total TRAIN samples (scenario={scenario}): {len(train_df)}")
    return train_df


# ---------------------------------------------------------------
# 6. בניית סט הערכה מאוזן (למדדי הוגנות)
# ---------------------------------------------------------------
def build_balanced_eval_subset(n_per_class: int = 300):
    print("\n=== Step 2: Building BALANCED EVAL subset ===")

    train_csv = os.path.join(CHEXPERT_ROOT, "train.csv")
    df = pd.read_csv(train_csv)

    df["Pneumonia"] = df["Pneumonia"].astype("float32")
    df["No Finding"] = df["No Finding"].astype("float32")

    df_norm = df[(df["Pneumonia"] == 0) & (df["No Finding"] == 1)].copy()
    df_pneu = df[df["Pneumonia"] == 1].copy()

    for sub in (df_norm, df_pneu):
        sub["rel"] = sub["Path"].apply(clean_path)
        sub["abs"] = sub["rel"].apply(lambda p: os.path.join(CHEXPERT_ROOT, p))

    df_norm = df_norm[df_norm["abs"].apply(os.path.exists)]
    df_pneu = df_pneu[df_pneu["abs"].apply(os.path.exists)]

    n_norm_eval = min(n_per_class, len(df_norm))
    n_pneu_eval = min(n_per_class, len(df_pneu))

    print(f"Eval NORMAL    : {n_norm_eval}")
    print(f"Eval PNEUMONIA : {n_pneu_eval}")

    df_norm_sel = df_norm.sample(n=n_norm_eval, random_state=RANDOM_STATE).assign(
        label=0
    )
    df_pneu_sel = df_pneu.sample(n=n_pneu_eval, random_state=RANDOM_STATE).assign(
        label=1
    )

    eval_df = pd.concat([df_norm_sel, df_pneu_sel], axis=0)
    eval_df = eval_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Total EVAL samples  : {len(eval_df)}")

    eval_transform = get_basic_transform()
    eval_ds = ChestDataset(eval_df, eval_transform)

    return eval_df, eval_ds


# ---------------------------------------------------------------
# 7. WeightedRandomSampler ל-oversampling
# ---------------------------------------------------------------
def build_oversampling_sampler(train_df: pd.DataFrame):
    labels = train_df["label"].values
    class_counts = np.bincount(labels)
    print(f"Class counts (train, before sampling): {dict(enumerate(class_counts))}")

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    # מספר דגימות באימון (למשל ~2 * max_count → "מאוזן" באפוק)
    num_samples = int(2 * class_counts.max())

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=num_samples,
        replacement=True,
    )

    print(f"Using WeightedRandomSampler with num_samples={num_samples}")
    return sampler


# ---------------------------------------------------------------
# 8. הצגת דגימות מהסט (למחקר)
# ---------------------------------------------------------------
def show_and_save_sample_grid(
    eval_df, out_dir: str, scenario: str, max_samples: int = 12, show: bool = False
):
    print("\n=== Step 3: Showing & saving sample images from EVAL set ===")
    os.makedirs(out_dir, exist_ok=True)

    if len(eval_df) == 0:
        print("No samples to show.")
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
        except Exception:
            ax.set_title("Error")
            ax.text(0.5, 0.5, "Error\nloading image", ha="center", va="center")
            ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Sample Chest X-Ray Images – augmentation experiment ({scenario})",
        fontsize=15,
        y=0.98,
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"sample_grid_{scenario}.png")
    plt.savefig(out_path, dpi=250)

    if show:
        plt.show()

    plt.close(fig)
    print(f"✔ Sample grid saved to: {out_path}")


# ---------------------------------------------------------------
# 9. אימון עם Oversampling + Augmentation (ללא שמירת מודל)
# ---------------------------------------------------------------
def train_model_with_augmentation(train_df, batch_size: int, device: torch.device):
    print(
        "\n=== Step 4: Training model with Oversampling + Augmentation (NO checkpoint save) ==="
    )

    # הדאטהסט: df + transform עם augmentation
    train_ds = ChestDataset(train_df, transform=get_augmented_transform())

    # sampler לאיזון יחסי
    sampler = build_oversampling_sampler(train_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # כאשר יש sampler, לא משתמשים ב-shuffle
    )

    # בניית מודל מאותה ארכיטקטורה כמו בפרויקט
    model = get_model(
        arch=MODEL_ARCH,
        num_classes=2,
        in_channels=1,
        pretrained=(MODEL_ARCH in ["resnet18", "densenet121"]),
    ).to(device)

    print(f"Using architecture: {MODEL_ARCH}")
    print(f"Training epochs   : {EPOCHS_EXP}")
    print(f"Learning rate     : {LR}")
    print(f"Batch size        : {batch_size}")

    # אימון (ללא שמירה ל-CKPT)
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        lr=LR,
        epochs=EPOCHS_EXP,
    )

    print("✔ Training finished (model kept in memory only).")
    return model, history


# ---------------------------------------------------------------
# 10. הרצה על סט ההערכה
# ---------------------------------------------------------------
def run_inference(model, device, eval_loader):
    print("\n=== Step 5: Running inference on balanced EVAL subset ===")
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
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob(PNEUMONIA)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print(f"Done. Collected {len(all_targets)} predictions.")
    return all_targets, all_preds, all_probs


# ---------------------------------------------------------------
# 11. חישוב מדדים + טבלה
# ---------------------------------------------------------------
def compute_metrics(all_targets, all_preds, all_probs):
    print("\n=== Step 6: Computing metrics ===")

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    overall_acc = (all_targets == all_preds).mean()

    report_dict = classification_report(
        all_targets,
        all_preds,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float("nan")

    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"ROC-AUC (PNEUMONIA vs NORMAL): {auc:.3f}")

    # טבלת מדדים נוחה
    rows = []
    for cls_idx in [0, 1]:
        cls_name = CLASS_NAMES[cls_idx]
        stats = report_dict[cls_name]
        rows.append(
            {
                "Class": cls_name,
                "Precision": stats["precision"],
                "Recall (TPR)": stats["recall"],
                "F1-score": stats["f1-score"],
                "Support": int(stats["support"]),
            }
        )

    overall_row = {
        "Class": "OVERALL",
        "Precision": report_dict["weighted avg"]["precision"],
        "Recall (TPR)": report_dict["weighted avg"]["recall"],
        "F1-score": report_dict["weighted avg"]["f1-score"],
        "Support": int(report_dict["macro avg"]["support"]),
    }
    rows.append(overall_row)

    metrics_df = pd.DataFrame(rows)

    print("\n=== METRICS TABLE (augmentation experiment) ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return cm, auc, report_dict, overall_acc, metrics_df


# ---------------------------------------------------------------
# 12. גרפים: Confusion Matrix, ROC, TPR per class
# ---------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(cm, scenario: str, out_dir: str, show: bool = False):
    ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title(
        f"Confusion Matrix – augmentation ({scenario})",
        fontsize=14,
        color=DARK_BLUE,
        pad=12,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([CLASS_NAMES[i] for i in tick_marks], fontsize=11)
    ax.set_yticklabels([CLASS_NAMES[i] for i in tick_marks], fontsize=11)

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    # קווי grid עדינים
    ax.set_xticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # מספרים בתוך התאים
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    out_path = os.path.join(out_dir, f"confusion_matrix_aug_{scenario}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    print(f"✔ Confusion matrix saved to: {out_path}")


def plot_roc_curve(
    all_targets, all_probs, auc, scenario: str, out_dir: str, show: bool = False
):
    ensure_dir(out_dir)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)

    fig = plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.3f}",
        color=MED_BLUE,
        linewidth=2.2,
    )

    plt.plot(
        [0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.2, label="Random"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        f"ROC Curve – augmentation ({scenario})", fontsize=14, color=DARK_BLUE, pad=10
    )

    plt.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

    plt.tight_layout()

    out_path = os.path.join(out_dir, f"roc_aug_{scenario}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    print(f"✔ ROC curve saved to: {out_path}")


def plot_tpr_bars(report_dict, scenario: str, out_dir: str, show: bool = False):
    ensure_dir(out_dir)
    tprs = [
        report_dict[CLASS_NAMES[0]]["recall"],
        report_dict[CLASS_NAMES[1]]["recall"],
    ]

    fig = plt.figure(figsize=(5, 4))

    bars = plt.bar(
        list(CLASS_NAMES.values()),
        tprs,
        color=MED_BLUE,
        edgecolor="black",
        linewidth=0.7,
    )

    for bar, v in zip(bars, tprs):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        plt.text(x, y + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    plt.ylim(0, 1.05)
    plt.ylabel("Recall (TPR)", fontsize=12)
    plt.title(
        f"Per-class TPR – augmentation ({scenario})", fontsize=14, color=DARK_BLUE
    )

    plt.tight_layout()

    out_path = os.path.join(out_dir, f"tpr_aug_{scenario}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    print(f"✔ Per-class TPR bar chart saved to: {out_path}")


# ---------------------------------------------------------------
# 13. Main
# ---------------------------------------------------------------
def main():
    args = parse_args()

    # קיבוע רנדום לשחזור
    set_seed(RANDOM_STATE)

    # תיקיית תוצאות לניסוי הזה
    out_dir = os.path.join(RESULTS_ROOT, f"xrt_aug_{args.scenario}")
    ensure_dir(out_dir)

    # 1) בניית דאטה אימון (לא מאוזן) לפי התרחיש
    train_df = build_train_df_for_scenario(args.scenario)

    # 2) בניית סט הערכה מאוזן
    eval_df, eval_ds = build_balanced_eval_subset(n_per_class=args.eval_per_class)

    # 3) גריד של תמונות מהסט להמחשה + לשימוש במחקר
    show_and_save_sample_grid(
        eval_df=eval_df,
        out_dir=out_dir,
        scenario=args.scenario,
        max_samples=12,
        show=args.show,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 4) אימון מודל חדש עם Oversampling+Augmentation (בלי לשמור ckpt)
    device = get_device()
    print(f"\nUsing device: {device}")
    model, history = train_model_with_augmentation(
        train_df=train_df,
        batch_size=args.batch_size,
        device=device,
    )

    # 5) אינפרנס על סט ההערכה המאוזן
    all_targets, all_preds, all_probs = run_inference(model, device, eval_loader)

    # 6) חישוב מדדים
    cm, auc, report_dict, overall_acc, metrics_df = compute_metrics(
        all_targets, all_preds, all_probs
    )

    # 7) גרפים
    plot_confusion_matrix(cm, args.scenario, out_dir, show=args.show)
    plot_roc_curve(all_targets, all_probs, auc, args.scenario, out_dir, show=args.show)
    plot_tpr_bars(report_dict, args.scenario, out_dir, show=args.show)

    print("\n✅ Finished augmentation experiment.")


if __name__ == "__main__":
    main()
