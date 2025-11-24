# eval_xrt_model.py
# ===============================================================
# XRT Evaluation – balanced / imbalanced model (script)
# Usage from Colab (after Init):
#   %cd /content/cnn-fairness-tradeoff-chest-x-Ray-diagnosis/project
#   !python scripts/eval_xrt_model.py --scenario balanced --show
#   !python scripts/eval_xrt_model.py --scenario imbalanced --show
# ===============================================================

import os
import sys
import argparse

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

# ---------------------------------------------------------------
# 1. Paths & imports from project (בדומה ל-train_xrt_model.py)
# ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMMON_PATH = os.path.join(PROJECT_ROOT, "common")

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if COMMON_PATH not in sys.path:
    sys.path.append(COMMON_PATH)

from model import get_model  # מתוך common/model.py
from utils import get_device  # מתוך common/utils.py
from train_xrt_model import (  # מתוך scripts/train_xrt_model.py
    ChestDataset,
    clean_path,
    CHEXPERT_ROOT,
    CKPT_DIR,
    MODEL_ARCH,
)

from torchvision import transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------
# 2. קבועים / הגדרות
# ---------------------------------------------------------------
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate XRT model (balanced / imbalanced)."
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="balanced",
        choices=["balanced", "imbalanced"],
        help="Which checkpoint to evaluate: balanced / imbalanced",
    )

    parser.add_argument(
        "--n_normal_test",
        type=int,
        default=300,
        help="Number of NORMAL samples in eval subset.",
    )

    parser.add_argument(
        "--n_pneumonia_test",
        type=int,
        default=300,
        help="Number of PNEUMONIA samples in eval subset.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots inline in Colab in addition to saving files.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation loader.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Where to save plots (PNG).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------
# 3. Build evaluation subset (NORMAL vs PNEUMONIA)
# ---------------------------------------------------------------
def build_eval_subset(n_normal_test: int, n_pneumonia_test: int):
    print("\n=== Step 1: Building evaluation subset from CheXpert ===")
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

    # בוחרים כמה דוגמאות
    n_norm_eval = min(n_normal_test, len(df_norm))
    n_pneu_eval = min(n_pneumonia_test, len(df_pneu))

    print(f"Requested NORMAL    : {n_normal_test} -> Using {n_norm_eval}")
    print(f"Requested PNEUMONIA : {n_pneumonia_test} -> Using {n_pneu_eval}")

    df_norm_sel = df_norm.sample(n=n_norm_eval, random_state=RANDOM_STATE).assign(
        label=0
    )
    df_pneu_sel = df_pneu.sample(n=n_pneu_eval, random_state=RANDOM_STATE).assign(
        label=1
    )

    eval_df = pd.concat([df_norm_sel, df_pneu_sel], axis=0)
    eval_df = eval_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Total EVAL samples  : {len(eval_df)}")

    # טרנספורמציה כמו באימון
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
# 3b. הצגת דוגמאות מהדאטה
# ---------------------------------------------------------------
def show_sample_images(eval_df, max_samples: int = 6):
    """
    מציג כמה דוגמאות מהסט להמחשה (NORMAL / PNEUMONIA).
    """
    print("\n=== Step 2: Showing sample images from evaluation set ===")

    if len(eval_df) == 0:
        print("No samples to show.")
        return

    n = min(max_samples, len(eval_df))
    samples = eval_df.sample(n=n, random_state=RANDOM_STATE)

    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, samples.iterrows()):
        img_path = row["abs"]
        label = int(row["label"])
        try:
            img = plt.imread(img_path)
            ax.imshow(img, cmap="gray")
            ax.set_title(CLASS_NAMES.get(label, str(label)))
            ax.axis("off")
        except Exception as e:
            ax.set_title("Error")
            ax.text(0.5, 0.5, "Error\nloading image", ha="center", va="center")
            ax.axis("off")

    # אם פחות מתאים 6, מסתירים את השאר
    for ax in axes[len(samples) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# 4. Load model + checkpoint
# ---------------------------------------------------------------
def load_model_and_ckpt(scenario: str):
    ckpt_paths = {
        "balanced": os.path.join(CKPT_DIR, f"chexpert_{MODEL_ARCH}_balanced.pth"),
        "imbalanced": os.path.join(CKPT_DIR, f"chexpert_{MODEL_ARCH}_imbalanced.pth"),
    }

    if scenario not in ckpt_paths:
        raise ValueError(f"Unknown scenario: {scenario}")

    ckpt_path = ckpt_paths[scenario]

    print("=== XRT Evaluation ===")
    print(f"Scenario   : {scenario}")
    print(f"Model arch : {MODEL_ARCH}")
    print(f"Checkpoint : {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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

    print("✔ Model loaded successfully")
    return model, device, ckpt_path


# ---------------------------------------------------------------
# 5. Run inference
# ---------------------------------------------------------------
def run_inference(model, device, eval_loader):
    print("\n=== Step 3: Running inference on evaluation subset ===")
    all_targets = []
    all_preds = []
    all_probs = []

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
# 6. Compute metrics
# ---------------------------------------------------------------
def compute_metrics(all_targets, all_preds, all_probs, scenario: str, n_samples: int):
    print("\n=== Step 4: Computing metrics ===")

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

    # טבלת מדדים
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

    print("\n=== METRICS TABLE ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return cm, auc, report_dict, overall_acc, metrics_df


# ---------------------------------------------------------------
# 7. Plot functions (save PNGs + אופציה להציג)
# ---------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(cm, scenario: str, out_dir: str, show: bool = False):
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix – {scenario} model")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([CLASS_NAMES[i] for i in tick_marks], rotation=45)
    ax.set_yticklabels([CLASS_NAMES[i] for i in tick_marks])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"confusion_matrix_{scenario}.png")
    plt.savefig(out_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)
    print(f"✔ Confusion matrix saved to: {out_path}")


def plot_roc_curve(
    all_targets, all_probs, auc, scenario: str, out_dir: str, show: bool = False
):
    ensure_dir(out_dir)
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – {scenario} model")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"roc_{scenario}.png")
    plt.savefig(out_path, dpi=200)

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
    plt.bar(list(CLASS_NAMES.values()), tprs)
    for i, v in enumerate(tprs):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.ylim(0, 1.05)
    plt.ylabel("Recall (TPR)")
    plt.title(f"Per-class TPR – {scenario} model")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"tpr_{scenario}.png")
    plt.savefig(out_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)
    print(f"✔ Per-class TPR bar chart saved to: {out_path}")


# ---------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------
def main():
    args = parse_args()

    # 1) בניית דאטה הערכה
    eval_df, eval_ds = build_eval_subset(
        n_normal_test=args.n_normal_test,
        n_pneumonia_test=args.n_pneumonia_test,
    )

    # אם ביקשו show – מציגים דוגמאות מהסט
    if args.show:
        show_sample_images(eval_df, max_samples=6)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 2) טעינת מודל וצ'קפוינט
    model, device, ckpt_path = load_model_and_ckpt(args.scenario)

    # 3) אינפרנס
    all_targets, all_preds, all_probs = run_inference(model, device, eval_loader)

    # 4) חישוב מדדים
    cm, auc, report_dict, overall_acc, metrics_df = compute_metrics(
        all_targets, all_preds, all_probs, args.scenario, len(eval_df)
    )

    # 5) גרפים לקבצים (PNG) + אופציה להציג בקולאב
    out_dir = os.path.join(args.out_dir, f"xrt_eval_{args.scenario}")
    plot_confusion_matrix(cm, args.scenario, out_dir, show=args.show)
    plot_roc_curve(all_targets, all_probs, auc, args.scenario, out_dir, show=args.show)
    plot_tpr_bars(report_dict, args.scenario, out_dir, show=args.show)

    print("\n✅ Finished XRT evaluation script.")


if __name__ == "__main__":
    main()
