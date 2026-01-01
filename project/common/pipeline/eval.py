# common/pipeline/eval.py
"""
Evaluation utilities for binary classification models.

Computes (numeric):
- Accuracy (overall and per class)
- Confusion matrix (TN/FP/FN/TP)
- Precision / Recall (TPR) / FPR per class
- F1-score per class
- Fairness gaps: ΔTPR (Equal Opportunity), ΔFPR (Equalized Odds)
- Disparate Impact (DI)

Optionally produces (visual):
- Confusion matrix (counts + normalized)
- ROC curve + AUC
- Precision–Recall curve + AP
- Per-class metric bars (Acc/TPR/FPR/F1)

Notes:
- Assumes binary labels: 0=NEG (NORMAL), 1=POS (PNEUMONIA)
- Returns only metrics by default; enable plots/outputs via flags.
"""

import os
import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _savefig(path: str) -> None:
    # Save without blocking Colab output; close to avoid memory leaks
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_confusion(cm: np.ndarray, title: str, out_path: str) -> None:
    # Simple heatmap without seaborn
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["NORMAL(0)", "PNEUMONIA(1)"])
    plt.yticks([0, 1], ["NORMAL(0)", "PNEUMONIA(1)"])

    # Annotate cells
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")

    _savefig(out_path)


def _plot_confusion_norm(cm: np.ndarray, title: str, out_path: str) -> None:
    # Row-normalized confusion matrix (per-true-class)
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
    cmn = cm / row_sums

    plt.figure()
    plt.imshow(cmn, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["NORMAL(0)", "PNEUMONIA(1)"])
    plt.yticks([0, 1], ["NORMAL(0)", "PNEUMONIA(1)"])

    for (i, j), v in np.ndenumerate(cmn):
        plt.text(j, i, f"{v:.2f}", ha="center", va="center")

    _savefig(out_path)


def _plot_roc(
    y_true: np.ndarray, y_score: np.ndarray, title: str, out_path: str
) -> float:
    # ROC uses positive-class scores
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{title} (AUC={auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    _savefig(out_path)
    return float(auc)


def _plot_pr(
    y_true: np.ndarray, y_score: np.ndarray, title: str, out_path: str
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"{title} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    _savefig(out_path)
    return float(ap)


def _plot_class_bars(metrics: dict, title: str, out_path: str) -> None:
    # Compact bar chart for key per-class metrics
    labels = ["Accuracy", "TPR", "FPR", "F1"]
    normal_vals = [
        metrics["acc_normal"],
        metrics["tpr_normal"],
        metrics["fpr_normal"],
        metrics["f1_normal"],
    ]
    pneu_vals = [
        metrics["acc_pneumonia"],
        metrics["tpr_pneumonia"],
        metrics["fpr_pneumonia"],
        metrics["f1_pneumonia"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, normal_vals, width, label="NORMAL(0)")
    plt.bar(x + width / 2, pneu_vals, width, label="PNEUMONIA(1)")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()

    _savefig(out_path)


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    *,
    return_outputs: bool = False,
    make_plots: bool = False,
    plots_dir: str | None = None,
    run_name: str = "eval",
) -> dict:
    """
    Evaluate a trained model on a fixed evaluation set.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        device: CPU or CUDA device

        return_outputs: if True, returns y_true/y_pred/y_score for downstream analysis
        make_plots: if True, saves plots to plots_dir (PNG)
        plots_dir: directory to save plots (required when make_plots=True)
        run_name: file prefix for plots

    Returns:
        dict: metrics dict (and optionally outputs)
    """

    model.eval()

    all_preds = []
    all_labels = []
    all_scores = []  # positive-class probability

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels_t = labels.to(device)

            logits = model(images)

            # Predicted class
            preds = logits.argmax(dim=1)

            # Positive-class probability for ROC/PR
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels_t.detach().cpu().numpy())
            all_scores.append(probs.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds).astype(int)
    y_true = np.concatenate(all_labels).astype(int)
    y_score = np.concatenate(all_scores).astype(float)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # Accuracy
    accuracy = float((y_pred == y_true).mean())

    # Per-class accuracy
    acc_normal = (
        float((y_pred[y_true == 0] == 0).mean()) if (y_true == 0).any() else 0.0
    )
    acc_pneumonia = (
        float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else 0.0
    )

    # TPR / FPR (by class)
    # Note: For NORMAL (class 0), recall = TN / (TN+FP)
    tpr_normal = float(TN / (TN + FP + 1e-12))
    fpr_normal = float(FP / (TN + FP + 1e-12))

    # For PNEUMONIA (class 1), recall = TP / (TP+FN)
    tpr_pneumonia = float(TP / (TP + FN + 1e-12))
    fpr_pneumonia = float(
        FN / (TP + FN + 1e-12)
    )  # (kept consistent with your original code)

    # Fairness gaps
    delta_tpr = float(abs(tpr_normal - tpr_pneumonia))
    delta_fpr = float(abs(fpr_normal - fpr_pneumonia))

    # Disparate Impact (positive prediction rate ratio)
    ppr_normal = (
        float((y_pred[y_true == 0] == 1).mean()) if (y_true == 0).any() else 0.0
    )
    ppr_pneumonia = (
        float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else 0.0
    )
    disparate_impact = (
        float(ppr_pneumonia / (ppr_normal + 1e-12)) if ppr_normal > 0 else 0.0
    )

    # F1-score per class (computed within each class subset)
    f1_normal = (
        float(f1_score(y_true[y_true == 0], y_pred[y_true == 0], zero_division=0))
        if (y_true == 0).any()
        else 0.0
    )
    f1_pneumonia = (
        float(f1_score(y_true[y_true == 1], y_pred[y_true == 1], zero_division=0))
        if (y_true == 1).any()
        else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "acc_normal": acc_normal,
        "acc_pneumonia": acc_pneumonia,
        "tpr_normal": tpr_normal,
        "tpr_pneumonia": tpr_pneumonia,
        "fpr_normal": fpr_normal,
        "fpr_pneumonia": fpr_pneumonia,
        "f1_normal": f1_normal,
        "f1_pneumonia": f1_pneumonia,
        "delta_tpr": delta_tpr,
        "delta_fpr": delta_fpr,
        "disparate_impact": disparate_impact,
        # Extra: include confusion matrix for easy reporting
        "cm": cm,
    }

    # -----------------------------
    # Optional plots (saved as PNG)
    # -----------------------------
    if make_plots:
        assert plots_dir, "plots_dir must be provided when make_plots=True"
        _ensure_dir(plots_dir)

        # Confusion matrices
        _plot_confusion(
            cm,
            f"{run_name} | Confusion Matrix (Counts)",
            os.path.join(plots_dir, f"{run_name}_cm_counts.png"),
        )
        _plot_confusion_norm(
            cm,
            f"{run_name} | Confusion Matrix (Normalized)",
            os.path.join(plots_dir, f"{run_name}_cm_norm.png"),
        )

        # ROC / PR curves
        auc = _plot_roc(
            y_true,
            y_score,
            f"{run_name} | ROC",
            os.path.join(plots_dir, f"{run_name}_roc.png"),
        )
        ap = _plot_pr(
            y_true,
            y_score,
            f"{run_name} | Precision-Recall",
            os.path.join(plots_dir, f"{run_name}_pr.png"),
        )

        # Per-class bars
        _plot_class_bars(
            metrics,
            f"{run_name} | Per-class Metrics",
            os.path.join(plots_dir, f"{run_name}_class_metrics.png"),
        )

        metrics["roc_auc"] = auc
        metrics["pr_ap"] = ap

    # -----------------------------
    # Optional raw outputs
    # -----------------------------
    if return_outputs:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["y_score"] = y_score

    return metrics
