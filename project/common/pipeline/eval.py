"""
eval.py
=======

Evaluation utilities for binary classification models.

Computes:
- Accuracy (overall and per group)
- F1-score per group
- Equalized Odds (TPR/FPR gaps)
- Disparate Impact (DI)
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """
    Evaluate a trained model on a fixed evaluation set.

    Args:
        model (nn.Module): Trained model
        dataloader: DataLoader for evaluation data
        device (torch.device): CPU or CUDA device

    Returns:
        dict: Dictionary of evaluation and fairness metrics
    """

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # Accuracy
    accuracy = (y_pred == y_true).mean()

    acc_normal = (y_pred[y_true == 0] == 0).mean()
    acc_pneumonia = (y_pred[y_true == 1] == 1).mean()

    # TPR / FPR
    tpr_normal = TN / (TN + FP + 1e-12)
    fpr_normal = FP / (TN + FP + 1e-12)

    tpr_pneumonia = TP / (TP + FN + 1e-12)
    fpr_pneumonia = FN / (TP + FN + 1e-12)

    # Fairness metrics
    delta_tpr = abs(tpr_normal - tpr_pneumonia)
    delta_fpr = abs(fpr_normal - fpr_pneumonia)

    # Disparate Impact (positive prediction rate ratio)
    ppr_normal = (y_pred[y_true == 0] == 1).mean()
    ppr_pneumonia = (y_pred[y_true == 1] == 1).mean()

    disparate_impact = ppr_pneumonia / (ppr_normal + 1e-12) if ppr_normal > 0 else 0.0

    # F1-score per group
    f1_normal = f1_score(y_true[y_true == 0], y_pred[y_true == 0], zero_division=0)
    f1_pneumonia = f1_score(y_true[y_true == 1], y_pred[y_true == 1], zero_division=0)

    return {
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
    }
