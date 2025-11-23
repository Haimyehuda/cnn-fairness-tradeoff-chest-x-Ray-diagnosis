import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def eval_model(model, loader, device):
    """
    Evaluate binary classification model.
    Returns:
        overall_acc
        class0_acc
        class1_acc
        metrics: dict with TPR/FPR gaps (fairness)
        confusion_matrix
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # confusion matrix (rows = true, cols = pred)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # basic metrics
    overall_acc = (preds == labels).mean() * 100

    class0_acc = (preds[labels == 0] == 0).mean() * 100
    class1_acc = (preds[labels == 1] == 1).mean() * 100

    # fairness metrics (per-group TPR / FPR)
    # cm:
    #  [[TN, FP],
    #   [FN, TP]]

    TN, FP, FN, TP = cm.ravel()

    # per-class TPR/FPR
    TPR_class0 = TN / (TN + FP + 1e-12)
    FPR_class0 = FP / (TN + FP + 1e-12)

    TPR_class1 = TP / (TP + FN + 1e-12)
    FPR_class1 = FN / (TP + FN + 1e-12)

    metrics = {
        "TPR_gap": abs(TPR_class0 - TPR_class1),
        "FPR_gap": abs(FPR_class0 - FPR_class1),
        "TPR_class0": TPR_class0,
        "TPR_class1": TPR_class1,
        "FPR_class0": FPR_class0,
        "FPR_class1": FPR_class1,
    }

    return overall_acc, class0_acc, class1_acc, metrics, cm
