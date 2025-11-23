# utils.py
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools


# ============================================================
# reproducibility + device
# ============================================================


def set_seed(seed=42):
    """קיבוע זרעים לשחזור תוצאות."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # אופציונלי: יותר דטרמיניסטי על חשבון ביצועים
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """מחזיר cuda אם זמין אחרת cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Confusion Matrix
# ============================================================


def plot_cm(cm, labels=None, title="Confusion Matrix"):
    """
    מצייר מטריצת בלבול.

    cm      : numpy array בגודל (C, C)
    labels  : רשימת תוויות לצירים, באורך C.
              אם None -> משתמש באינדקסים [0..C-1]
    title   : כותרת לגרף
    """
    cm = np.array(cm)

    num_classes = cm.shape[0]
    if labels is None:
        # ברירת מחדל בינארית – מתאים ל-CheXpert
        if num_classes == 2:
            labels = ["NORMAL", "PNEUMONIA"]
        else:
            labels = [str(i) for i in range(num_classes)]

    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() != 0 else 0.5

    # כתיבה על כל תא במטריצה
    for i, j in itertools.product(range(num_classes), range(num_classes)):
        plt.text(
            j,
            i,
            int(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# ============================================================
# Show sample images per class
# ============================================================


def show_samples(dataset, n_per_class=2, class_names=None):
    """
    מציג מספר דוגמאות (n_per_class) מתוך הדאטה לכל מחלקה בינארית (0/1).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        אובייקט Dataset שמחזיר (image_tensor, label).
        עבור CheXpert זה יהיה X-ray עם label 0/1.
    n_per_class : int
        כמות תמונות להציג לכל מחלקה.
    class_names : dict או None
        מילון תוויות {0: "NORMAL", 1: "PNEUMONIA"}.
        אם None -> ישתמש ב-Class 0 / Class 1.

    Notes
    -----
    * תומך ב-in_channels=1 (grayscale) או 3 (RGB).
    * מנרמל חזרה את התמונה לפי (mean=0.5, std=0.5), כמו ב-dataset.py.
    """

    # איסוף תמונות לכל מחלקה
    images = {0: [], 1: []}
    counts = {0: 0, 1: 0}

    for img, label in dataset:
        lbl = int(label)
        if lbl not in images:
            # אם במקרה יש תוויות אחרות – מדלגים
            continue

        if counts[lbl] < n_per_class:
            images[lbl].append(img)
            counts[lbl] += 1

        # עצירה אם אספנו מספיק ל־0 ול־1
        if all(counts[c] >= n_per_class for c in [0, 1]):
            break

    # אם אין מספיק דגימות – לא ננסה לצייר
    if counts[0] == 0 or counts[1] == 0:
        print("Not enough samples in dataset for classes 0 and 1.")
        return

    rows = 2
    cols = n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # אם n_per_class == 1, axes יהיה וקטור – ניישר לצורה אחידה
    if n_per_class == 1:
        axes = np.array(axes).reshape(rows, 1)

    for cls in [0, 1]:
        for j in range(n_per_class):
            ax = axes[cls, j]
            img = images[cls][j]

            # unnormalize לפי (mean=0.5, std=0.5)
            img = img * 0.5 + 0.5  # [0,1]

            img_np = img.numpy()

            if img_np.shape[0] == 1:
                # grayscale (1, H, W)
                ax.imshow(img_np[0], cmap="gray")
            else:
                # RGB (3, H, W)
                ax.imshow(img_np.transpose(1, 2, 0))

            ax.axis("off")
            title = (
                class_names[cls]
                if class_names and cls in class_names
                else f"Class {cls}"
            )
            ax.set_title(title)

    plt.tight_layout()
    plt.show()
