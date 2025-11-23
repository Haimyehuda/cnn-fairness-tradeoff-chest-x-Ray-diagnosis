# utils.py
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_cm(cm, labels=("cat", "dog")):
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(2), range(2)):
        plt.text(
            j, i, cm[i, j], ha="center", color="white" if cm[i, j] > thresh else "black"
        )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# ============================================================
# NEW: Show sample X-ray images per class
# ============================================================


def show_samples(dataset, n_per_class=2, class_names=None):
    """
    מציג מספר דוגמאות (n_per_class) מתוך הדאטה לכל מחלקה.

    dataset:   אובייקט Dataset שמחזיר (image_tensor, label)
    n_per_class: כמות תמונות להציג לכל מחלקה
    class_names: מילון תוויות {0: "NORMAL", 1: "PNEUMONIA"}

    * תומך ב-in_channels=1 (grayscale) או 3 (RGB)
    * מנרמל חזרה את התמונה (0.5 * std + mean)
    """

    # איסוף תמונות
    images = {0: [], 1: []}
    counts = {0: 0, 1: 0}

    for img, label in dataset:
        lbl = int(label)
        if counts[lbl] < n_per_class:
            images[lbl].append(img)
            counts[lbl] += 1
        if all(counts[c] >= n_per_class for c in [0, 1]):
            break

    # ציור
    rows = 2
    cols = n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for cls in [0, 1]:
        for j in range(n_per_class):
            ax = axes[cls, j]
            img = images[cls][j]

            # unnormalize לפי (mean=0.5, std=0.5) כמו ב-dataset.py
            img = img * 0.5 + 0.5

            img_np = img.numpy()

            if img_np.shape[0] == 1:
                # grayscale
                ax.imshow(img_np[0], cmap="gray")
            else:
                # RGB
                ax.imshow(img_np.transpose(1, 2, 0))

            ax.axis("off")
            title = class_names[cls] if class_names else f"Class {cls}"
            ax.set_title(title)

    plt.tight_layout()
    plt.show()
