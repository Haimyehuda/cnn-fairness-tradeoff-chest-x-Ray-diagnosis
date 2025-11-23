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
