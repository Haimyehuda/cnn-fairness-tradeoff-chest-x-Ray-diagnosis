# eval.py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def eval_model(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())

    preds, labels = np.array(preds), np.array(labels)
    cm = confusion_matrix(labels, preds)

    overall = (preds == labels).mean() * 100
    cat_acc = (preds[labels == 0] == 0).mean() * 100
    dog_acc = (preds[labels == 1] == 1).mean() * 100

    return overall, cat_acc, dog_acc, cm
