"""
common/pipeline/train.py

Training utilities for CNN-based binary classification.

This module implements a deterministic training routine with a fixed
number of epochs and static optimization settings, designed for
controlled experimental comparisons.

Responsibilities:
- Train a CNN model under fixed training conditions
- Provide transparent epoch-level progress logging
- Support reproducible experiments without checkpoint persistence
"""

import time
import torch
from torch import nn
from torch.optim import Adam


def train_model(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
) -> None:
    """
    Train a model using cross-entropy loss.

    Args:
        model (nn.Module): Model to train
        train_loader: DataLoader for training data
        device (torch.device): CPU or CUDA device
        epochs (int): Number of training epochs
        lr (float): Learning rate
    """

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = running_loss / max(1, len(train_loader))

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"| avg loss: {avg_loss:.4f} "
            f"| time: {epoch_time:.1f}s"
        )
