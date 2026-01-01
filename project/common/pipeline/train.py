"""
train.py
========

Training utilities for binary classification models.

Responsibilities:
- Train a model for a fixed number of epochs
- Return the trained model
"""

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

    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
