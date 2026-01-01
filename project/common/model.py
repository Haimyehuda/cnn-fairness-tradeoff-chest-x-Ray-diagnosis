"""
model.py
========

CNN model definition for Chest X-Ray classification.

This study uses a single, fixed architecture:
- DenseNet-121

The module exposes a minimal factory function for model creation.
"""

import torch.nn as nn
from torchvision.models import densenet121


def get_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create a DenseNet-121 model for binary classification.

    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Whether to use ImageNet-pretrained weights

    Returns:
        torch.nn.Module: Initialized DenseNet-121 model
    """

    model = densenet121(pretrained=pretrained)

    # Replace classifier head to match the task
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model
