"""
model.py
========

CNN model definition for Chest X-Ray classification.

This study uses a single, fixed architecture:
- DenseNet-121 (torchvision implementation)

Key design choices:
-------------------
1. Transfer Learning:
   - Uses ImageNet-pretrained weights to leverage generic visual features.

2. Grayscale Input Adaptation:
   - Chest X-Ray images are single-channel (Grayscale).
   - The first convolution layer (conv0) is modified to accept 1 channel
     instead of the original 3 (RGB).
   - Initialization strategy:
     The new weights are initialized by averaging the original RGB weights,
     preserving as much pretrained information as possible.

3. Binary Classification Head:
   - The final classifier layer is replaced with a linear layer
     matching the number of task-specific classes (default: 2).
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121


def get_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create a DenseNet-121 model adapted for grayscale chest X-ray classification.

    Args:
        num_classes (int):
            Number of output classes.
            Default: 2 (Normal / Pneumonia)

        pretrained (bool):
            Whether to initialize the model with ImageNet-pretrained weights.
            Default: True

    Returns:
        nn.Module:
            Initialized DenseNet-121 model with:
            - 1-channel input
            - task-specific classifier head
    """

    # ---------------------------------------------------------
    # Load base DenseNet-121 architecture
    # ---------------------------------------------------------
    model = densenet121(pretrained=pretrained)

    # ---------------------------------------------------------
    # Adapt first convolution layer for Grayscale input
    # Original conv0:
    #   in_channels = 3 (RGB)
    #   out_channels = 64
    #   kernel_size = 7x7
    # ---------------------------------------------------------
    old_conv = model.features.conv0

    new_conv = nn.Conv2d(
        in_channels=1,  # Grayscale input
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    if pretrained:
        # Initialize new conv weights by averaging RGB channels
        # Shape before: (64, 3, 7, 7)
        # Shape after:  (64, 1, 7, 7)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))

    model.features.conv0 = new_conv

    # ---------------------------------------------------------
    # Replace classifier head for task-specific output
    # ---------------------------------------------------------
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model
