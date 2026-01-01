"""
utils.py
========

Utility functions for reproducible experiments and device handling.

This module contains only infrastructure-level helpers and does not
include any experiment-specific logic.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for full reproducibility.

    This affects:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the computation device.

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
