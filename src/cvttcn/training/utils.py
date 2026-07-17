"""Reproducibility and device helpers for training."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and torch (CPU and CUDA) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Map a device string to a torch.device.

    ``"auto"`` picks CUDA when available and falls back to CPU; ``"cuda"`` and
    ``"cpu"`` are honoured as given.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
