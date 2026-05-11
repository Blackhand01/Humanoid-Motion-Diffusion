"""Deterministic seeding utilities for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool, benchmark_cudnn: bool) -> None:
    """Set deterministic seeds and torch backend flags for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic_torch
    torch.backends.cudnn.benchmark = benchmark_cudnn
