"""Learning-rate schedule helpers."""

from __future__ import annotations

import math

import torch


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a linear-warmup plus cosine-decay LambdaLR scheduler."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    warmup = min(warmup_steps, total_steps)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(float(step + 1) / float(warmup), 1e-8)
        progress_denominator = max(total_steps - warmup, 1)
        progress = min(max((step - warmup) / progress_denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
