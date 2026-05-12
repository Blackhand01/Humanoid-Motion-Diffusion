"""Exponential moving average utilities for stable diffusion inference."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintain shadow weights for model parameters.

    The class stores detached CPU/GPU tensors with the same device as the source
    parameters. It is intentionally small and checkpoint-friendly.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @property
    def enabled(self) -> bool:
        """Return whether EMA updates should be applied."""
        return self.decay > 0.0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights from the current model parameters."""
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            value = param.detach()
            if name not in self.shadow:
                self.shadow[name] = value.clone()
            else:
                self.shadow[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        """Return checkpointable EMA state."""
        return {"decay": self.decay, "shadow": deepcopy(self.shadow)}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore EMA state from a checkpoint."""
        self.decay = float(state_dict.get("decay", self.decay))
        raw_shadow = state_dict.get("shadow", {})
        if not isinstance(raw_shadow, dict):
            raise ValueError("EMA state_dict['shadow'] must be a dictionary")
        self.shadow = {str(name): tensor.detach().clone() for name, tensor in raw_shadow.items()}

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap model parameters with EMA shadow weights."""
        if not self.enabled:
            yield
            return
        backup: dict[str, torch.Tensor] = {}
        try:
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
            yield
        finally:
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
