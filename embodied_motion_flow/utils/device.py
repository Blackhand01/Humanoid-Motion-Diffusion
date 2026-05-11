"""Device selection utilities for CPU, CUDA, and Apple MPS."""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Return the torch device according to user preference and availability."""
    preference = preference.lower()
    if preference not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device preference: {preference}")

    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "mps":
        has_mps = getattr(torch.backends, "mps", None)
        if has_mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = getattr(torch.backends, "mps", None)
    if has_mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
