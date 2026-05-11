"""Evaluation metrics for reconstruction, smoothness, anomalies, and constraints."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def mse_reconstruction(reference: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Mean squared reconstruction error."""
    if reference.shape != reconstructed.shape:
        raise ValueError("reference and reconstructed tensors must match")
    return float(torch.mean((reference - reconstructed) ** 2).item())


def temporal_smoothness(trajectory: torch.Tensor) -> float:
    """Smoothness as mean squared velocity (lower is smoother)."""
    if trajectory.ndim != 3:
        raise ValueError("trajectory must be [B, T, J]")
    velocity = trajectory[:, 1:, :] - trajectory[:, :-1, :]
    return float(torch.mean(velocity**2).item())


def physical_constraint_violations(
    trajectory: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> float:
    """Fraction of joint-time samples violating explicit ranges."""
    if trajectory.ndim != 3:
        raise ValueError("trajectory must be [B, T, J]")
    lower = lower_limits.view(1, 1, -1).to(trajectory.device)
    upper = upper_limits.view(1, 1, -1).to(trajectory.device)
    violations = (trajectory < lower) | (trajectory > upper)
    return float(torch.mean(violations.float()).item())


def anomaly_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC for anomaly classification."""
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be 1D arrays")
    unique = np.unique(labels)
    if unique.shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))
