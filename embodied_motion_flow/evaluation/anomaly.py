"""Anomaly scoring functions for trajectory reconstruction and biomechanics."""

from __future__ import annotations

import torch

from embodied_motion_flow.losses.biomechanical import BiomechanicalConsistencyLoss


def reconstruction_error_score(reference: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Per-sample reconstruction error score."""
    if reference.shape != reconstructed.shape:
        raise ValueError("reference and reconstructed must have identical shape")
    return (reference - reconstructed).pow(2).mean(dim=(1, 2))


def anomaly_score_reconstruction_plus_biomechanical(
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
    biomechanical_loss: BiomechanicalConsistencyLoss,
    biomechanical_weight: float = 0.5,
) -> torch.Tensor:
    """Combine reconstruction and biomechanical inconsistency for anomaly scoring."""
    rec_score = reconstruction_error_score(reference, reconstructed)
    biomech = biomechanical_loss(reconstructed)["physical_loss_per_sample"]
    return rec_score + biomechanical_weight * biomech
