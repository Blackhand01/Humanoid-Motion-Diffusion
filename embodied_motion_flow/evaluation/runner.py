"""Evaluation runner for reconstruction, smoothness, AUROC, and constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from embodied_motion_flow.config import ExperimentConfig
from embodied_motion_flow.evaluation.anomaly import anomaly_score_reconstruction_plus_biomechanical
from embodied_motion_flow.evaluation.metrics import (
    anomaly_auroc,
    physical_constraint_violations,
    temporal_smoothness,
)
from embodied_motion_flow.losses.biomechanical import BiomechanicalConsistencyLoss
from embodied_motion_flow.models.diffusion import DDPMScheduler
from embodied_motion_flow.models.transformer_denoiser import TemporalTransformerDenoiser


@dataclass(frozen=True)
class EvaluationOutputs:
    """Evaluation metrics and exemplar trajectories for visualization."""

    metrics: dict[str, float]
    noisy_example: np.ndarray
    denoised_example: np.ndarray


@torch.no_grad()
def evaluate_model(
    config: ExperimentConfig,
    model: TemporalTransformerDenoiser,
    scheduler: DDPMScheduler,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    biomechanical_loss: BiomechanicalConsistencyLoss,
    device: torch.device,
) -> EvaluationOutputs:
    """Evaluate model over a dataloader."""
    model.eval()
    start_step = min(config.evaluation.reconstruction_steps, scheduler.timesteps - 1)

    mse_total = 0.0
    smooth_total = 0.0
    violation_total = 0.0
    batches = 0

    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    noisy_example: np.ndarray | None = None
    denoised_example: np.ndarray | None = None
    best_visual_score = -float("inf")

    lower_limits = torch.tensor(config.data.joint_limits.lower, dtype=torch.float32, device=device)
    upper_limits = torch.tensor(config.data.joint_limits.upper, dtype=torch.float32, device=device)

    for batch in dataloader:
        x0 = batch["motion"].to(device)
        labels = batch["anomaly_label"].detach().cpu().numpy()
        modes = list(batch.get("mode", ["unknown"] * x0.shape[0]))
        noisy_xt, reconstructed = scheduler.reconstruct(model=model, x0=x0, start_timestep=start_step)

        velocity = x0[:, 1:, :] - x0[:, :-1, :]
        motion_energy = velocity.pow(2).mean(dim=(1, 2)).detach().cpu().numpy()
        for sample_idx, energy in enumerate(motion_energy):
            is_clean = labels[sample_idx] == 0
            is_walking = modes[sample_idx] == "walking"
            visual_score = float(energy) + (10.0 if is_walking else 0.0) + (1.0 if is_clean else 0.0)
            if visual_score > best_visual_score:
                best_visual_score = visual_score
                noisy_example = noisy_xt[sample_idx].detach().cpu().numpy()
                denoised_example = reconstructed[sample_idx].detach().cpu().numpy()

        mse_total += float(F.mse_loss(reconstructed, x0).item())
        smooth_total += temporal_smoothness(reconstructed)
        violation_total += physical_constraint_violations(reconstructed, lower_limits, upper_limits)

        score = anomaly_score_reconstruction_plus_biomechanical(
            reference=x0,
            reconstructed=reconstructed,
            biomechanical_loss=biomechanical_loss,
        )
        all_labels.append(labels)
        all_scores.append(score.detach().cpu().numpy())
        batches += 1

    labels_np = np.concatenate(all_labels, axis=0)
    scores_np = np.concatenate(all_scores, axis=0)
    metrics = {
        "mse_reconstruction": mse_total / max(batches, 1),
        "temporal_smoothness": smooth_total / max(batches, 1),
        "auroc_anomaly": anomaly_auroc(labels_np, scores_np),
        "physical_constraint_violations": violation_total / max(batches, 1),
    }

    if noisy_example is None or denoised_example is None:
        raise RuntimeError("No evaluation batches were produced.")
    return EvaluationOutputs(metrics=metrics, noisy_example=noisy_example, denoised_example=denoised_example)
