"""Evaluation runner for reconstruction, smoothness, AUROC, and constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from embodied_motion_flow.config import ExperimentConfig
from embodied_motion_flow.evaluation.anomaly import anomaly_score_reconstruction_plus_biomechanical
from embodied_motion_flow.evaluation.metrics import (
    anomaly_auroc,
    beat_alignment_score,
    default_smpl_joint_limits,
    joint_limit_violation_rate,
    physical_constraint_violations,
    temporal_smoothness,
    temporal_smoothness_index,
)
from embodied_motion_flow.losses.biomechanical import BiomechanicalConsistencyLoss
from embodied_motion_flow.models.cross_attention_diffusion import AudioConditionedTransformerDenoiser
from embodied_motion_flow.models.diffusion import DDPMScheduler


def _guided_model_fn(model: nn.Module, audio_context: torch.Tensor, guidance_scale: float):
    """Return model callable with classifier-free guidance when supported."""

    def model_fn(sample: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        if not isinstance(model, AudioConditionedTransformerDenoiser):
            return model(sample, step)
        cond = model(sample, step, audio_context)
        scale = float(guidance_scale)
        if scale == 1.0:
            return cond
        uncond = model(sample, step, torch.zeros_like(audio_context))
        return uncond + scale * (cond - uncond)

    return model_fn


@dataclass(frozen=True)
class EvaluationOutputs:
    """Evaluation metrics and exemplar trajectories for visualization."""

    metrics: dict[str, float]
    noisy_example: np.ndarray
    denoised_example: np.ndarray
    reference_example: np.ndarray | None = None
    beat_indicator_example: np.ndarray | None = None
    source_id: str = "hero_validation"


@torch.no_grad()
def evaluate_model(
    config: ExperimentConfig,
    model: nn.Module,
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
    tsi_total = 0.0
    jlvr_total = 0.0
    bas_total = 0.0
    bas_batches = 0
    reference_bas_total = 0.0
    reference_bas_batches = 0
    collision_total = 0.0
    violation_total = 0.0
    batches = 0

    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    noisy_example: np.ndarray | None = None
    denoised_example: np.ndarray | None = None
    reference_example: np.ndarray | None = None
    beat_indicator_example: np.ndarray | None = None
    source_id = "hero_validation"
    best_visual_score = -float("inf")

    smpl_lower_limits, smpl_upper_limits = default_smpl_joint_limits(device=device)
    if config.data.representation.startswith("smpl") and config.data.input_dim == 72:
        lower_limits, upper_limits = smpl_lower_limits, smpl_upper_limits
    else:
        lower_limits = torch.tensor(config.data.joint_limits.lower, dtype=torch.float32, device=device)
        upper_limits = torch.tensor(config.data.joint_limits.upper, dtype=torch.float32, device=device)

    for batch in dataloader:
        x0 = batch["motion"].to(device)
        raw_audio = batch.get("audio_context")
        if isinstance(raw_audio, torch.Tensor):
            audio_context = raw_audio.to(device)
        else:
            audio_context = torch.zeros(
                (x0.shape[0], x0.shape[1], config.model.audio_dim),
                dtype=x0.dtype,
                device=device,
            )
        model_fn = _guided_model_fn(model, audio_context, config.inference.guidance_scale)
        labels = batch["anomaly_label"].detach().cpu().numpy()
        modes = list(batch.get("mode", ["unknown"] * x0.shape[0]))
        source_paths = batch.get("source_path")
        raw_beat_indicator = batch.get("beat_indicator")
        noisy_xt, reconstructed = scheduler.reconstruct(model=model_fn, x0=x0, start_timestep=start_step)

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
                reference_example = x0[sample_idx].detach().cpu().numpy()
                if isinstance(raw_beat_indicator, torch.Tensor):
                    beat_indicator_example = raw_beat_indicator[sample_idx].detach().cpu().numpy()
                if isinstance(source_paths, list) and sample_idx < len(source_paths):
                    source_id = str(source_paths[sample_idx])

        mse_total += float(F.mse_loss(reconstructed, x0).item())
        smooth_total += temporal_smoothness(reconstructed)
        tsi_total += float(temporal_smoothness_index(reconstructed, reduce="mean").item())
        if reconstructed.shape[-1] == 72:
            jlvr_total += float(
                joint_limit_violation_rate(reconstructed, smpl_lower_limits, smpl_upper_limits, reduce="mean").item()
            )
            raw_beats = batch.get("beat_frames", batch.get("beat_indicator"))
            if isinstance(raw_beats, torch.Tensor):
                bas_total += float(beat_alignment_score(reconstructed, raw_beats.to(device), reduce="mean").item())
                bas_batches += 1
                reference_bas_total += float(beat_alignment_score(x0, raw_beats.to(device), reduce="mean").item())
                reference_bas_batches += 1
        collision_total += float(biomechanical_loss.self_collision_loss(reconstructed).mean().item())
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
    generated_bas = bas_total / bas_batches if bas_batches else float("nan")
    reference_bas = reference_bas_total / reference_bas_batches if reference_bas_batches else float("nan")
    metrics = {
        "mse_reconstruction": mse_total / max(batches, 1),
        "temporal_smoothness": smooth_total / max(batches, 1),
        "temporal_smoothness_index": tsi_total / max(batches, 1),
        "joint_limit_violation_rate": jlvr_total / max(batches, 1),
        "beat_alignment_score": generated_bas,
        "reference_beat_alignment_score": reference_bas,
        "beat_alignment_gap": reference_bas - generated_bas,
        "self_collision": collision_total / max(batches, 1),
        "auroc_anomaly": anomaly_auroc(labels_np, scores_np),
        "physical_constraint_violations": violation_total / max(batches, 1),
    }

    if noisy_example is None or denoised_example is None:
        raise RuntimeError("No evaluation batches were produced.")
    return EvaluationOutputs(
        metrics=metrics,
        noisy_example=noisy_example,
        denoised_example=denoised_example,
        reference_example=reference_example,
        beat_indicator_example=beat_indicator_example,
        source_id=source_id,
    )
