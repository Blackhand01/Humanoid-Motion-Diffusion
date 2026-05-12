"""Evaluation metrics for reconstruction, smoothness, anomalies, and constraints.

The SMPL-facing metrics accept either flattened axis-angle motion tensors
``[B, T, 72]`` or joint-axis tensors ``[B, T, 24, 3]``. Scalar legacy metrics
return Python floats for logging, while the newer research metrics return
per-sample PyTorch tensors by default so they remain batch-ready.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

SMPL_NUM_JOINTS = 24
SMPL_POSE_DIM = 72
SMPL_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)
SMPL_EXTREMITY_JOINTS = (10, 11, 20, 21, 22, 23)


def _as_batched_motion(motion: torch.Tensor) -> torch.Tensor:
    """Return motion as ``[B, T, D]`` for temporal finite differences."""
    if motion.ndim == 3:
        return motion
    if motion.ndim == 4 and motion.shape[-2:] == (SMPL_NUM_JOINTS, 3):
        return motion.reshape(motion.shape[0], motion.shape[1], SMPL_POSE_DIM)
    raise ValueError(f"Expected motion shape [B,T,D] or [B,T,24,3], received {tuple(motion.shape)}")


def _as_smpl_pose(motion: torch.Tensor) -> torch.Tensor:
    """Return flattened or joint-axis SMPL pose as ``[B, T, 24, 3]``."""
    if motion.ndim == 4 and motion.shape[-2:] == (SMPL_NUM_JOINTS, 3):
        return motion
    if motion.ndim == 3 and motion.shape[-1] == SMPL_POSE_DIM:
        return motion.reshape(motion.shape[0], motion.shape[1], SMPL_NUM_JOINTS, 3)
    raise ValueError(f"Expected SMPL pose [B,T,72] or [B,T,24,3], received {tuple(motion.shape)}")


def _reduce_per_sample(values: torch.Tensor, reduce: str) -> torch.Tensor:
    """Reduce per-sample values according to a stable metric API."""
    if reduce == "none":
        return values
    if reduce == "mean":
        return values.mean()
    if reduce == "sum":
        return values.sum()
    raise ValueError(f"Unsupported reduce={reduce!r}; expected 'none', 'mean', or 'sum'.")


def default_smpl_joint_limits(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Return heuristic anatomical SMPL axis-angle limits as flattened ``[72]`` tensors.

    SMPL stores each joint as an axis-angle triplet. These bounds are deliberately
    conservative research diagnostics, not a replacement for full kinematic
    constraints in an articulated body simulator.
    """
    pi = math.pi
    lower = torch.full((SMPL_NUM_JOINTS, 3), -1.2, dtype=dtype, device=device)
    upper = torch.full((SMPL_NUM_JOINTS, 3), 1.2, dtype=dtype, device=device)

    # Global root orientation is not anatomical; allow full rotations.
    lower[0] = torch.tensor([-pi, -pi, -pi], dtype=dtype, device=device)
    upper[0] = torch.tensor([pi, pi, pi], dtype=dtype, device=device)

    for joint in (1, 2):  # hips
        lower[joint] = torch.tensor([-2.1, -0.9, -1.0], dtype=dtype, device=device)
        upper[joint] = torch.tensor([1.3, 0.9, 1.0], dtype=dtype, device=device)
    for joint in (4, 5):  # knees
        lower[joint] = torch.tensor([-0.2, -0.35, -0.45], dtype=dtype, device=device)
        upper[joint] = torch.tensor([2.7, 0.35, 0.45], dtype=dtype, device=device)
    for joint in (7, 8, 10, 11):  # ankles and feet
        lower[joint] = torch.tensor([-0.9, -0.65, -0.65], dtype=dtype, device=device)
        upper[joint] = torch.tensor([0.9, 0.65, 0.65], dtype=dtype, device=device)
    for joint in (3, 6, 9, 12, 15):  # spine, neck, head
        lower[joint] = torch.tensor([-0.75, -0.55, -0.75], dtype=dtype, device=device)
        upper[joint] = torch.tensor([0.75, 0.55, 0.75], dtype=dtype, device=device)
    for joint in (13, 14, 16, 17):  # clavicles and shoulders
        lower[joint] = torch.tensor([-2.5, -1.6, -2.2], dtype=dtype, device=device)
        upper[joint] = torch.tensor([2.5, 1.6, 2.2], dtype=dtype, device=device)
    for joint in (18, 19):  # elbows
        lower[joint] = torch.tensor([-0.25, -0.45, -0.65], dtype=dtype, device=device)
        upper[joint] = torch.tensor([2.6, 0.45, 0.65], dtype=dtype, device=device)
    for joint in (20, 21, 22, 23):  # wrists and hands
        lower[joint] = torch.tensor([-1.2, -0.9, -1.2], dtype=dtype, device=device)
        upper[joint] = torch.tensor([1.2, 0.9, 1.2], dtype=dtype, device=device)

    return lower.reshape(-1), upper.reshape(-1)


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


def temporal_smoothness_index(motion: torch.Tensor, reduce: str = "none") -> torch.Tensor:
    """Temporal Smoothness Index from second-order pose derivatives.

    Args:
        motion: Motion tensor shaped ``[B,T,D]`` or SMPL pose ``[B,T,24,3]``.
        reduce: ``"none"`` returns ``[B]``; ``"mean"`` or ``"sum"`` reduce batch.

    Returns:
        L2 acceleration magnitude averaged over time for each sequence. Lower is
        smoother and large spikes indicate abrupt pose changes.
    """
    x = _as_batched_motion(motion)
    if x.shape[1] < 3:
        return _reduce_per_sample(torch.zeros(x.shape[0], dtype=x.dtype, device=x.device), reduce)
    acceleration = x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]
    per_frame = torch.linalg.vector_norm(acceleration, ord=2, dim=-1)
    return _reduce_per_sample(per_frame.mean(dim=1), reduce)


def joint_limit_violation_rate(
    motion: torch.Tensor,
    lower_limits: torch.Tensor | None = None,
    upper_limits: torch.Tensor | None = None,
    reduce: str = "none",
) -> torch.Tensor:
    """Joint Limit Violation Rate for SMPL 24-joint axis-angle poses.

    A frame-joint is counted as violating if any of its three axis-angle
    components is outside the supplied anatomical range.
    """
    pose = _as_smpl_pose(motion)
    if lower_limits is None or upper_limits is None:
        lower_limits, upper_limits = default_smpl_joint_limits(device=pose.device, dtype=pose.dtype)
    lower = lower_limits.to(device=pose.device, dtype=pose.dtype).reshape(1, 1, SMPL_NUM_JOINTS, 3)
    upper = upper_limits.to(device=pose.device, dtype=pose.dtype).reshape(1, 1, SMPL_NUM_JOINTS, 3)
    violations = ((pose < lower) | (pose > upper)).any(dim=-1).float()
    per_sample = violations.mean(dim=(1, 2))
    return _reduce_per_sample(per_sample, reduce)


def _beat_frames_to_indicator(
    beat_frames: torch.Tensor | list[list[int]] | list[int],
    batch_size: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert sparse beat frame ids or dense indicators to ``[B,T]``."""
    if isinstance(beat_frames, list):
        indicator = torch.zeros((batch_size, num_frames), dtype=dtype, device=device)
        if beat_frames and isinstance(beat_frames[0], int):
            beat_frames = [beat_frames for _ in range(batch_size)]  # type: ignore[list-item]
        for batch_idx, frames in enumerate(beat_frames):
            if batch_idx >= batch_size:
                break
            for frame in frames:
                if 0 <= int(frame) < num_frames:
                    indicator[batch_idx, int(frame)] = 1.0
        return indicator

    beats = beat_frames.to(device=device)
    if beats.ndim == 1:
        indicator = torch.zeros((batch_size, num_frames), dtype=dtype, device=device)
        frame_ids = beats.long()
        frame_ids = frame_ids[(frame_ids >= 0) & (frame_ids < num_frames)]
        indicator[:, frame_ids] = 1.0
        return indicator
    if beats.ndim == 2 and beats.shape == (batch_size, num_frames):
        return beats.to(dtype=dtype)
    if beats.ndim == 2:
        indicator = torch.zeros((batch_size, num_frames), dtype=dtype, device=device)
        for batch_idx in range(min(batch_size, beats.shape[0])):
            valid = beats[batch_idx].long()
            valid = valid[(valid >= 0) & (valid < num_frames)]
            indicator[batch_idx, valid] = 1.0
        return indicator
    raise ValueError("beat_frames must be frame ids [K]/[B,K] or dense indicators [B,T].")


def beat_alignment_score(
    motion: torch.Tensor,
    beat_frames: torch.Tensor | list[list[int]] | list[int],
    extremity_joint_indices: tuple[int, ...] = SMPL_EXTREMITY_JOINTS,
    tolerance_frames: int = 3,
    reduce: str = "none",
) -> torch.Tensor:
    """Beat Alignment Score between motion-energy peaks and audio beats.

    The score is a cosine similarity between normalized extremity acceleration
    energy and a beat impulse train expanded by ``tolerance_frames``. It returns
    values near 1 when motion accents coincide with beats and near 0 when they
    are uncorrelated.
    """
    pose = _as_smpl_pose(motion)
    batch_size, num_frames = pose.shape[:2]
    if num_frames < 3:
        return _reduce_per_sample(torch.zeros(batch_size, dtype=pose.dtype, device=pose.device), reduce)

    acceleration = pose[:, 2:, extremity_joint_indices, :] - 2.0 * pose[:, 1:-1, extremity_joint_indices, :] + pose[:, :-2, extremity_joint_indices, :]
    energy = acceleration.pow(2).sum(dim=(-1, -2)).sqrt()
    energy = energy - energy.mean(dim=1, keepdim=True)
    energy = torch.relu(energy)

    beat_indicator = _beat_frames_to_indicator(
        beat_frames=beat_frames,
        batch_size=batch_size,
        num_frames=num_frames,
        device=pose.device,
        dtype=pose.dtype,
    )[:, 1:-1]
    if tolerance_frames > 0:
        width = 2 * tolerance_frames + 1
        kernel = torch.ones((1, 1, width), dtype=pose.dtype, device=pose.device)
        beat_indicator = torch.nn.functional.conv1d(
            beat_indicator.unsqueeze(1),
            kernel,
            padding=tolerance_frames,
        ).squeeze(1).clamp(max=1.0)

    numerator = (energy * beat_indicator).sum(dim=1)
    denominator = torch.linalg.vector_norm(energy, dim=1) * torch.linalg.vector_norm(beat_indicator, dim=1)
    score = numerator / denominator.clamp_min(1e-8)
    return _reduce_per_sample(score.clamp(0.0, 1.0), reduce)


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
