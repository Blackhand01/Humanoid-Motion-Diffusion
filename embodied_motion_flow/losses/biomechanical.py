"""Biomechanical consistency losses for humanoid trajectories."""

from __future__ import annotations

import torch
from torch import nn

SMPL_NUM_JOINTS = 24
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


def _non_adjacent_pairs(num_joints: int, parents: tuple[int, ...] = SMPL_PARENTS) -> torch.Tensor:
    """Return joint index pairs excluding self and direct kinematic neighbors."""
    pairs: list[tuple[int, int]] = []
    for left in range(num_joints):
        for right in range(left + 1, num_joints):
            left_parent = parents[left] if left < len(parents) else -1
            right_parent = parents[right] if right < len(parents) else -1
            if left_parent == right or right_parent == left:
                continue
            pairs.append((left, right))
    return torch.tensor(pairs, dtype=torch.long)


class SelfCollisionLoss(nn.Module):
    """Heuristic collision penalty for non-adjacent SMPL joint centers.

    Args:
        margin: Minimum allowed pairwise distance in the units of the supplied
            joint-center tensor. For flattened SMPL axis-angle tensors
            ``[B,T,72]`` this acts as a pose-space diagnostic proxy.
    """

    def __init__(self, margin: float = 0.08, num_joints: int = SMPL_NUM_JOINTS) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("Self-collision margin must be positive.")
        self.margin = margin
        self.num_joints = num_joints
        self.register_buffer("pair_indices", _non_adjacent_pairs(num_joints), persistent=False)

    def _as_joint_centers(self, trajectory: torch.Tensor) -> torch.Tensor | None:
        if trajectory.ndim == 4 and trajectory.shape[-2:] == (self.num_joints, 3):
            return trajectory
        if trajectory.ndim == 3 and trajectory.shape[-1] == self.num_joints * 3:
            return trajectory.reshape(trajectory.shape[0], trajectory.shape[1], self.num_joints, 3)
        return None

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Return per-sample self-collision penalty ``[B]``."""
        centers = self._as_joint_centers(trajectory)
        if centers is None:
            if trajectory.ndim < 1:
                raise ValueError("trajectory must include a batch dimension")
            return torch.zeros(trajectory.shape[0], dtype=trajectory.dtype, device=trajectory.device)

        pairs = self.pair_indices.to(centers.device)
        left = centers[:, :, pairs[:, 0], :]
        right = centers[:, :, pairs[:, 1], :]
        distances = torch.linalg.vector_norm(left - right, dim=-1)
        penalty = torch.relu(self.margin - distances).pow(2)
        return penalty.mean(dim=(1, 2))


class BiomechanicalConsistencyLoss(nn.Module):
    """Physical plausibility regularizer for trajectories [B, T, J]."""

    def __init__(
        self,
        lower_joint_limits: torch.Tensor,
        upper_joint_limits: torch.Tensor,
        acceleration_weight: float,
        joint_limit_weight: float,
        temporal_jitter_weight: float,
        self_collision_weight: float = 0.0,
        self_collision_margin: float = 0.08,
    ) -> None:
        super().__init__()
        if lower_joint_limits.shape != upper_joint_limits.shape:
            raise ValueError("Lower and upper joint limits must have identical shape.")
        self.register_buffer("lower_joint_limits", lower_joint_limits.view(1, 1, -1))
        self.register_buffer("upper_joint_limits", upper_joint_limits.view(1, 1, -1))
        self.acceleration_weight = acceleration_weight
        self.joint_limit_weight = joint_limit_weight
        self.temporal_jitter_weight = temporal_jitter_weight
        self.self_collision_weight = self_collision_weight
        self.self_collision_loss = SelfCollisionLoss(margin=self_collision_margin)

    @staticmethod
    def _ensure_sequence_shape(trajectory: torch.Tensor) -> None:
        if trajectory.ndim != 3:
            raise ValueError(f"Expected trajectory shape [B, T, J], got {tuple(trajectory.shape)}")

    def joint_limit_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Per-sample penalty for violating anatomical ranges."""
        self._ensure_sequence_shape(trajectory)
        lower_violation = torch.relu(self.lower_joint_limits - trajectory)
        upper_violation = torch.relu(trajectory - self.upper_joint_limits)
        penalty = (lower_violation + upper_violation).pow(2).mean(dim=(1, 2))
        return penalty

    def acceleration_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Per-sample penalty on second temporal derivative (abrupt acceleration)."""
        self._ensure_sequence_shape(trajectory)
        velocity = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
        return acceleration.pow(2).mean(dim=(1, 2))

    def temporal_jitter_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Per-sample penalty on third temporal derivative (frame-to-frame jitter)."""
        self._ensure_sequence_shape(trajectory)
        velocity = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
        jerk = acceleration[:, 1:, :] - acceleration[:, :-1, :]
        return jerk.abs().mean(dim=(1, 2))

    def forward(self, trajectory: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute weighted physical loss and scalar components."""
        limit = self.joint_limit_penalty(trajectory)
        accel = self.acceleration_penalty(trajectory)
        jitter = self.temporal_jitter_penalty(trajectory)
        collision = self.self_collision_loss(trajectory)
        per_sample = (
            self.acceleration_weight * accel
            + self.joint_limit_weight * limit
            + self.temporal_jitter_weight * jitter
            + self.self_collision_weight * collision
        )
        return {
            "joint_limit_loss": limit.mean(),
            "acceleration_loss": accel.mean(),
            "temporal_jitter_loss": jitter.mean(),
            "self_collision_loss": collision.mean(),
            "physical_loss": per_sample.mean(),
            "physical_loss_per_sample": per_sample,
        }
