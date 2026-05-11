"""Biomechanical consistency losses for humanoid trajectories."""

from __future__ import annotations

import torch
from torch import nn


class BiomechanicalConsistencyLoss(nn.Module):
    """Physical plausibility regularizer for trajectories [B, T, J]."""

    def __init__(
        self,
        lower_joint_limits: torch.Tensor,
        upper_joint_limits: torch.Tensor,
        acceleration_weight: float,
        joint_limit_weight: float,
        temporal_jitter_weight: float,
    ) -> None:
        super().__init__()
        if lower_joint_limits.shape != upper_joint_limits.shape:
            raise ValueError("Lower and upper joint limits must have identical shape.")
        self.register_buffer("lower_joint_limits", lower_joint_limits.view(1, 1, -1))
        self.register_buffer("upper_joint_limits", upper_joint_limits.view(1, 1, -1))
        self.acceleration_weight = acceleration_weight
        self.joint_limit_weight = joint_limit_weight
        self.temporal_jitter_weight = temporal_jitter_weight

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
        per_sample = (
            self.acceleration_weight * accel
            + self.joint_limit_weight * limit
            + self.temporal_jitter_weight * jitter
        )
        return {
            "joint_limit_loss": limit.mean(),
            "acceleration_loss": accel.mean(),
            "temporal_jitter_loss": jitter.mean(),
            "physical_loss": per_sample.mean(),
            "physical_loss_per_sample": per_sample,
        }
