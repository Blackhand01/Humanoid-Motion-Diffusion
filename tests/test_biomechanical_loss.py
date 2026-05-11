"""Tests for biomechanical loss components."""

from __future__ import annotations

import torch

from embodied_motion_flow.losses.biomechanical import BiomechanicalConsistencyLoss


def _build_loss() -> BiomechanicalConsistencyLoss:
    lower = torch.full((12,), -1.0)
    upper = torch.full((12,), 1.0)
    return BiomechanicalConsistencyLoss(
        lower_joint_limits=lower,
        upper_joint_limits=upper,
        acceleration_weight=0.2,
        joint_limit_weight=0.3,
        temporal_jitter_weight=0.1,
    )


def test_joint_limit_penalty_detects_violations() -> None:
    loss_fn = _build_loss()
    trajectory = torch.zeros(2, 20, 12)
    trajectory[0, :, 0] = 1.5
    penalties = loss_fn.joint_limit_penalty(trajectory)
    assert penalties[0] > penalties[1]
    assert penalties[1].item() == 0.0


def test_acceleration_penalty_increases_on_abrupt_change() -> None:
    loss_fn = _build_loss()
    smooth = torch.linspace(0.0, 0.5, 24).view(1, 24, 1).repeat(1, 1, 12)
    abrupt = smooth.clone()
    abrupt[:, 12:, :] += 1.5
    assert loss_fn.acceleration_penalty(abrupt).item() > loss_fn.acceleration_penalty(smooth).item()


def test_forward_returns_expected_components() -> None:
    loss_fn = _build_loss()
    trajectory = torch.randn(3, 18, 12) * 0.1
    outputs = loss_fn(trajectory)
    expected = {
        "joint_limit_loss",
        "acceleration_loss",
        "temporal_jitter_loss",
        "physical_loss",
        "physical_loss_per_sample",
    }
    assert expected.issubset(outputs.keys())
    assert outputs["physical_loss_per_sample"].shape == (3,)
