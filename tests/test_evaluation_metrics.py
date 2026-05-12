"""Tests for quantitative motion evaluation metrics."""

from __future__ import annotations

import torch

from embodied_motion_flow.evaluation.metrics import (
    beat_alignment_score,
    default_smpl_joint_limits,
    joint_limit_violation_rate,
    temporal_smoothness_index,
)


def test_temporal_smoothness_index_increases_for_abrupt_motion() -> None:
    smooth = torch.linspace(0.0, 1.0, 20).view(1, 20, 1).repeat(1, 1, 72)
    abrupt = smooth.clone()
    abrupt[:, 10:, :] += 2.0
    assert temporal_smoothness_index(abrupt).item() > temporal_smoothness_index(smooth).item()


def test_joint_limit_violation_rate_detects_smpl_axis_angle_violations() -> None:
    lower, upper = default_smpl_joint_limits()
    motion = torch.zeros(2, 8, 72)
    motion[0, :, 4 * 3] = upper[4 * 3] + 1.0
    jlvr = joint_limit_violation_rate(motion, lower, upper)
    assert jlvr[0] > 0.0
    assert jlvr[1].item() == 0.0


def test_beat_alignment_score_prefers_accents_on_beat() -> None:
    motion_on = torch.zeros(1, 16, 72)
    motion_off = torch.zeros(1, 16, 72)
    wrist_axis = 20 * 3
    motion_on[:, 8, wrist_axis] = 4.0
    motion_off[:, 12, wrist_axis] = 4.0
    beat_frames = torch.tensor([[8]])
    on_score = beat_alignment_score(motion_on, beat_frames, tolerance_frames=1).item()
    off_score = beat_alignment_score(motion_off, beat_frames, tolerance_frames=1).item()
    assert on_score > off_score
