"""Tests for SMPL 3D rendering math."""

from __future__ import annotations

import numpy as np

from embodied_motion_flow.visualization.smpl_render import smpl_axis_angle_to_joints, smpl_motion_to_joints


def test_smpl_axis_angle_to_joints_shape() -> None:
    pose = np.zeros((72,), dtype=np.float32)
    joints = smpl_axis_angle_to_joints(pose)
    assert joints.shape == (24, 3)
    assert np.isfinite(joints).all()


def test_smpl_motion_to_joints_shape() -> None:
    motion = np.zeros((8, 72), dtype=np.float32)
    joints = smpl_motion_to_joints(motion)
    assert joints.shape == (8, 24, 3)
    assert np.isfinite(joints).all()
