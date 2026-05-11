"""Tests for AIST++ SMPL loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from embodied_motion_flow.config import AISTConfig
from embodied_motion_flow.data.aist_loader import AISTMotionDataset, extract_smpl_poses


def _aist_config(root: Path) -> AISTConfig:
    return AISTConfig(
        root_dir=str(root),
        file_glob="**/*",
        pose_keys=["smpl_poses"],
        source_fps=60.0,
        target_fps=30.0,
        clip_stride=10,
        train_split=0.8,
        val_split=0.1,
        max_files_per_split=None,
    )


def test_extract_smpl_poses_accepts_flat_and_joint_shapes() -> None:
    flat = np.zeros((12, 72), dtype=np.float32)
    joint = np.zeros((12, 24, 3), dtype=np.float32)
    assert extract_smpl_poses({"smpl_poses": flat}, ["smpl_poses"]).shape == (12, 72)
    assert extract_smpl_poses({"smpl_poses": joint}, ["smpl_poses"]).shape == (12, 72)


def test_aist_motion_dataset_segments_npz_file(tmp_path: Path) -> None:
    poses = np.random.default_rng(0).normal(0.0, 0.1, size=(80, 72)).astype(np.float32)
    motion_path = tmp_path / "sample_motion.npz"
    np.savez_compressed(motion_path, smpl_poses=poses)

    dataset = AISTMotionDataset(files=[motion_path], aist_cfg=_aist_config(tmp_path), sequence_length=20)
    sample = dataset[0]

    assert len(dataset) > 0
    assert sample["motion"].shape == (20, 72)
    assert sample["anomaly_label"].item() == 0
    assert sample["mode"] == "aistpp_smpl"
