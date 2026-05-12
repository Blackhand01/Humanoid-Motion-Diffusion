"""Tests for AIST++ SMPL loading."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from embodied_motion_flow.config import AISTConfig, AudioConfig
from embodied_motion_flow.data.aist_loader import AISTMotionDataset, extract_smpl_poses, select_split_motion_files


def _write_sine_wav(path: Path, sample_rate: int = 22050, seconds: float = 1.0) -> None:
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _aist_config(root: Path) -> AISTConfig:
    return AISTConfig(
        root_dir=str(root),
        split_dir=None,
        ignore_list_path=None,
        file_glob="**/*",
        pose_keys=["smpl_poses"],
        source_fps=60.0,
        target_fps=30.0,
        clip_stride=10,
        train_split=0.8,
        val_split=0.1,
        max_files_per_split=None,
        toy_mode=False,
        toy_max_files_per_split=10,
    )


def test_extract_smpl_poses_accepts_flat_and_joint_shapes() -> None:
    flat = np.zeros((12, 72), dtype=np.float32)
    joint = np.zeros((12, 24, 3), dtype=np.float32)
    assert extract_smpl_poses({"smpl_poses": flat}, ["smpl_poses"]).shape == (12, 72)
    assert extract_smpl_poses({"smpl_poses": joint}, ["smpl_poses"]).shape == (12, 72)


def test_aist_motion_dataset_segments_npz_file(tmp_path: Path) -> None:
    poses = np.random.default_rng(0).normal(0.0, 0.1, size=(80, 72)).astype(np.float32)
    trans = np.ones((80, 3), dtype=np.float32)
    motion_path = tmp_path / "sample_motion.npz"
    np.savez_compressed(motion_path, smpl_poses=poses, smpl_trans=trans, smpl_scaling=np.array([2.0], dtype=np.float32))

    dataset = AISTMotionDataset(files=[motion_path], aist_cfg=_aist_config(tmp_path), sequence_length=20)
    sample = dataset[0]

    assert len(dataset) > 0
    assert sample["motion"].shape == (20, 72)
    assert sample["smpl_poses"].shape == (20, 24, 3)
    assert sample["smpl_trans"].shape == (20, 3)
    assert sample["smpl_trans"][0, 0].item() == 2.0
    assert sample["smpl_scaling"].item() == 2.0
    assert sample["has_audio"].item() is False
    assert sample["anomaly_label"].item() == 0
    assert sample["mode"] == "aistpp_smpl"


def test_aist_motion_dataset_loads_sequence_audio(tmp_path: Path) -> None:
    poses = np.random.default_rng(0).normal(0.0, 0.1, size=(80, 72)).astype(np.float32)
    trans = np.ones((80, 3), dtype=np.float32)
    motion_path = tmp_path / "gBR_sBM_cAll_d04_mBR0_ch08.npz"
    np.savez_compressed(motion_path, smpl_poses=poses, smpl_trans=trans, smpl_scaling=np.array([1.0], dtype=np.float32))
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _write_sine_wav(audio_root / "gBR_sBM_cAll_d04_mBR0_ch08.wav", seconds=2.0)
    audio_cfg = AudioConfig(
        enabled=True,
        root_dir=str(audio_root),
        sample_rate=22050,
        hop_length=512,
        feature_dim=14,
        cache_dir=str(tmp_path / "audio_cache"),
        require_audio=True,
        min_coverage=1.0,
    )

    dataset = AISTMotionDataset(
        files=[motion_path],
        aist_cfg=_aist_config(tmp_path),
        sequence_length=20,
        audio_cfg=audio_cfg,
    )
    sample = dataset[0]
    assert sample["audio_context"].shape == (20, 14)
    assert sample["beat_indicator"].shape == (20,)
    assert sample["has_audio"].item() is True
    assert dataset.audio_coverage["clip_fraction"] == 1.0


def test_aist_motion_dataset_skips_missing_pose_key(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_motion.npz"
    good_path = tmp_path / "good_motion.npz"
    np.savez_compressed(bad_path, unrelated=np.zeros((80, 12), dtype=np.float32))
    np.savez_compressed(
        good_path,
        smpl_poses=np.random.default_rng(1).normal(0.0, 0.1, size=(80, 72)).astype(np.float32),
        smpl_trans=np.zeros((80, 3), dtype=np.float32),
        smpl_scaling=np.array([1.0], dtype=np.float32),
    )

    dataset = AISTMotionDataset(files=[bad_path, good_path], aist_cfg=_aist_config(tmp_path), sequence_length=20)
    assert len(dataset) > 0
    assert dataset[0]["motion"].shape == (20, 72)


def test_official_text_split_filters_files(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.txt").write_text("keep_motion\n", encoding="utf-8")
    cfg = AISTConfig(
        root_dir=str(tmp_path),
        split_dir=str(split_dir),
        ignore_list_path=None,
        file_glob="**/*",
        pose_keys=["smpl_poses"],
        source_fps=60.0,
        target_fps=30.0,
        clip_stride=10,
        train_split=0.8,
        val_split=0.1,
        max_files_per_split=None,
        toy_mode=False,
        toy_max_files_per_split=10,
    )
    keep = tmp_path / "keep_motion.pkl"
    drop = tmp_path / "drop_motion.pkl"
    keep.touch()
    drop.touch()
    selected = select_split_motion_files([drop, keep], cfg, "train")
    assert selected == [keep]


def test_ignore_list_filters_files(tmp_path: Path) -> None:
    ignore_path = tmp_path / "ignore_list.txt"
    ignore_path.write_text("drop_motion\n", encoding="utf-8")
    cfg = AISTConfig(
        root_dir=str(tmp_path),
        split_dir=None,
        ignore_list_path=str(ignore_path),
        file_glob="**/*",
        pose_keys=["smpl_poses"],
        source_fps=60.0,
        target_fps=30.0,
        clip_stride=10,
        train_split=0.8,
        val_split=0.1,
        max_files_per_split=None,
        toy_mode=False,
        toy_max_files_per_split=10,
    )
    keep = tmp_path / "keep_motion.pkl"
    drop = tmp_path / "drop_motion.pkl"
    keep.touch()
    drop.touch()
    selected = select_split_motion_files([drop, keep], cfg, "train")
    assert drop not in selected
