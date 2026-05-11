"""AIST++ SMPL motion loader.

This module loads AIST++-style SMPL pose files and exposes clips as tensors
with shape [time, 72], where 72 = 24 SMPL joints x 3 axis-angle channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from embodied_motion_flow.config import AISTConfig


SMPL_NUM_JOINTS = 24
SMPL_AXIS_ANGLE_DIM = 72


@dataclass(frozen=True)
class AISTClipIndex:
    """Index metadata for one SMPL clip."""

    motion_id: int
    start_frame: int
    end_frame: int
    source_path: Path


def _load_motion_file(path: Path) -> dict[str, Any]:
    """Load a pickle or npz motion file into a dictionary."""
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    elif suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        payload = {key: npz[key] for key in npz.files}
    else:
        raise ValueError(f"Unsupported AIST++ motion file suffix: {path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError(f"AIST++ file must contain a dictionary payload: {path}")
    return payload


def _candidate_arrays(payload: Any, pose_keys: list[str]) -> list[np.ndarray]:
    """Recursively collect arrays likely to contain SMPL pose axis-angles."""
    arrays: list[np.ndarray] = []
    if isinstance(payload, dict):
        for key in pose_keys:
            if key in payload:
                arrays.append(np.asarray(payload[key]))
        for value in payload.values():
            arrays.extend(_candidate_arrays(value, pose_keys))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            arrays.extend(_candidate_arrays(value, pose_keys))
    return arrays


def extract_smpl_poses(payload: dict[str, Any], pose_keys: list[str]) -> np.ndarray:
    """Extract SMPL axis-angle poses with shape [frames, 72]."""
    for array in _candidate_arrays(payload, pose_keys):
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-2:] == (SMPL_NUM_JOINTS, 3):
            return arr.reshape(arr.shape[0], SMPL_AXIS_ANGLE_DIM)
        if arr.ndim == 2 and arr.shape[-1] == SMPL_AXIS_ANGLE_DIM:
            return arr
        if arr.ndim == 2 and arr.shape[-1] > SMPL_AXIS_ANGLE_DIM:
            return arr[:, :SMPL_AXIS_ANGLE_DIM]

    available = sorted(payload.keys())
    raise ValueError(
        "Could not find SMPL poses [T,72] or [T,24,3]. "
        f"Candidate keys={pose_keys}; available top-level keys={available}"
    )


def discover_aist_motion_files(root_dir: str | Path, file_glob: str) -> list[Path]:
    """Return sorted AIST++ .pkl/.npz motion files under root_dir."""
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"AIST++ root_dir does not exist: {root}")
    files = [
        path
        for path in sorted(root.glob(file_glob))
        if path.is_file() and path.suffix.lower() in {".pkl", ".npz"}
    ]
    if not files:
        raise FileNotFoundError(f"No AIST++ .pkl/.npz files found under {root} with glob {file_glob!r}")
    return files


def split_motion_files(files: list[Path], split: str, train_split: float, val_split: float) -> list[Path]:
    """Deterministically split sorted motion files into train/val/test partitions."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0, 1)")
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0, 1)")
    if train_split + val_split >= 1.0:
        raise ValueError("train_split + val_split must be < 1")

    n_files = len(files)
    train_end = max(1, int(round(n_files * train_split)))
    val_end = min(n_files, train_end + int(round(n_files * val_split)))
    if split == "train":
        return files[:train_end]
    if split == "val":
        return files[train_end:val_end] or files[:1]
    return files[val_end:] or files[-1:]


class AISTMotionDataset(Dataset[dict[str, Any]]):
    """Dataset of AIST++ SMPL pose clips.

    Each sample contains:
        motion: Tensor [time, 72], SMPL axis-angle pose in radians.
        anomaly_label: LongTensor scalar, always 0 for real nominal clips.
        mode: String motion source label for downstream visualization selection.
        source_path: Original file path.
    """

    def __init__(self, files: list[Path], aist_cfg: AISTConfig, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self.aist_cfg = aist_cfg
        self._motions: list[np.ndarray] = []
        self._indices: list[AISTClipIndex] = []

        downsample_stride = max(1, int(round(aist_cfg.source_fps / aist_cfg.target_fps)))
        for motion_id, path in enumerate(files):
            payload = _load_motion_file(path)
            poses = extract_smpl_poses(payload, aist_cfg.pose_keys)[::downsample_stride]
            if poses.shape[0] < sequence_length:
                continue
            poses = np.nan_to_num(poses.astype(np.float32), copy=False)
            self._motions.append(poses)

            stride = max(1, aist_cfg.clip_stride)
            for start in range(0, poses.shape[0] - sequence_length + 1, stride):
                self._indices.append(
                    AISTClipIndex(
                        motion_id=len(self._motions) - 1,
                        start_frame=start,
                        end_frame=start + sequence_length,
                        source_path=path,
                    )
                )

        if not self._indices:
            raise ValueError(
                "AIST++ files were found, but no clips were long enough after downsampling. "
                f"Required sequence_length={sequence_length}."
            )

    @property
    def source_paths(self) -> list[Path]:
        """Unique source files loaded by this dataset."""
        return sorted({index.source_path for index in self._indices})

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self._indices[index]
        motion = self._motions[clip.motion_id][clip.start_frame : clip.end_frame]
        return {
            "motion": torch.tensor(motion, dtype=torch.float32),
            "anomaly_label": torch.tensor(0, dtype=torch.long),
            "mode": "aistpp_smpl",
            "source_path": str(clip.source_path),
            "start_frame": torch.tensor(clip.start_frame, dtype=torch.long),
        }


def build_aist_dataset(aist_cfg: AISTConfig, sequence_length: int, split: str) -> AISTMotionDataset:
    """Build one deterministic AIST++ split dataset."""
    files = discover_aist_motion_files(aist_cfg.root_dir, aist_cfg.file_glob)
    split_files = split_motion_files(files, split, aist_cfg.train_split, aist_cfg.val_split)
    if aist_cfg.max_files_per_split is not None:
        split_files = split_files[: aist_cfg.max_files_per_split]
    return AISTMotionDataset(files=split_files, aist_cfg=aist_cfg, sequence_length=sequence_length)
