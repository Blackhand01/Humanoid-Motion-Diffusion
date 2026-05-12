"""AIST++ SMPL motion loader.

This module loads AIST++-style SMPL pose files and exposes clips as tensors
with shape [time, 72], where 72 = 24 SMPL joints x 3 axis-angle channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
from typing import Any
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from embodied_motion_flow.audio.audio_processor import find_audio_for_motion, load_or_extract_audio_features
from embodied_motion_flow.config import AISTConfig, AudioConfig


SMPL_NUM_JOINTS = 24
SMPL_AXIS_ANGLE_DIM = 72
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AISTClipIndex:
    """Index metadata for one SMPL clip."""

    motion_id: int
    start_frame: int
    end_frame: int
    source_path: Path


@dataclass(frozen=True)
class AISTMotionRecord:
    """Loaded official AIST++ SMPL motion record."""

    poses: np.ndarray
    translations: np.ndarray
    scaling: float


def _load_motion_file(path: Path) -> dict[str, Any]:
    """Load a pickle or npz motion file into a dictionary."""
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as handle:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                payload = pickle.load(handle)
    elif suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        payload = {key: npz[key] for key in npz.files}
    else:
        raise ValueError(f"Unsupported AIST++ motion file suffix: {path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError(f"AIST++ file must contain a dictionary payload: {path}")
    return payload


def load_ignore_list(ignore_list_path: str | Path | None) -> set[str]:
    """Load official AIST++ ignore ids from ignore_list.txt."""
    if ignore_list_path is None:
        return set()
    path = Path(ignore_list_path).expanduser()
    if not path.exists():
        LOGGER.warning("AIST++ ignore list not found; no sequences filtered: %s", path)
        return set()
    ignored: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        token = raw_line.strip()
        if not token or token.startswith("#"):
            continue
        ignored.add(Path(token.split()[0]).stem)
    return ignored


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


def extract_official_smpl_record(payload: dict[str, Any]) -> AISTMotionRecord:
    """Extract official AIST++ SMPL pose, translation, and scaling fields.

    Official motion files contain:
        smpl_poses: [frames, 24, 3] or flattened [frames, 72] axis-angle radians.
        smpl_trans: [frames, 3] root translation.
        smpl_scaling: scalar body scale.

    Translation is scaled by smpl_scaling. Axis-angle pose values are rotations
    in radians and are not scaled, because scaling rotation magnitudes would
    corrupt the SMPL kinematic state.
    """
    missing = [key for key in ("smpl_poses", "smpl_trans") if key not in payload]
    if missing:
        raise ValueError(f"Missing official AIST++ key(s): {missing}")

    raw_poses = np.asarray(payload["smpl_poses"], dtype=np.float32)
    if raw_poses.ndim == 3 and raw_poses.shape[-2:] == (SMPL_NUM_JOINTS, 3):
        poses = raw_poses.reshape(raw_poses.shape[0], SMPL_AXIS_ANGLE_DIM)
    elif raw_poses.ndim == 2 and raw_poses.shape[-1] == SMPL_AXIS_ANGLE_DIM:
        poses = raw_poses
    else:
        raise ValueError(f"smpl_poses must be [N,24,3] or [N,72], received {raw_poses.shape}")

    translations = np.asarray(payload["smpl_trans"], dtype=np.float32)
    if translations.ndim != 2 or translations.shape[-1] != 3:
        raise ValueError(f"smpl_trans must be [N,3], received {translations.shape}")
    if translations.shape[0] != poses.shape[0]:
        raise ValueError(f"smpl_trans frame count {translations.shape[0]} != smpl_poses frame count {poses.shape[0]}")

    raw_scaling = payload.get("smpl_scaling", 1.0)
    scaling = float(np.asarray(raw_scaling, dtype=np.float32).reshape(-1)[0])
    translations = translations * scaling
    return AISTMotionRecord(
        poses=np.nan_to_num(poses.astype(np.float32), copy=False),
        translations=np.nan_to_num(translations.astype(np.float32), copy=False),
        scaling=scaling,
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


def _motion_id(path: Path) -> str:
    """Return canonical AIST++ motion id for file-to-split matching."""
    return path.stem


def _read_text_split(path: Path) -> set[str]:
    ids: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        token = raw_line.strip()
        if not token or token.startswith("#"):
            continue
        ids.add(Path(token.split()[0]).stem)
    return ids


def _read_json_split(path: Path, split: str) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get(split, payload.get(f"{split}s", []))
    else:
        values = payload
    return {Path(str(value)).stem for value in values}


def _read_numpy_split(path: Path) -> set[str]:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        values: list[Any] = []
        for key in arr.files:
            values.extend(arr[key].tolist())
    else:
        values = arr.tolist()
    if not isinstance(values, list):
        values = [values]
    return {Path(str(value)).stem for value in values}


def discover_official_split_ids(split_dir: str | Path | None, split: str) -> set[str] | None:
    """Read official AIST++ split ids if split files are available."""
    if split_dir is None:
        return None
    root = Path(split_dir).expanduser()
    if not root.exists():
        LOGGER.warning("AIST++ split_dir does not exist; using deterministic file split: %s", root)
        return None

    candidate_files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and split.lower() in path.stem.lower() and path.suffix.lower() in {".txt", ".json", ".npy", ".npz"}
    ]
    if not candidate_files:
        LOGGER.warning("No official AIST++ %s split file found under %s; using deterministic file split.", split, root)
        return None

    ids: set[str] = set()
    for path in candidate_files:
        try:
            if path.suffix.lower() == ".txt":
                ids.update(_read_text_split(path))
            elif path.suffix.lower() == ".json":
                ids.update(_read_json_split(path, split))
            elif path.suffix.lower() in {".npy", ".npz"}:
                ids.update(_read_numpy_split(path))
        except Exception as exc:
            LOGGER.warning("Could not parse AIST++ split file %s: %s", path, exc)

    if not ids:
        LOGGER.warning("Official AIST++ %s split files were empty/unreadable; using deterministic file split.", split)
        return None
    return ids


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


def select_split_motion_files(files: list[Path], aist_cfg: AISTConfig, split: str) -> list[Path]:
    """Select motion files using official split ids when available."""
    ignored = load_ignore_list(aist_cfg.ignore_list_path)
    if ignored:
        before = len(files)
        files = [path for path in files if _motion_id(path) not in ignored and path.name not in ignored]
        LOGGER.info("Filtered %d AIST++ ignored sequence(s).", before - len(files))

    split_ids = discover_official_split_ids(aist_cfg.split_dir, split)
    if split_ids is None:
        return split_motion_files(files, split, aist_cfg.train_split, aist_cfg.val_split)

    selected = [path for path in files if _motion_id(path) in split_ids or path.name in split_ids]
    if not selected:
        LOGGER.warning(
            "Official AIST++ %s split matched zero files under %s; using deterministic file split.",
            split,
            aist_cfg.root_dir,
        )
        return split_motion_files(files, split, aist_cfg.train_split, aist_cfg.val_split)
    return selected


class AISTMotionDataset(Dataset[dict[str, Any]]):
    """Dataset of AIST++ SMPL pose clips.

    Each sample contains:
        motion: Tensor [time, 72], SMPL axis-angle pose in radians.
        anomaly_label: LongTensor scalar, always 0 for real nominal clips.
        mode: String motion source label for downstream visualization selection.
        source_path: Original file path.
    """

    def __init__(
        self,
        files: list[Path],
        aist_cfg: AISTConfig,
        sequence_length: int,
        audio_cfg: AudioConfig | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self.aist_cfg = aist_cfg
        self._motions: list[np.ndarray] = []
        self._translations: list[np.ndarray] = []
        self._scalings: list[float] = []
        self._audio_contexts: list[np.ndarray] = []
        self._beat_masks: list[np.ndarray] = []
        self._audio_paths: list[str] = []
        self._has_audio: list[bool] = []
        self._indices: list[AISTClipIndex] = []

        downsample_stride = max(1, int(round(aist_cfg.source_fps / aist_cfg.target_fps)))
        for path in files:
            try:
                payload = _load_motion_file(path)
                record = extract_official_smpl_record(payload)
                poses = record.poses[::downsample_stride]
                translations = record.translations[::downsample_stride]
            except Exception as exc:
                LOGGER.warning("Skipping AIST++ motion file %s: %s", path, exc)
                continue
            if poses.shape[0] < sequence_length:
                LOGGER.warning(
                    "Skipping short AIST++ motion file %s: %d frames after downsampling, need %d.",
                    path,
                    poses.shape[0],
                    sequence_length,
                )
                continue
            self._motions.append(poses)
            self._translations.append(translations)
            self._scalings.append(record.scaling)
            audio_context, beat_mask, audio_path = self._load_audio_alignment(path, poses.shape[0], audio_cfg)
            self._audio_contexts.append(audio_context)
            self._beat_masks.append(beat_mask)
            self._audio_paths.append(str(audio_path) if audio_path is not None else "")
            self._has_audio.append(audio_path is not None)

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
        if audio_cfg is not None and audio_cfg.enabled:
            coverage = self.audio_coverage
            LOGGER.info(
                "AIST++ audio coverage: %.1f%% clips (%d/%d source motions) under %s.",
                100.0 * coverage["clip_fraction"],
                int(coverage["source_motions_with_audio"]),
                int(coverage["source_motions"]),
                audio_cfg.root_dir,
            )
            if coverage["clip_fraction"] < audio_cfg.min_coverage:
                message = (
                    f"AIST++ audio coverage {coverage['clip_fraction']:.3f} is below "
                    f"audio.min_coverage={audio_cfg.min_coverage:.3f}."
                )
                if audio_cfg.require_audio:
                    raise RuntimeError(message)
                LOGGER.warning(message)

    @property
    def source_paths(self) -> list[Path]:
        """Unique source files loaded by this dataset."""
        return sorted({index.source_path for index in self._indices})

    @property
    def audio_coverage(self) -> dict[str, float | int]:
        """Return clip/source-level audio availability diagnostics."""
        clips_with_audio = sum(1 for index in self._indices if self._has_audio[index.motion_id])
        return {
            "clips": len(self._indices),
            "clips_with_audio": clips_with_audio,
            "clip_fraction": clips_with_audio / max(len(self._indices), 1),
            "source_motions": len(self._motions),
            "source_motions_with_audio": sum(1 for value in self._has_audio if value),
        }

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self._indices[index]
        motion = self._motions[clip.motion_id][clip.start_frame : clip.end_frame]
        translation = self._translations[clip.motion_id][clip.start_frame : clip.end_frame]
        return {
            "motion": torch.tensor(motion, dtype=torch.float32),
            "smpl_poses": torch.tensor(motion.reshape(self.sequence_length, SMPL_NUM_JOINTS, 3), dtype=torch.float32),
            "smpl_trans": torch.tensor(translation, dtype=torch.float32),
            "smpl_scaling": torch.tensor(self._scalings[clip.motion_id], dtype=torch.float32),
            "audio_context": torch.tensor(
                self._audio_contexts[clip.motion_id][clip.start_frame : clip.end_frame],
                dtype=torch.float32,
            ),
            "beat_indicator": torch.tensor(
                self._beat_masks[clip.motion_id][clip.start_frame : clip.end_frame],
                dtype=torch.float32,
            ),
            "has_audio": torch.tensor(self._has_audio[clip.motion_id], dtype=torch.bool),
            "audio_path": self._audio_paths[clip.motion_id],
            "anomaly_label": torch.tensor(0, dtype=torch.long),
            "mode": "aistpp_smpl",
            "source_path": str(clip.source_path),
            "start_frame": torch.tensor(clip.start_frame, dtype=torch.long),
        }

    def _load_audio_alignment(
        self,
        motion_path: Path,
        motion_frame_count: int,
        audio_cfg: AudioConfig | None,
    ) -> tuple[np.ndarray, np.ndarray, Path | None]:
        """Return audio context and beat mask aligned to downsampled motion frames."""
        feature_dim = audio_cfg.feature_dim if audio_cfg is not None else 14
        zeros = np.zeros((motion_frame_count, feature_dim), dtype=np.float32)
        beat_zeros = np.zeros((motion_frame_count,), dtype=np.float32)
        if audio_cfg is None or not audio_cfg.enabled:
            return zeros, beat_zeros, None
        if not Path(audio_cfg.root_dir).expanduser().exists():
            if audio_cfg.require_audio:
                raise FileNotFoundError(f"audio.root_dir does not exist: {audio_cfg.root_dir}")
            if audio_cfg.log_missing_audio:
                LOGGER.warning("AIST++ audio root not found: %s; using zero audio context.", audio_cfg.root_dir)
            return zeros, beat_zeros, None
        try:
            audio_path = find_audio_for_motion(motion_path, audio_cfg.root_dir, audio_cfg.allowed_extensions)
        except ValueError as exc:
            LOGGER.warning("Could not infer audio id for %s: %s", motion_path, exc)
            if audio_cfg.require_audio:
                raise
            return zeros, beat_zeros, None
        if audio_path is None:
            if audio_cfg.require_audio:
                raise FileNotFoundError(f"No AIST++ audio found for {motion_path} under {audio_cfg.root_dir}")
            if audio_cfg.log_missing_audio:
                LOGGER.warning("No AIST++ audio found for %s under %s; using zero audio context.", motion_path, audio_cfg.root_dir)
            return zeros, beat_zeros, None
        try:
            features = load_or_extract_audio_features(
                audio_path=audio_path,
                motion_frame_count=motion_frame_count,
                motion_fps=self.aist_cfg.target_fps,
                sample_rate=audio_cfg.sample_rate,
                hop_length=audio_cfg.hop_length,
                cache_dir=audio_cfg.cache_dir,
            )
            if features.frame_features.shape[1] != feature_dim:
                message = (
                    "Audio feature dim "
                    f"{features.frame_features.shape[1]} != configured feature_dim {feature_dim} for {audio_path}."
                )
                if audio_cfg.require_audio:
                    raise ValueError(message)
                LOGGER.warning(
                    "%s Using zero audio context.",
                    message,
                )
                return zeros, beat_zeros, None
            return features.frame_features.astype(np.float32), features.beat_mask.astype(np.float32), audio_path
        except Exception as exc:
            if audio_cfg.require_audio:
                raise
            LOGGER.warning("Could not extract audio features for %s: %s", audio_path, exc)
            return zeros, beat_zeros, None


def build_aist_dataset(
    aist_cfg: AISTConfig,
    sequence_length: int,
    split: str,
    audio_cfg: AudioConfig | None = None,
) -> AISTMotionDataset:
    """Build one deterministic AIST++ split dataset."""
    files = discover_aist_motion_files(aist_cfg.root_dir, aist_cfg.file_glob)
    split_files = select_split_motion_files(files, aist_cfg, split)
    file_cap = aist_cfg.toy_max_files_per_split if aist_cfg.toy_mode else aist_cfg.max_files_per_split
    if file_cap is not None:
        split_files = split_files[:file_cap]
    return AISTMotionDataset(files=split_files, aist_cfg=aist_cfg, sequence_length=sequence_length, audio_cfg=audio_cfg)
