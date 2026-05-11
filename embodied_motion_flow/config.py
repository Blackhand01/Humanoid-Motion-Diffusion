"""Structured configuration loading for Embodied-Motion-Flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project metadata."""

    name: str
    version: str
    output_dir: str
    log_level: str


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Reproducibility settings used by all entrypoints."""

    seed: int
    deterministic_torch: bool
    benchmark_cudnn: bool


@dataclass(frozen=True)
class DeviceConfig:
    """Preferred runtime device policy."""

    preference: str


@dataclass(frozen=True)
class MotionModeConfig:
    """Parameters for a nominal motion pattern."""

    enabled: bool
    frequency_hz: float
    amplitude_scale: float


@dataclass(frozen=True)
class AnomalyConfig:
    """Probabilities and scales for synthetic anomaly injection."""

    enabled: bool
    anomaly_fraction: float
    unstable_pose_probability: float
    sensor_spike_probability: float
    jitter_probability: float
    long_tail_probability: float
    abrupt_acceleration_probability: float
    sensor_noise_std: float
    spike_scale: float
    jitter_scale: float
    drift_scale: float


@dataclass(frozen=True)
class JointLimitConfig:
    """Lower and upper per-joint limits in radians."""

    lower: list[float]
    upper: list[float]


@dataclass(frozen=True)
class AISTConfig:
    """AIST++ SMPL loader settings.

    Fields:
        root_dir: Dataset root containing recursive .pkl/.npz SMPL motion files.
        file_glob: Recursive glob pattern relative to root_dir.
        pose_keys: Candidate dictionary keys for SMPL axis-angle poses [T, 72].
        source_fps: Frame rate of raw AIST++ motion files.
        target_fps: Target frame rate after deterministic temporal downsampling.
        clip_stride: Sliding-window stride in frames after downsampling.
        train_split: Fraction of sorted files assigned to train.
        val_split: Fraction of sorted files assigned to validation.
        max_files_per_split: Optional cap for lightweight validation notebooks.
    """

    root_dir: str
    file_glob: str
    pose_keys: list[str]
    source_fps: float
    target_fps: float
    clip_stride: int
    train_split: float
    val_split: float
    max_files_per_split: int | None


@dataclass(frozen=True)
class DataConfig:
    """Motion data generation/loading and dataloader settings."""

    source: str
    representation: str
    num_joints: int
    input_dim: int
    sequence_length: int
    train_samples: int
    val_samples: int
    test_samples: int
    batch_size: int
    num_workers: int
    sample_rate_hz: float
    motion_modes: dict[str, MotionModeConfig]
    anomalies: AnomalyConfig
    joint_limits: JointLimitConfig
    aist: AISTConfig


@dataclass(frozen=True)
class ModelConfig:
    """Temporal denoiser architecture settings."""

    name: str
    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    time_embedding_dim: int


@dataclass(frozen=True)
class DiffusionConfig:
    """DDPM scheduler settings."""

    timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float
    prediction_target: str


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop and optimizer parameters."""

    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    mixed_precision: bool
    save_every_epochs: int


@dataclass(frozen=True)
class LossConfig:
    """Reconstruction and biomechanical loss weighting."""

    reconstruction_weight: float
    lambda_phys: float
    acceleration_weight: float
    joint_limit_weight: float
    temporal_jitter_weight: float


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation metrics and reconstruction controls."""

    metrics: list[str]
    anomaly_score: str
    reconstruction_steps: int


@dataclass(frozen=True)
class VisualizationConfig:
    """Animation and plotting controls."""

    fps: int
    max_frames: int
    dpi: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete typed configuration for the experiment."""

    project: ProjectConfig
    reproducibility: ReproducibilityConfig
    device: DeviceConfig
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig


def _build_motion_modes(raw_modes: dict[str, Any]) -> dict[str, MotionModeConfig]:
    return {name: MotionModeConfig(**cfg) for name, cfg in raw_modes.items()}


def _expand_env_defaults(value: Any) -> Any:
    """Expand ${VAR:-default} and ${VAR} expressions recursively in YAML values."""
    if isinstance(value, dict):
        return {key: _expand_env_defaults(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_defaults(item) for item in value]
    if not isinstance(value, str):
        return value

    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default = match.group(2)
        if env_name in os.environ:
            return os.environ[env_name]
        return default if default is not None else ""

    return pattern.sub(replace, os.path.expandvars(value))


def _from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Create an :class:`ExperimentConfig` from a parsed YAML dictionary."""
    return ExperimentConfig(
        project=ProjectConfig(**raw["project"]),
        reproducibility=ReproducibilityConfig(**raw["reproducibility"]),
        device=DeviceConfig(**raw["device"]),
        data=DataConfig(
            source=raw["data"].get("source", "synthetic"),
            representation=raw["data"].get("representation", "synthetic_12dof"),
            num_joints=raw["data"]["num_joints"],
            input_dim=raw["data"].get("input_dim", raw["data"]["num_joints"]),
            sequence_length=raw["data"]["sequence_length"],
            train_samples=raw["data"]["train_samples"],
            val_samples=raw["data"]["val_samples"],
            test_samples=raw["data"]["test_samples"],
            batch_size=raw["data"]["batch_size"],
            num_workers=raw["data"]["num_workers"],
            sample_rate_hz=raw["data"]["sample_rate_hz"],
            motion_modes=_build_motion_modes(raw["data"]["motion_modes"]),
            anomalies=AnomalyConfig(**raw["data"]["anomalies"]),
            joint_limits=JointLimitConfig(**raw["data"]["joint_limits"]),
            aist=AISTConfig(**raw["data"]["aist"]),
        ),
        model=ModelConfig(**raw["model"]),
        diffusion=DiffusionConfig(**raw["diffusion"]),
        training=TrainingConfig(**raw["training"]),
        loss=LossConfig(**raw["loss"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        visualization=VisualizationConfig(**raw["visualization"]),
    )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load YAML config and return a typed experiment config object."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = _expand_env_defaults(yaml.safe_load(handle))
    return _from_dict(raw)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert typed config into a plain dictionary for checkpointing/logging."""
    return asdict(config)
