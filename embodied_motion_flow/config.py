"""Structured configuration loading for Embodied-Motion-Flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
        split_dir: Optional directory containing official AIST++ split files.
        ignore_list_path: Optional official AIST++ ignore_list.txt path.
        file_glob: Recursive glob pattern relative to root_dir.
        pose_keys: Candidate dictionary keys for SMPL axis-angle poses [T, 72].
        source_fps: Frame rate of raw AIST++ motion files.
        target_fps: Target frame rate after deterministic temporal downsampling.
        clip_stride: Sliding-window stride in frames after downsampling.
        train_split: Fraction of sorted files assigned to train.
        val_split: Fraction of sorted files assigned to validation.
        max_files_per_split: Optional cap for lightweight validation notebooks.
        toy_mode: Restrict local runs to toy_max_files_per_split files per split.
        toy_max_files_per_split: File cap used when toy_mode is true.
    """

    root_dir: str
    split_dir: str | None
    ignore_list_path: str | None
    file_glob: str
    pose_keys: list[str]
    source_fps: float
    target_fps: float
    clip_stride: int
    train_split: float
    val_split: float
    max_files_per_split: int | None
    toy_mode: bool
    toy_max_files_per_split: int


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
    audio_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    time_embedding_dim: int


@dataclass(frozen=True)
class AudioConfig:
    """Audio feature extraction settings for music conditioning."""

    enabled: bool
    root_dir: str
    sample_rate: int
    hop_length: int
    feature_dim: int
    cache_dir: str = "outputs/audio_features"
    allowed_extensions: list[str] = field(default_factory=lambda: [".wav", ".mp3", ".flac", ".m4a", ".mp4"])
    require_audio: bool = False
    min_coverage: float = 0.0
    log_missing_audio: bool = False


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
    max_duration_seconds: float | None
    cond_dropout: float = 0.0
    ema_decay: float = 0.0
    accumulation_steps: int = 1
    warmup_epochs: float = 0.0
    warmup_steps: int | None = None
    min_learning_rate_ratio: float = 0.05
    auto_resume: bool = False
    resume_from_checkpoint: str | None = None
    log_every_steps: int = 25


@dataclass(frozen=True)
class InferenceConfig:
    """Sampling and classifier-free guidance controls."""

    guidance_scale: float = 1.0
    use_ema: bool = True
    sliding_window_frames: int = 240
    prefix_frames: int = 60
    generation_frames: int = 450
    diffusion_steps: int | None = None


@dataclass(frozen=True)
class ShowcaseConfig:
    """Kaggle showcase generation controls."""

    track_path: str = "data/stardust.wav"
    clip_start_seconds: float = 46.0
    clip_duration_seconds: float = 15.0
    output_dir: str = "outputs/showcase"
    viral_fps: int = 30
    research_fps: int = 30
    render_dpi: int = 160


@dataclass(frozen=True)
class LossConfig:
    """Reconstruction and biomechanical loss weighting."""

    reconstruction_weight: float
    lambda_phys: float
    acceleration_weight: float
    joint_limit_weight: float
    temporal_jitter_weight: float
    self_collision_weight: float = 0.0
    self_collision_margin: float = 0.08


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation metrics and reconstruction controls."""

    metrics: list[str]
    anomaly_score: str
    reconstruction_steps: int
    tsi_failure_threshold: float = 0.25
    jlvr_failure_threshold: float = 0.02
    self_collision_failure_threshold: float = 0.01


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
    audio: AudioConfig
    diffusion: DiffusionConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    showcase: ShowcaseConfig = field(default_factory=ShowcaseConfig)


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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge profile overrides into a base config dictionary."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config_path(config_path: str | Path) -> Path:
    """Resolve a config file path or named profile under ``configs/``."""
    path = Path(config_path)
    candidates = [path]
    if path.suffix not in {".yaml", ".yml"}:
        candidates.append(Path("configs") / f"{path}.yaml")
    elif not path.is_absolute():
        candidates.append(Path("configs") / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve config '{config_path}'. Checked: {checked}")


def _resolve_inherited_path(parent: str | Path, current_path: Path) -> Path:
    parent_path = Path(parent)
    if parent_path.is_absolute() and parent_path.exists():
        return parent_path
    relative = current_path.parent / parent_path
    if relative.exists():
        return relative
    return resolve_config_path(parent_path)


def _load_yaml_profile(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML profile and recursively apply its ``inherits`` chain."""
    seen = seen or set()
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Recursive config inheritance detected at {path}")
    seen.add(resolved)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    inherited = raw.pop("inherits", None)
    if inherited is None:
        return raw
    base = _load_yaml_profile(_resolve_inherited_path(inherited, path), seen)
    return _deep_merge(base, raw)


def _from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Create an :class:`ExperimentConfig` from a parsed YAML dictionary."""
    training_raw = dict(raw["training"])
    inference_raw = dict(raw.get("inference", {}))
    showcase_raw = dict(raw.get("showcase", {}))
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
        audio=AudioConfig(**raw["audio"]),
        diffusion=DiffusionConfig(**raw["diffusion"]),
        training=TrainingConfig(**training_raw),
        loss=LossConfig(**raw["loss"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        visualization=VisualizationConfig(**raw["visualization"]),
        inference=InferenceConfig(**inference_raw),
        showcase=ShowcaseConfig(**showcase_raw),
    )


def load_config(config_path: str | Path = "base") -> ExperimentConfig:
    """Load a YAML config path or named profile and return a typed config."""
    path = resolve_config_path(config_path)
    raw = _expand_env_defaults(_load_yaml_profile(path))
    return _from_dict(raw)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert typed config into a plain dictionary for checkpointing/logging."""
    return asdict(config)
