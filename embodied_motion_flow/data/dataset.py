"""Dataset wrappers for humanoid trajectory tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from embodied_motion_flow.config import ExperimentConfig
from embodied_motion_flow.data.aist_loader import build_aist_dataset
from embodied_motion_flow.data.synthetic_generator import generate_synthetic_batch


class MotionTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for synthetic trajectories.

    Shapes:
        motion: [time, joints]
        anomaly_label: scalar {0, 1}
    """

    def __init__(self, trajectories: np.ndarray, anomaly_labels: np.ndarray, modes: list[str] | None = None) -> None:
        if trajectories.ndim != 3:
            raise ValueError(f"Expected trajectories [B, T, J], received shape {trajectories.shape}")
        if modes is not None and len(modes) != trajectories.shape[0]:
            raise ValueError("modes length must match trajectory batch size")
        self._trajectories = torch.tensor(trajectories, dtype=torch.float32)
        self._labels = torch.tensor(anomaly_labels, dtype=torch.long)
        self._modes = modes or ["unknown"] * int(trajectories.shape[0])

    def __len__(self) -> int:
        return int(self._trajectories.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "motion": self._trajectories[index],
            "anomaly_label": self._labels[index],
            "mode": self._modes[index],
        }


@dataclass(frozen=True)
class DataSplits:
    """Train/val/test dataloaders and backing datasets."""

    train_loader: DataLoader[dict[str, torch.Tensor]]
    val_loader: DataLoader[dict[str, torch.Tensor]]
    test_loader: DataLoader[dict[str, torch.Tensor]]
    train_dataset: Dataset[dict[str, torch.Tensor]]
    val_dataset: Dataset[dict[str, torch.Tensor]]
    test_dataset: Dataset[dict[str, torch.Tensor]]


def _build_split_dataset(config: ExperimentConfig, split: str, size: int, seed: int) -> MotionTrajectoryDataset:
    batch = generate_synthetic_batch(batch_size=size, data_cfg=config.data, seed=seed)
    return MotionTrajectoryDataset(
        trajectories=batch.trajectories,
        anomaly_labels=batch.anomaly_labels,
        modes=batch.modes,
    )


def build_dataloaders(config: ExperimentConfig) -> DataSplits:
    """Build deterministic train/val/test data splits and dataloaders."""
    base_seed = config.reproducibility.seed
    if config.data.source == "aistpp":
        audio_cfg = config.audio if config.audio.enabled else None
        train_ds = build_aist_dataset(config.data.aist, config.data.sequence_length, "train", audio_cfg=audio_cfg)
        val_ds = build_aist_dataset(config.data.aist, config.data.sequence_length, "val", audio_cfg=audio_cfg)
        test_ds = build_aist_dataset(config.data.aist, config.data.sequence_length, "test", audio_cfg=audio_cfg)
    elif config.data.source == "synthetic":
        train_ds = _build_split_dataset(config, "train", config.data.train_samples, base_seed + 1)
        val_ds = _build_split_dataset(config, "val", config.data.val_samples, base_seed + 2)
        test_ds = _build_split_dataset(config, "test", config.data.test_samples, base_seed + 3)
    else:
        raise ValueError(f"Unsupported data.source: {config.data.source}")

    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": False,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return DataSplits(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )
