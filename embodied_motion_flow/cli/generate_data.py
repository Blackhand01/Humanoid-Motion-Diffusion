"""CLI entrypoint for synthetic humanoid trajectory generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_motion_flow.config import config_to_dict, load_config
from embodied_motion_flow.data.synthetic_generator import generate_synthetic_batch
from embodied_motion_flow.reproducibility import set_global_seed
from embodied_motion_flow.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic humanoid trajectories.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override number of samples. Defaults to train_samples + val_samples + test_samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(
        seed=config.reproducibility.seed,
        deterministic_torch=config.reproducibility.deterministic_torch,
        benchmark_cudnn=config.reproducibility.benchmark_cudnn,
    )

    output_dir = Path(config.project.output_dir) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(config.project.log_level, log_file=Path(config.project.output_dir) / "logs" / "generate_data.log")
    logger = get_logger("embodied_motion_flow.generate_data")
    logger.info("Active config: %s", json.dumps(config_to_dict(config), indent=2))

    total_samples = (
        args.samples
        if args.samples is not None
        else config.data.train_samples + config.data.val_samples + config.data.test_samples
    )
    batch = generate_synthetic_batch(
        batch_size=total_samples,
        data_cfg=config.data,
        seed=config.reproducibility.seed + 99,
    )
    out_path = output_dir / "synthetic_dataset.npz"
    np.savez_compressed(
        out_path,
        trajectories=batch.trajectories,
        anomaly_labels=batch.anomaly_labels,
        modes=np.asarray(batch.modes),
    )
    logger.info("Saved synthetic dataset to %s with shape %s", out_path, batch.trajectories.shape)


if __name__ == "__main__":
    main()
