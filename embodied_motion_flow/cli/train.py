"""CLI entrypoint for diffusion training."""

from __future__ import annotations

import argparse
from pathlib import Path

from embodied_motion_flow.config import load_config
from embodied_motion_flow.reproducibility import set_global_seed
from embodied_motion_flow.training.engine import run_training_pipeline
from embodied_motion_flow.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Embodied-Motion-Flow diffusion model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log_file = Path(config.project.output_dir) / "logs" / "train_cli.log"
    configure_logging(config.project.log_level, log_file=log_file)
    logger = get_logger("embodied_motion_flow.train_cli")
    set_global_seed(
        seed=config.reproducibility.seed,
        deterministic_torch=config.reproducibility.deterministic_torch,
        benchmark_cudnn=config.reproducibility.benchmark_cudnn,
    )
    metrics, artifacts = run_training_pipeline(config)
    logger.info("Training complete. Metrics: %s", metrics)
    logger.info("Artifacts: %s", artifacts.as_dict())


if __name__ == "__main__":
    main()
