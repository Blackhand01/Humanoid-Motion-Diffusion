"""Standard repository entrypoint for training and showcase generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from embodied_motion_flow.config import config_to_dict, load_config
from embodied_motion_flow.pipelines.showcase_pipeline import (
    configure_showcase_runtime,
    run_showcase_generation,
    write_showcase_zip,
)
from embodied_motion_flow.reproducibility import set_global_seed
from embodied_motion_flow.training.engine import run_training_pipeline
from embodied_motion_flow.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Humanoid Motion Diffusion training/showcase pipelines.")
    parser.add_argument("mode", choices=("train", "showcase", "full"), help="Pipeline stage to execute.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config path or profile name.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path for showcase mode.")
    parser.add_argument("--track", type=str, default="", help="Optional showcase audio path override.")
    parser.add_argument("--zip-path", type=str, default="", help="Optional zip archive output path.")
    parser.add_argument("--fresh-start", action="store_true", help="Delete config.project.output_dir before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode in {"showcase", "full"}:
        config = configure_showcase_runtime(config, auto_resume=not args.fresh_start, track_path=args.track or None)
    if args.fresh_start:
        output_root = Path(config.project.output_dir)
        if output_root.exists():
            shutil.rmtree(output_root)

    log_root = Path(config.project.output_dir) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    configure_logging(config.project.log_level, log_file=log_root / "run_pipeline.log")
    logger = get_logger("embodied_motion_flow.run_pipeline")
    logger.info("Active config: %s", json.dumps(config_to_dict(config), indent=2))
    set_global_seed(
        seed=config.reproducibility.seed,
        deterministic_torch=config.reproducibility.deterministic_torch,
        benchmark_cudnn=config.reproducibility.benchmark_cudnn,
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(config.project.output_dir) / "checkpoints" / "model.pt"
    if args.mode in {"train", "full"}:
        _, training_artifacts = run_training_pipeline(config)
        checkpoint_path = training_artifacts.checkpoint_path
    if args.mode in {"showcase", "full"}:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        showcase_artifacts = run_showcase_generation(config, checkpoint_path, logger)
        if args.zip_path:
            zip_path = write_showcase_zip(config, showcase_artifacts, args.config, args.zip_path)
            logger.info("Downloadable archive: %s", zip_path)


if __name__ == "__main__":
    main()
