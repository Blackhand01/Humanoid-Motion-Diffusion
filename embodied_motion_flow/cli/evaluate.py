"""CLI entrypoint for evaluation from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from embodied_motion_flow.config import config_to_dict, load_config
from embodied_motion_flow.data.dataset import build_dataloaders
from embodied_motion_flow.evaluation.metrics import default_smpl_joint_limits
from embodied_motion_flow.evaluation.runner import evaluate_model
from embodied_motion_flow.losses.biomechanical import BiomechanicalConsistencyLoss
from embodied_motion_flow.models.diffusion import DDPMScheduler
from embodied_motion_flow.models.factory import build_denoiser
from embodied_motion_flow.reproducibility import set_global_seed
from embodied_motion_flow.training.ema import ExponentialMovingAverage
from embodied_motion_flow.utils.device import resolve_device
from embodied_motion_flow.utils.logging import configure_logging, get_logger
from embodied_motion_flow.visualization.animation import save_denoising_animation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Embodied-Motion-Flow checkpoint.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/model.pt",
        help="Checkpoint path.",
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

    output_root = Path(config.project.output_dir)
    metrics_dir = output_root / "metrics"
    animations_dir = output_root / "animations"
    logs_dir = output_root / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    animations_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(config.project.log_level, log_file=logs_dir / "evaluate.log")
    logger = get_logger("embodied_motion_flow.evaluate")
    device = resolve_device(config.device.preference)
    logger.info("Active config: %s", json.dumps(config_to_dict(config), indent=2))
    logger.info("Selected device: %s", device)

    data_splits = build_dataloaders(config)
    model = build_denoiser(config).to(device)
    scheduler = DDPMScheduler(
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule=config.diffusion.beta_schedule,
    ).to(device)
    if config.data.representation.startswith("smpl") and config.data.input_dim == 72:
        lower_limits, upper_limits = default_smpl_joint_limits(device=device)
    else:
        lower_limits = torch.tensor(config.data.joint_limits.lower, dtype=torch.float32, device=device)
        upper_limits = torch.tensor(config.data.joint_limits.upper, dtype=torch.float32, device=device)
    biomechanical_loss = BiomechanicalConsistencyLoss(
        lower_joint_limits=lower_limits,
        upper_joint_limits=upper_limits,
        acceleration_weight=config.loss.acceleration_weight,
        joint_limit_weight=config.loss.joint_limit_weight,
        temporal_jitter_weight=config.loss.temporal_jitter_weight,
        self_collision_weight=config.loss.self_collision_weight,
        self_collision_margin=config.loss.self_collision_margin,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ema = ExponentialMovingAverage(model, decay=config.training.ema_decay)
    if config.inference.use_ema and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    with ema.average_parameters(model) if config.inference.use_ema else torch.no_grad():
        eval_outputs = evaluate_model(
            config=config,
            model=model,
            scheduler=scheduler,
            dataloader=data_splits.test_loader,
            biomechanical_loss=biomechanical_loss,
            device=device,
        )
    metrics_path = metrics_dir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_outputs.metrics, handle, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    animation_paths = save_denoising_animation(
        noisy_trajectory=eval_outputs.noisy_example,
        denoised_trajectory=eval_outputs.denoised_example,
        gif_path=animations_dir / "denoising.gif",
        mp4_path=animations_dir / "trajectory.mp4",
        fps=config.visualization.fps,
        dpi=config.visualization.dpi,
        max_frames=config.visualization.max_frames,
    )
    logger.info("Saved animations to %s", animation_paths)


if __name__ == "__main__":
    main()
