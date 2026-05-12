"""Render saved checkpoint outputs as research-grade SMPL previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from embodied_motion_flow.config import config_to_dict, load_config
from embodied_motion_flow.data.dataset import build_dataloaders
from embodied_motion_flow.models.cross_attention_diffusion import AudioConditionedTransformerDenoiser
from embodied_motion_flow.models.diffusion import DDPMScheduler
from embodied_motion_flow.models.factory import build_denoiser
from embodied_motion_flow.rendering.smpl_renderer import render_batch_previews
from embodied_motion_flow.reproducibility import set_global_seed
from embodied_motion_flow.training.ema import ExponentialMovingAverage
from embodied_motion_flow.utils.device import resolve_device
from embodied_motion_flow.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for posterior checkpoint visualization."""
    parser = argparse.ArgumentParser(description="Render SMPL previews from a trained checkpoint.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/model.pt", help="Checkpoint path.")
    parser.add_argument("--split", type=str, choices=("train", "val", "test"), default="test", help="Dataset split to render.")
    parser.add_argument("--output-dir", type=str, default="outputs/previews", help="Directory for GIF/MP4 previews.")
    parser.add_argument("--max-batches", type=int, default=1, help="Maximum dataloader batches to render.")
    parser.add_argument("--max-items", type=int, default=8, help="Maximum sequences to render across selected batches.")
    parser.add_argument("--no-mp4", action="store_true", help="Only render GIF files.")
    return parser.parse_args()


def _audio_context_from_batch(
    batch: dict[str, torch.Tensor],
    motion: torch.Tensor,
    audio_dim: int,
    device: torch.device,
) -> torch.Tensor:
    raw_audio = batch.get("audio_context")
    if isinstance(raw_audio, torch.Tensor):
        return raw_audio.to(device)
    return torch.zeros((motion.shape[0], motion.shape[1], audio_dim), dtype=motion.dtype, device=device)


def _predict_noise(
    model: torch.nn.Module,
    motion: torch.Tensor,
    timesteps: torch.Tensor,
    audio_context: torch.Tensor,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    if isinstance(model, AudioConditionedTransformerDenoiser):
        cond = model(motion, timesteps, audio_context)
        if guidance_scale == 1.0:
            return cond
        uncond = model(motion, timesteps, torch.zeros_like(audio_context))
        return uncond + float(guidance_scale) * (cond - uncond)
    return model(motion, timesteps)


def main() -> None:
    """Load checkpoint and render reconstructed test-batch previews."""
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(
        seed=config.reproducibility.seed,
        deterministic_torch=config.reproducibility.deterministic_torch,
        benchmark_cudnn=config.reproducibility.benchmark_cudnn,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(config.project.log_level, log_file=Path(config.project.output_dir) / "logs" / "visualize_results.log")
    logger = get_logger("embodied_motion_flow.visualize_results")
    device = resolve_device(config.device.preference)
    logger.info("Active config: %s", json.dumps(config_to_dict(config), indent=2))
    logger.info("Selected device: %s", device)

    data_splits = build_dataloaders(config)
    dataloader = {
        "train": data_splits.train_loader,
        "val": data_splits.val_loader,
        "test": data_splits.test_loader,
    }[args.split]

    model = build_denoiser(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ema = ExponentialMovingAverage(model, decay=config.training.ema_decay)
    if config.inference.use_ema and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()

    scheduler = DDPMScheduler(
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule=config.diffusion.beta_schedule,
    ).to(device)
    start_step = min(config.evaluation.reconstruction_steps, scheduler.timesteps - 1)

    rendered = 0
    with ema.average_parameters(model) if config.inference.use_ema else torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.max_batches or rendered >= args.max_items:
                break
            reference = batch["motion"].to(device)
            audio_context = _audio_context_from_batch(batch, reference, config.model.audio_dim, device)
            model_fn = lambda sample, step: _predict_noise(
                model,
                sample,
                step,
                audio_context,
                guidance_scale=config.inference.guidance_scale,
            )
            _, generated = scheduler.reconstruct(model=model_fn, x0=reference, start_timestep=start_step)

            remaining = args.max_items - rendered
            reference = reference[:remaining]
            generated = generated[:remaining]
            source_paths = batch.get("source_path")
            if isinstance(source_paths, list):
                sequence_ids = [Path(path).stem for path in source_paths[: reference.shape[0]]]
            else:
                sequence_ids = [f"{args.split}_{batch_idx:03d}_{idx:03d}" for idx in range(reference.shape[0])]
            beat_indicators = batch.get("beat_indicator")
            if isinstance(beat_indicators, torch.Tensor):
                beat_indicators = beat_indicators[: reference.shape[0]]
            else:
                beat_indicators = None

            paths = render_batch_previews(
                motions=reference,
                generated_motions=generated,
                beat_indicators=beat_indicators,
                sequence_ids=sequence_ids,
                output_dir=output_dir,
                fps=config.visualization.fps,
                max_frames=config.visualization.max_frames,
                dpi=config.visualization.dpi,
                save_mp4=not args.no_mp4,
            )
            rendered += len(paths)
            logger.info("Rendered %d preview(s) from batch %d.", len(paths), batch_idx)

    logger.info("Visualization complete. Rendered %d preview(s) to %s.", rendered, output_dir)


if __name__ == "__main__":
    main()
