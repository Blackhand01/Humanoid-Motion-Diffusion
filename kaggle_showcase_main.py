"""Kaggle full-training and dual-mode showcase entrypoint.

This script is intentionally standalone: on Kaggle it can clone the repository,
train or resume a checkpoint, generate a 15-second EMA+CFG motion sample for the
Stardust track segment 0:46-1:01, and export both social and research videos.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Embodied-Motion-Flow Kaggle showcase pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--repo-url", type=str, default="", help="Optional repo URL used when running outside the repo.")
    parser.add_argument("--workdir", type=str, default="/kaggle/working/Embodied-Motion-Flow")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint override for inference.")
    parser.add_argument("--skip-train", action="store_true", help="Use an existing checkpoint and only generate the showcase.")
    parser.add_argument("--track", type=str, default="", help="Optional Stardust audio path override.")
    return parser.parse_args()


def _ensure_repo(repo_url: str, workdir: Path) -> Path:
    """Clone the repo on Kaggle when this script is launched from a bare environment."""
    local_root = Path.cwd()
    if (local_root / "embodied_motion_flow").exists():
        return local_root
    if not repo_url:
        raise RuntimeError("Repository package not found. Provide --repo-url or run from the repository root.")
    if not workdir.exists():
        subprocess.run(["git", "clone", repo_url, str(workdir)], check=True)
    return workdir


def _slice_track(
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
) -> Path:
    """Slice the source audio to the exact showcase segment."""
    if not source_path.exists():
        mp3_fallback = source_path.with_suffix(".mp3")
        if mp3_fallback.exists():
            source_path = mp3_fallback
        else:
            raise FileNotFoundError(f"Showcase track not found: {source_path}")
    audio, sr = librosa.load(source_path, sr=sample_rate, mono=True, offset=start_seconds, duration=duration_seconds)
    if audio.size == 0:
        raise ValueError(f"Audio slice is empty: {source_path} [{start_seconds}, {start_seconds + duration_seconds}]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)
    return output_path


def _config_for_kaggle(config):
    """Enable robust recovery defaults for long Kaggle runs."""
    training = replace(config.training, auto_resume=True, mixed_precision=True)
    device = replace(config.device, preference="auto")
    return replace(config, training=training, device=device)


def _load_model_for_inference(config, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load checkpoint and swap to EMA weights when available."""
    from embodied_motion_flow.models.factory import build_denoiser
    from embodied_motion_flow.training.ema import ExponentialMovingAverage

    model = build_denoiser(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ema = ExponentialMovingAverage(model, decay=config.training.ema_decay)
    if config.inference.use_ema and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        with ema.average_parameters(model):
            ema_model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(ema_model_state)
    model.eval()
    return model


def _run_showcase_generation(config: ExperimentConfig, checkpoint_path: Path, logger: object) -> dict[str, str]:
    """Run EMA+CFG long-form generation and dual-mode rendering."""
    from embodied_motion_flow.audio.audio_processor import extract_audio_features
    from embodied_motion_flow.generation.sampling import generate_sliding_window
    from embodied_motion_flow.models.diffusion import DDPMScheduler
    from embodied_motion_flow.rendering.showcase_renderer import render_research_motion, render_viral_motion
    from embodied_motion_flow.utils.device import resolve_device

    device = resolve_device(config.device.preference)
    output_root = Path(config.showcase.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    track_path = Path(config.showcase.track_path if not config.showcase.track_path.startswith("${") else "data/stardust.wav")
    sliced_audio = _slice_track(
        source_path=track_path,
        output_path=output_root / "stardust_0046_0101.wav",
        start_seconds=config.showcase.clip_start_seconds,
        duration_seconds=config.showcase.clip_duration_seconds,
        sample_rate=config.audio.sample_rate,
    )
    logger.info("Prepared showcase audio slice: %s", sliced_audio)

    frames = int(config.inference.generation_frames)
    features = extract_audio_features(
        audio_path=sliced_audio,
        motion_frame_count=frames,
        motion_fps=config.data.sample_rate_hz,
        sample_rate=config.audio.sample_rate,
        hop_length=config.audio.hop_length,
    )
    audio_context = torch.tensor(features.frame_features[None], dtype=torch.float32, device=device)

    model = _load_model_for_inference(config, checkpoint_path, device)
    scheduler = DDPMScheduler(
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule=config.diffusion.beta_schedule,
    ).to(device)
    generated = generate_sliding_window(
        model=model,
        scheduler=scheduler,
        audio_context=audio_context,
        output_frames=frames,
        motion_dim=config.model.input_dim,
        window_frames=config.inference.sliding_window_frames,
        prefix_frames=config.inference.prefix_frames,
        guidance_scale=config.inference.guidance_scale,
        device=device,
        steps=config.inference.diffusion_steps,
    ).squeeze(0)
    motion_np = generated.detach().cpu().numpy().astype(np.float32)
    np.save(output_root / "stardust_0046_0101_generated_motion.npy", motion_np)

    viral_path = render_viral_motion(
        motion=motion_np,
        mp4_path=output_root / "stardust_0046_0101_viral.mp4",
        title="Music Sounds Better With You",
        fps=config.showcase.viral_fps,
        dpi=config.showcase.render_dpi,
        max_frames=frames,
    )
    research_path = render_research_motion(
        motion=motion_np,
        audio_context=features.frame_features,
        beat_indicator=features.beat_mask,
        mp4_path=output_root / "stardust_0046_0101_research.mp4",
        title="Embodied Motion Flow - Music-Conditioned SMPL Generation",
        fps=config.showcase.research_fps,
        dpi=config.showcase.render_dpi,
        max_frames=frames,
        diffusion_steps=config.diffusion.timesteps,
    )
    manifest = {
        "checkpoint": str(checkpoint_path),
        "audio_slice": str(sliced_audio),
        "generated_motion": str(output_root / "stardust_0046_0101_generated_motion.npy"),
        "viral_mp4": str(viral_path),
        "research_mp4": str(research_path),
        "guidance_scale": str(config.inference.guidance_scale),
        "ema": str(config.inference.use_ema),
        "frames": str(frames),
    }
    (output_root / "showcase_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    repo_root = _ensure_repo(args.repo_url, Path(args.workdir))
    if repo_root != Path.cwd():
        sys.path.insert(0, str(repo_root))
        os.chdir(repo_root)

    from embodied_motion_flow.config import config_to_dict, load_config
    from embodied_motion_flow.reproducibility import set_global_seed
    from embodied_motion_flow.training.engine import run_training_pipeline
    from embodied_motion_flow.utils.logging import configure_logging, get_logger

    config = _config_for_kaggle(load_config(args.config))
    if args.track:
        config = replace(config, showcase=replace(config.showcase, track_path=args.track))

    log_root = Path(config.project.output_dir) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    configure_logging(config.project.log_level, log_file=log_root / "kaggle_showcase.log")
    logger = get_logger("embodied_motion_flow.kaggle_showcase")
    logger.info("Active Kaggle showcase config: %s", json.dumps(config_to_dict(config), indent=2))
    set_global_seed(
        seed=config.reproducibility.seed,
        deterministic_torch=config.reproducibility.deterministic_torch,
        benchmark_cudnn=config.reproducibility.benchmark_cudnn,
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(config.project.output_dir) / "checkpoints" / "model.pt"
    if not args.skip_train:
        _, artifacts = run_training_pipeline(config)
        checkpoint_path = artifacts.checkpoint_path
    elif not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist for --skip-train: {checkpoint_path}")

    manifest = _run_showcase_generation(config, checkpoint_path, logger)
    logger.info("Showcase complete: %s", manifest)


if __name__ == "__main__":
    main()
