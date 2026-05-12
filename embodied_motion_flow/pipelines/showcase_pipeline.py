"""End-to-end showcase inference and artifact packaging.

This module owns the production inference path: audio segment preparation,
EMA checkpoint loading, CFG sliding-window sampling, dual-mode rendering, and
single-archive export for Kaggle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from embodied_motion_flow.audio.audio_processor import extract_audio_features, slice_audio_segment
from embodied_motion_flow.config import ExperimentConfig
from embodied_motion_flow.generation.sampling import generate_sliding_window
from embodied_motion_flow.models.diffusion import DDPMScheduler
from embodied_motion_flow.models.factory import build_denoiser
from embodied_motion_flow.rendering.showcase_renderer import render_research_motion, render_viral_motion
from embodied_motion_flow.training.ema import ExponentialMovingAverage
from embodied_motion_flow.utils.device import resolve_device


@dataclass(frozen=True)
class ShowcaseArtifacts:
    """Files produced by a showcase inference run."""

    checkpoint: Path
    audio_slice: Path
    generated_motion: Path
    viral_mp4: Path
    research_mp4: Path
    manifest_path: Path
    guidance_scale: float
    ema: bool
    frames: int

    def as_manifest(self) -> dict[str, str]:
        """Return a JSON-serializable manifest for downstream packaging."""
        return {
            "checkpoint": str(self.checkpoint),
            "audio_slice": str(self.audio_slice),
            "generated_motion": str(self.generated_motion),
            "viral_mp4": str(self.viral_mp4),
            "viral_audio_source": str(self.audio_slice),
            "research_mp4": str(self.research_mp4),
            "guidance_scale": str(self.guidance_scale),
            "ema": str(self.ema),
            "frames": str(self.frames),
        }


def configure_showcase_runtime(
    config: ExperimentConfig,
    *,
    auto_resume: bool,
    track_path: str | Path | None = None,
) -> ExperimentConfig:
    """Apply runtime-only showcase defaults without mutating the source config."""
    training = replace(config.training, auto_resume=auto_resume, mixed_precision=True)
    device = replace(config.device, preference="auto")
    showcase = config.showcase
    if track_path is not None:
        showcase = replace(showcase, track_path=str(track_path))
    return replace(config, training=training, device=device, showcase=showcase)


def load_model_for_inference(config: ExperimentConfig, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load a denoiser checkpoint and swap in EMA weights when available."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model = build_denoiser(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    ema = ExponentialMovingAverage(model, decay=config.training.ema_decay)
    if config.inference.use_ema and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        with ema.average_parameters(model):
            ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(ema_state)

    model.eval()
    return model


def _resolve_showcase_track(config: ExperimentConfig) -> Path:
    track_path = config.showcase.track_path
    if track_path.startswith("${"):
        return Path("data/stardust.wav")
    return Path(track_path)


def run_showcase_generation(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    logger: logging.Logger | None = None,
) -> ShowcaseArtifacts:
    """Generate long-form motion and dual-mode videos from a trained checkpoint."""
    log = logger or logging.getLogger(__name__)
    device = resolve_device(config.device.preference)
    output_root = Path(config.showcase.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sliced_audio = slice_audio_segment(
        source_path=_resolve_showcase_track(config),
        output_path=output_root / "stardust_0046_0101.wav",
        start_seconds=config.showcase.clip_start_seconds,
        duration_seconds=config.showcase.clip_duration_seconds,
        sample_rate=config.audio.sample_rate,
    )
    log.info("Prepared showcase audio slice: %s", sliced_audio)

    frames = int(config.inference.generation_frames)
    features = extract_audio_features(
        audio_path=sliced_audio,
        motion_frame_count=frames,
        motion_fps=config.data.sample_rate_hz,
        sample_rate=config.audio.sample_rate,
        hop_length=config.audio.hop_length,
    )
    audio_context = torch.tensor(features.frame_features[None], dtype=torch.float32, device=device)

    model = load_model_for_inference(config, checkpoint_path, device)
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

    generated_motion = output_root / "stardust_0046_0101_generated_motion.npy"
    motion_np = generated.detach().cpu().numpy().astype(np.float32)
    np.save(generated_motion, motion_np)

    viral_path = render_viral_motion(
        motion=motion_np,
        mp4_path=output_root / "stardust_0046_0101_viral.mp4",
        title="Music Sounds Better With You",
        audio_path=sliced_audio,
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

    artifacts = ShowcaseArtifacts(
        checkpoint=Path(checkpoint_path),
        audio_slice=sliced_audio,
        generated_motion=generated_motion,
        viral_mp4=viral_path,
        research_mp4=research_path,
        manifest_path=output_root / "showcase_manifest.json",
        guidance_scale=config.inference.guidance_scale,
        ema=config.inference.use_ema,
        frames=frames,
    )
    artifacts.manifest_path.write_text(json.dumps(artifacts.as_manifest(), indent=2) + "\n", encoding="utf-8")
    return artifacts


def write_showcase_zip(
    config: ExperimentConfig,
    artifacts: ShowcaseArtifacts,
    config_path: str | Path,
    zip_path: str | Path,
) -> Path:
    """Package the Kaggle run into one downloadable archive."""
    output_root = Path(config.project.output_dir)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    patterns = (
        "showcase/*",
        "metrics/*",
        "plots/*",
        "logs/*",
        "animations/*",
        "previews/*",
        "checkpoints/model.pt",
    )
    written: set[str] = set()

    def add_file(archive: ZipFile, source: Path, archive_name: Path) -> None:
        if not source.exists() or not source.is_file() or source.resolve() == destination.resolve():
            return
        key = str(archive_name)
        if key in written:
            return
        archive.write(source, key)
        written.add(key)

    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for pattern in patterns:
            for source in sorted(output_root.glob(pattern)):
                add_file(archive, source, Path("outputs") / source.relative_to(output_root))
        add_file(archive, Path(config_path), Path(config_path).name)
        zip_manifest = artifacts.as_manifest()
        zip_manifest["archive_file_count"] = str(len(written))
        zip_manifest["archive_path"] = str(destination)
        archive.writestr("zip_manifest.json", json.dumps(zip_manifest, indent=2) + "\n")

    return destination
