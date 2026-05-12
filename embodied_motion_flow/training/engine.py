"""Training engine for Embodied-Motion-Flow."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from embodied_motion_flow.config import ExperimentConfig, config_to_dict
from embodied_motion_flow.data.dataset import build_dataloaders
from embodied_motion_flow.diagnostics.failure_analysis import FailureCaseAnalyzer
from embodied_motion_flow.evaluation.metrics import (
    beat_alignment_score,
    default_smpl_joint_limits,
    joint_limit_violation_rate,
    temporal_smoothness,
    temporal_smoothness_index,
)
from embodied_motion_flow.evaluation.runner import EvaluationOutputs, evaluate_model
from embodied_motion_flow.losses.biomechanical import BiomechanicalConsistencyLoss
from embodied_motion_flow.models.cross_attention_diffusion import AudioConditionedTransformerDenoiser
from embodied_motion_flow.models.diffusion import DDPMScheduler
from embodied_motion_flow.models.factory import build_denoiser
from embodied_motion_flow.rendering.smpl_renderer import RenderThresholds, render_comparison
from embodied_motion_flow.training.ema import ExponentialMovingAverage
from embodied_motion_flow.training.schedules import build_warmup_cosine_scheduler
from embodied_motion_flow.utils.device import resolve_device
from embodied_motion_flow.utils.logging import configure_logging, get_logger
from embodied_motion_flow.visualization.animation import save_denoising_animation
from embodied_motion_flow.visualization.plots import save_training_plots


@dataclass(frozen=True)
class TrainArtifacts:
    """Paths produced by the training and evaluation pipeline."""

    checkpoint_path: Path
    metrics_path: Path
    training_loss_plot: Path
    biomechanical_loss_plot: Path
    smoothness_plot: Path
    animation_gif: Path
    animation_mp4: Path
    hero_gif: Path | None
    hero_mp4: Path | None
    log_file: Path
    evaluation_report: Path
    failure_cases_dir: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "training_loss_plot": str(self.training_loss_plot),
            "biomechanical_loss_plot": str(self.biomechanical_loss_plot),
            "smoothness_plot": str(self.smoothness_plot),
            "animation_gif": str(self.animation_gif),
            "animation_mp4": str(self.animation_mp4),
            "hero_gif": str(self.hero_gif) if self.hero_gif is not None else "",
            "hero_mp4": str(self.hero_mp4) if self.hero_mp4 is not None else "",
            "log_file": str(self.log_file),
            "evaluation_report": str(self.evaluation_report),
            "failure_cases_dir": str(self.failure_cases_dir),
        }


def _create_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "plots": output_root / "plots",
        "animations": output_root / "animations",
        "previews": output_root / "previews",
        "metrics": output_root / "metrics",
        "logs": output_root / "logs",
        "failure_cases": output_root / "failure_cases",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _reset_failure_case_outputs(failure_dir: Path) -> None:
    """Remove stale root-level failure artifacts from previous training runs."""
    for path in failure_dir.glob("epoch_*.pt"):
        path.unlink()
    failure_log = failure_dir / "failure_log.csv"
    if failure_log.exists():
        failure_log.unlink()


def _init_history() -> dict[str, list[float]]:
    return {
        "train_total_loss": [],
        "train_reconstruction_loss": [],
        "train_physical_loss": [],
        "train_joint_limit_loss": [],
        "train_acceleration_loss": [],
        "train_temporal_jitter_loss": [],
        "train_self_collision_loss": [],
        "val_temporal_smoothness": [],
        "val_tsi": [],
        "val_jlvr": [],
        "val_bas": [],
        "val_self_collision": [],
    }


def _audio_context_from_batch(
    batch: dict[str, torch.Tensor],
    motion: torch.Tensor,
    audio_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Return audio context [B, T, C], using zeros until dataset audio is available."""
    raw_audio = batch.get("audio_context")
    if isinstance(raw_audio, torch.Tensor):
        return raw_audio.to(device)
    return torch.zeros((motion.shape[0], motion.shape[1], audio_dim), dtype=motion.dtype, device=device)


def _apply_conditioning_dropout(audio_context: torch.Tensor, probability: float) -> torch.Tensor:
    """Drop complete audio conditioning vectors per sample for CFG training."""
    if probability <= 0.0:
        return audio_context
    if probability >= 1.0:
        return torch.zeros_like(audio_context)
    keep = (torch.rand((audio_context.shape[0], 1, 1), device=audio_context.device) >= probability).to(audio_context.dtype)
    return audio_context * keep


def _batch_audio_fraction(batch: dict[str, torch.Tensor]) -> float:
    """Return fraction of batch samples with nonzero real audio context."""
    raw_has_audio = batch.get("has_audio")
    if isinstance(raw_has_audio, torch.Tensor):
        return float(raw_has_audio.float().mean().item())
    raw_audio = batch.get("audio_context")
    if isinstance(raw_audio, torch.Tensor):
        per_sample_energy = raw_audio.float().abs().mean(dim=(1, 2))
        return float((per_sample_energy > 1e-8).float().mean().item())
    return 0.0


def _log_audio_coverage(logger: object, data_splits: object) -> None:
    """Log audio coverage if datasets expose AIST++ coverage metadata."""
    for split_name in ("train", "val", "test"):
        dataset = getattr(data_splits, f"{split_name}_dataset")
        coverage = getattr(dataset, "audio_coverage", None)
        if isinstance(coverage, dict):
            logger.info(
                "%s audio coverage | clips=%d with_audio=%d fraction=%.3f source_motions=%d/%d",
                split_name,
                int(coverage["clips"]),
                int(coverage["clips_with_audio"]),
                float(coverage["clip_fraction"]),
                int(coverage["source_motions_with_audio"]),
                int(coverage["source_motions"]),
            )


def _predict_noise(
    model: nn.Module,
    motion: torch.Tensor,
    timesteps: torch.Tensor,
    audio_context: torch.Tensor | None,
) -> torch.Tensor:
    """Call either unconditional or audio-conditioned denoiser."""
    if isinstance(model, AudioConditionedTransformerDenoiser):
        if audio_context is None:
            raise ValueError("audio_context is required for AudioConditionedTransformerDenoiser")
        return model(motion, timesteps, audio_context)
    return model(motion, timesteps)


def _cfg_model_fn(
    model: nn.Module,
    audio_context: torch.Tensor,
    guidance_scale: float,
) -> object:
    """Return model callable with dual-pass classifier-free guidance."""

    def model_fn(sample: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        if not isinstance(model, AudioConditionedTransformerDenoiser):
            return model(sample, step)
        cond = model(sample, step, audio_context)
        scale = float(guidance_scale)
        if scale == 1.0:
            return cond
        uncond = model(sample, step, torch.zeros_like(audio_context))
        return uncond + scale * (cond - uncond)

    return model_fn


def _grad_global_norm(parameters: object) -> float:
    """Compute global gradient L2 norm for logging."""
    total = 0.0
    for param in parameters:
        grad = getattr(param, "grad", None)
        if grad is not None:
            total += float(grad.detach().float().pow(2).sum().item())
    return total ** 0.5


def _cuda_memory_mb(device: torch.device) -> float:
    """Return allocated CUDA memory in MiB, or NaN for non-CUDA devices."""
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.memory_allocated(device) / (1024**2))


def _extract_beat_frames(batch: dict[str, torch.Tensor], motion: torch.Tensor) -> torch.Tensor | None:
    """Return beat frame ids or dense beat indicators if a dataset provides them."""
    raw_beats = batch.get("beat_frames")
    if isinstance(raw_beats, torch.Tensor):
        return raw_beats.to(motion.device)
    raw_indicator = batch.get("beat_indicator")
    if isinstance(raw_indicator, torch.Tensor):
        return raw_indicator.to(motion.device)
    return None


def _append_evaluation_report(report_path: Path, row: dict[str, float | int | str]) -> None:
    """Append one epoch-level diagnostics row to ``evaluation_report.csv``."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "split",
        "total_loss",
        "reconstruction_loss",
        "physical_loss",
        "temporal_smoothness",
        "tsi",
        "jlvr",
        "bas",
        "self_collision",
        "failure_cases",
    ]
    write_header = not report_path.exists()
    with report_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _training_joint_limits(config: ExperimentConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Use SMPL anatomical limits for 72D SMPL runs, config limits otherwise."""
    if config.data.representation.startswith("smpl") and config.data.input_dim == 72:
        return default_smpl_joint_limits(device=device)
    return (
        torch.tensor(config.data.joint_limits.lower, dtype=torch.float32, device=device),
        torch.tensor(config.data.joint_limits.upper, dtype=torch.float32, device=device),
    )


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    ema: ExponentialMovingAverage,
    history: dict[str, list[float]],
    config: ExperimentConfig,
    epoch: int,
    global_step: int,
) -> dict[str, object]:
    """Build a complete restartable training checkpoint."""
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "history": history,
        "config": config_to_dict(config),
        "epoch": int(epoch),
        "global_step": int(global_step),
    }


def _resolve_resume_path(config: ExperimentConfig, output_dirs: dict[str, Path]) -> Path | None:
    """Return checkpoint path to resume from, if configured and present."""
    if config.training.resume_from_checkpoint:
        path = Path(config.training.resume_from_checkpoint)
        return path if path.exists() else None
    if config.training.auto_resume:
        latest = output_dirs["checkpoints"] / "model.pt"
        return latest if latest.exists() else None
    return None


def _load_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    ema: ExponentialMovingAverage,
    device: torch.device,
) -> tuple[dict[str, list[float]], int, int]:
    """Restore model, optimizer, LR scheduler, scaler, EMA, history, and step counters."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    history = checkpoint.get("history", _init_history())
    if not isinstance(history, dict):
        history = _init_history()
    history.setdefault("train_audio_coverage", [])
    history.setdefault("grad_norm", [])
    history.setdefault("learning_rate", [])
    history.setdefault("amp_scale", [])
    history.setdefault("cuda_memory_mb", [])
    start_epoch = int(checkpoint.get("epoch", len(history.get("train_total_loss", []))))
    global_step = int(checkpoint.get("global_step", 0))
    return history, start_epoch, global_step


def _validation_diagnostics(
    model: nn.Module,
    scheduler: DDPMScheduler,
    val_loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    audio_dim: int,
    reconstruction_steps: int,
    guidance_scale: float,
    biomechanical_loss: BiomechanicalConsistencyLoss,
    failure_analyzer: FailureCaseAnalyzer,
    epoch: int,
    max_batches: int = 2,
) -> dict[str, float]:
    """Reconstruct validation batches and compute quantitative motion diagnostics."""
    model.eval()
    start_step = min(reconstruction_steps, scheduler.timesteps - 1)
    smooth_scores: list[float] = []
    tsi_scores: list[float] = []
    jlvr_scores: list[float] = []
    bas_scores: list[float] = []
    collision_scores: list[float] = []
    failure_count = 0
    lower_smpl, upper_smpl = default_smpl_joint_limits(device=device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x0 = batch["motion"].to(device)
            audio_context = _audio_context_from_batch(batch, x0, audio_dim, device)
            model_fn = _cfg_model_fn(model, audio_context, guidance_scale)
            _, reconstructed = scheduler.reconstruct(model=model_fn, x0=x0, start_timestep=start_step)

            tsi = temporal_smoothness_index(reconstructed, reduce="none")
            jlvr = joint_limit_violation_rate(reconstructed, lower_smpl, upper_smpl, reduce="none")
            collision = biomechanical_loss.self_collision_loss(reconstructed)
            beat_frames = _extract_beat_frames(batch, reconstructed)
            if beat_frames is not None and reconstructed.shape[-1] == 72:
                bas = beat_alignment_score(reconstructed, beat_frames, reduce="none")
                bas_scores.append(float(bas.mean().item()))

            smooth_scores.append(temporal_smoothness(reconstructed))
            tsi_scores.append(float(tsi.mean().item()))
            jlvr_scores.append(float(jlvr.mean().item()))
            collision_scores.append(float(collision.mean().item()))

            source_paths = batch.get("source_path")
            records = failure_analyzer.analyze_batch(
                motion=reconstructed,
                tsi=tsi,
                jlvr=jlvr,
                self_collision=collision,
                epoch=epoch,
                split="val",
                source_paths=source_paths if isinstance(source_paths, list) else None,
            )
            failure_count += len(records)
            if batch_idx + 1 >= max_batches:
                break

    return {
        "temporal_smoothness": float(sum(smooth_scores) / max(len(smooth_scores), 1)),
        "tsi": float(sum(tsi_scores) / max(len(tsi_scores), 1)),
        "jlvr": float(sum(jlvr_scores) / max(len(jlvr_scores), 1)),
        "bas": float(sum(bas_scores) / len(bas_scores)) if bas_scores else float("nan"),
        "self_collision": float(sum(collision_scores) / max(len(collision_scores), 1)),
        "failure_cases": float(failure_count),
    }


def run_training_pipeline(config: ExperimentConfig) -> tuple[dict[str, float], TrainArtifacts]:
    """Train model, run evaluation, and generate outputs."""
    output_dirs = _create_output_dirs(Path(config.project.output_dir))
    log_file = output_dirs["logs"] / "train.log"
    evaluation_report_path = output_dirs["metrics"] / "evaluation_report.csv"
    if evaluation_report_path.exists() and not (config.training.auto_resume or config.training.resume_from_checkpoint):
        evaluation_report_path.unlink()
    if not (config.training.auto_resume or config.training.resume_from_checkpoint):
        _reset_failure_case_outputs(output_dirs["failure_cases"])
    configure_logging(config.project.log_level, log_file=log_file)
    logger = get_logger("embodied_motion_flow.training")

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
    lower_limits, upper_limits = _training_joint_limits(config, device)
    biomechanical_loss = BiomechanicalConsistencyLoss(
        lower_joint_limits=lower_limits,
        upper_joint_limits=upper_limits,
        acceleration_weight=config.loss.acceleration_weight,
        joint_limit_weight=config.loss.joint_limit_weight,
        temporal_jitter_weight=config.loss.temporal_jitter_weight,
        self_collision_weight=config.loss.self_collision_weight,
        self_collision_margin=config.loss.self_collision_margin,
    ).to(device)
    failure_analyzer = FailureCaseAnalyzer(
        output_dir=output_dirs["failure_cases"],
        tsi_threshold=config.evaluation.tsi_failure_threshold,
        jlvr_threshold=config.evaluation.jlvr_failure_threshold,
        self_collision_threshold=config.evaluation.self_collision_failure_threshold,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    use_amp = bool(config.training.mixed_precision and device.type == "cuda")
    scaler = GradScaler(device="cuda", enabled=use_amp)
    accumulation_steps = max(1, int(config.training.accumulation_steps))
    updates_per_epoch = max(1, (len(data_splits.train_loader) + accumulation_steps - 1) // accumulation_steps)
    total_update_steps = max(1, config.training.epochs * updates_per_epoch)
    warmup_steps = (
        int(config.training.warmup_steps)
        if config.training.warmup_steps is not None
        else int(round(config.training.warmup_epochs * updates_per_epoch))
    )
    lr_scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_update_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=config.training.min_learning_rate_ratio,
    )
    ema = ExponentialMovingAverage(model, decay=config.training.ema_decay)
    history = _init_history()
    history["train_audio_coverage"] = []
    history["grad_norm"] = []
    history["learning_rate"] = []
    history["amp_scale"] = []
    history["cuda_memory_mb"] = []
    start_epoch = 0
    global_step = 0
    resume_path = _resolve_resume_path(config, output_dirs)
    if resume_path is not None:
        history, start_epoch, global_step = _load_training_state(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
        )
        logger.info("Resumed training state from %s at epoch=%d global_step=%d", resume_path, start_epoch, global_step)
    logger.info(
        "Training stability | amp=%s accumulation_steps=%d warmup_steps=%d total_update_steps=%d ema_decay=%.6f cond_dropout=%.3f",
        use_amp,
        accumulation_steps,
        warmup_steps,
        total_update_steps,
        config.training.ema_decay,
        config.training.cond_dropout,
    )
    train_started_at = time.monotonic()
    stop_requested = False
    _log_audio_coverage(logger, data_splits)

    for epoch in range(start_epoch, config.training.epochs):
        if stop_requested:
            break
        model.train()
        total_loss_sum = 0.0
        rec_loss_sum = 0.0
        phys_loss_sum = 0.0
        joint_sum = 0.0
        accel_sum = 0.0
        jitter_sum = 0.0
        collision_sum = 0.0
        audio_fraction_sum = 0.0
        batch_count = 0

        pbar = tqdm(
            data_splits.train_loader,
            desc=f"Epoch {epoch + 1}/{config.training.epochs}",
            leave=False,
        )
        optimizer.zero_grad(set_to_none=True)
        epoch_grad_norms: list[float] = []
        for step_idx, batch in enumerate(pbar):
            x0 = batch["motion"].to(device)
            audio_context = _audio_context_from_batch(batch, x0, config.model.audio_dim, device)
            audio_fraction_sum += _batch_audio_fraction(batch)
            train_audio_context = _apply_conditioning_dropout(audio_context, config.training.cond_dropout)
            timesteps = scheduler.sample_timesteps(batch_size=x0.shape[0], device=device)
            noise = torch.randn_like(x0)
            xt = scheduler.add_noise(x0=x0, noise=noise, timesteps=timesteps)

            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with autocast(device_type=autocast_device, enabled=use_amp):
                pred_noise = _predict_noise(model, xt, timesteps, train_audio_context)
                reconstruction_loss = F.mse_loss(pred_noise, noise)
                x0_hat = scheduler.predict_x0_from_noise(xt=xt, pred_noise=pred_noise, timesteps=timesteps)
                phys = biomechanical_loss(x0_hat)
                total_loss = (
                    config.loss.reconstruction_weight * reconstruction_loss
                    + config.loss.lambda_phys * phys["physical_loss"]
                )
                backward_loss = total_loss / accumulation_steps

            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, batch {step_idx + 1}: {float(total_loss.item())}")
            scaler.scale(backward_loss).backward()
            should_step = (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(data_splits.train_loader)
            grad_norm = float("nan")
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = _grad_global_norm(model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                ema.update(model)
                global_step += 1
                epoch_grad_norms.append(grad_norm)
                if config.training.log_every_steps > 0 and global_step % config.training.log_every_steps == 0:
                    logger.info(
                        "Step %d | epoch=%d lr=%.6e grad_norm=%.4f amp_scale=%.1f cuda_mem_mb=%.1f",
                        global_step,
                        epoch + 1,
                        optimizer.param_groups[0]["lr"],
                        grad_norm,
                        float(scaler.get_scale()) if use_amp else 1.0,
                        _cuda_memory_mb(device),
                    )

            total_loss_sum += float(total_loss.item())
            rec_loss_sum += float(reconstruction_loss.item())
            phys_loss_sum += float(phys["physical_loss"].item())
            joint_sum += float(phys["joint_limit_loss"].item())
            accel_sum += float(phys["acceleration_loss"].item())
            jitter_sum += float(phys["temporal_jitter_loss"].item())
            collision_sum += float(phys["self_collision_loss"].item())
            batch_count += 1

            pbar.set_postfix(
                total=f"{total_loss_sum / batch_count:.4f}",
                recon=f"{rec_loss_sum / batch_count:.4f}",
                phys=f"{phys_loss_sum / batch_count:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )
            if config.training.max_duration_seconds is not None:
                elapsed = time.monotonic() - train_started_at
                if elapsed >= config.training.max_duration_seconds:
                    logger.info(
                        "Stopping training after %.1f seconds due to training.max_duration_seconds.",
                        elapsed,
                    )
                    stop_requested = True
                    break

        history["train_total_loss"].append(total_loss_sum / max(batch_count, 1))
        history["train_reconstruction_loss"].append(rec_loss_sum / max(batch_count, 1))
        history["train_physical_loss"].append(phys_loss_sum / max(batch_count, 1))
        history["train_joint_limit_loss"].append(joint_sum / max(batch_count, 1))
        history["train_acceleration_loss"].append(accel_sum / max(batch_count, 1))
        history["train_temporal_jitter_loss"].append(jitter_sum / max(batch_count, 1))
        history["train_self_collision_loss"].append(collision_sum / max(batch_count, 1))
        history["train_audio_coverage"].append(audio_fraction_sum / max(batch_count, 1))
        history["grad_norm"].append(float(sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1)))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["amp_scale"].append(float(scaler.get_scale()) if use_amp else 1.0)
        history["cuda_memory_mb"].append(_cuda_memory_mb(device))

        with ema.average_parameters(model) if config.inference.use_ema else torch.no_grad():
            val_metrics = _validation_diagnostics(
                model=model,
                scheduler=scheduler,
                val_loader=data_splits.val_loader,
                device=device,
                audio_dim=config.model.audio_dim,
                reconstruction_steps=config.evaluation.reconstruction_steps,
                guidance_scale=config.inference.guidance_scale,
                biomechanical_loss=biomechanical_loss,
                failure_analyzer=failure_analyzer,
                epoch=epoch + 1,
            )
        history["val_temporal_smoothness"].append(val_metrics["temporal_smoothness"])
        history["val_tsi"].append(val_metrics["tsi"])
        history["val_jlvr"].append(val_metrics["jlvr"])
        history["val_bas"].append(val_metrics["bas"])
        history["val_self_collision"].append(val_metrics["self_collision"])
        _append_evaluation_report(
            evaluation_report_path,
            {
                "epoch": epoch + 1,
                "split": "val",
                "total_loss": history["train_total_loss"][-1],
                "reconstruction_loss": history["train_reconstruction_loss"][-1],
                "physical_loss": history["train_physical_loss"][-1],
                "temporal_smoothness": history["val_temporal_smoothness"][-1],
                "tsi": history["val_tsi"][-1],
                "jlvr": history["val_jlvr"][-1],
                "bas": history["val_bas"][-1],
                "self_collision": history["val_self_collision"][-1],
                "failure_cases": int(val_metrics["failure_cases"]),
            },
        )
        logger.info(
            "Epoch %d summary | total=%.6f recon=%.6f phys=%.6f val_smooth=%.6f tsi=%.6f jlvr=%.6f bas=%.6f train_audio=%.3f grad_norm=%.4f lr=%.6e amp_scale=%.1f cuda_mem_mb=%.1f failures=%d",
            epoch + 1,
            history["train_total_loss"][-1],
            history["train_reconstruction_loss"][-1],
            history["train_physical_loss"][-1],
            history["val_temporal_smoothness"][-1],
            history["val_tsi"][-1],
            history["val_jlvr"][-1],
            history["val_bas"][-1],
            history["train_audio_coverage"][-1],
            history["grad_norm"][-1],
            history["learning_rate"][-1],
            history["amp_scale"][-1],
            history["cuda_memory_mb"][-1],
            int(val_metrics["failure_cases"]),
        )

        if (epoch + 1) % config.training.save_every_epochs == 0:
            epoch_ckpt = output_dirs["checkpoints"] / f"model_epoch_{epoch + 1}.pt"
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    ema=ema,
                    history=history,
                    config=config,
                    epoch=epoch + 1,
                    global_step=global_step,
                )
                | {"scheduler_state_dict": scheduler.state_dict()},
                epoch_ckpt,
            )
            logger.info("Saved epoch checkpoint to %s", epoch_ckpt)

    checkpoint_path = output_dirs["checkpoints"] / "model.pt"
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            ema=ema,
            history=history,
            config=config,
            epoch=len(history["train_total_loss"]),
            global_step=global_step,
        )
        | {"scheduler_state_dict": scheduler.state_dict()},
        checkpoint_path,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)

    plot_paths = save_training_plots(history=history, plot_dir=output_dirs["plots"])

    with ema.average_parameters(model) if config.inference.use_ema else torch.no_grad():
        eval_outputs: EvaluationOutputs = evaluate_model(
            config=config,
            model=model,
            scheduler=scheduler,
            dataloader=data_splits.test_loader,
            biomechanical_loss=biomechanical_loss,
            device=device,
        )
    smoothness_series = history["val_temporal_smoothness"]
    eval_outputs.metrics["validation_smoothness_last_epoch"] = smoothness_series[-1] if smoothness_series else float("nan")

    metrics_path = output_dirs["metrics"] / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_outputs.metrics, handle, indent=2)
    logger.info("Saved evaluation metrics to %s", metrics_path)

    animation_paths = save_denoising_animation(
        noisy_trajectory=eval_outputs.noisy_example,
        denoised_trajectory=eval_outputs.denoised_example,
        gif_path=output_dirs["animations"] / "denoising.gif",
        mp4_path=output_dirs["animations"] / "trajectory.mp4",
        fps=config.visualization.fps,
        dpi=config.visualization.dpi,
        max_frames=config.visualization.max_frames,
    )
    logger.info("Saved animations to %s and %s", animation_paths["gif_path"], animation_paths["mp4_path"])

    hero_gif: Path | None = None
    hero_mp4: Path | None = None
    if eval_outputs.reference_example is not None and eval_outputs.reference_example.ndim == 2 and eval_outputs.reference_example.shape[-1] == 72:
        hero_gif = output_dirs["previews"] / "hero_validation.gif"
        hero_mp4 = output_dirs["previews"] / "hero_validation.mp4"
        hero_paths = render_comparison(
            ground_truth=eval_outputs.reference_example,
            generated=eval_outputs.denoised_example,
            gif_path=hero_gif,
            mp4_path=hero_mp4,
            beat_indicator=eval_outputs.beat_indicator_example,
            denoising_history=[eval_outputs.noisy_example],
            title=f"Validation Hero Preview - {Path(eval_outputs.source_id).stem}",
            thresholds=RenderThresholds(
                tsi=config.evaluation.tsi_failure_threshold,
                jlvr=config.evaluation.jlvr_failure_threshold,
            ),
            fps=config.visualization.fps,
            dpi=config.visualization.dpi,
            max_frames=config.visualization.max_frames,
        )
        logger.info("Saved hero preview to %s", hero_paths)

    artifacts = TrainArtifacts(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        training_loss_plot=plot_paths["training_loss_plot"],
        biomechanical_loss_plot=plot_paths["biomechanical_loss_plot"],
        smoothness_plot=plot_paths["smoothness_plot"],
        animation_gif=animation_paths["gif_path"],
        animation_mp4=animation_paths["mp4_path"],
        hero_gif=hero_gif,
        hero_mp4=hero_mp4,
        log_file=log_file,
        evaluation_report=evaluation_report_path,
        failure_cases_dir=output_dirs["failure_cases"],
    )
    return eval_outputs.metrics, artifacts
