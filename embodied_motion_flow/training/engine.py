"""Training engine for Embodied-Motion-Flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from embodied_motion_flow.config import ExperimentConfig, config_to_dict
from embodied_motion_flow.data.dataset import build_dataloaders
from embodied_motion_flow.evaluation.metrics import temporal_smoothness
from embodied_motion_flow.evaluation.runner import EvaluationOutputs, evaluate_model
from embodied_motion_flow.losses.biomechanical import BiomechanicalConsistencyLoss
from embodied_motion_flow.models.diffusion import DDPMScheduler
from embodied_motion_flow.models.transformer_denoiser import TemporalTransformerDenoiser
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
    log_file: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "training_loss_plot": str(self.training_loss_plot),
            "biomechanical_loss_plot": str(self.biomechanical_loss_plot),
            "smoothness_plot": str(self.smoothness_plot),
            "animation_gif": str(self.animation_gif),
            "animation_mp4": str(self.animation_mp4),
            "log_file": str(self.log_file),
        }


def _create_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "plots": output_root / "plots",
        "animations": output_root / "animations",
        "metrics": output_root / "metrics",
        "logs": output_root / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _init_history() -> dict[str, list[float]]:
    return {
        "train_total_loss": [],
        "train_reconstruction_loss": [],
        "train_physical_loss": [],
        "train_joint_limit_loss": [],
        "train_acceleration_loss": [],
        "train_temporal_jitter_loss": [],
        "val_temporal_smoothness": [],
    }


def _validation_smoothness(
    model: TemporalTransformerDenoiser,
    scheduler: DDPMScheduler,
    val_loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    reconstruction_steps: int,
    max_batches: int = 2,
) -> float:
    model.eval()
    start_step = min(reconstruction_steps, scheduler.timesteps - 1)
    scores: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x0 = batch["motion"].to(device)
            _, reconstructed = scheduler.reconstruct(model=model, x0=x0, start_timestep=start_step)
            scores.append(temporal_smoothness(reconstructed))
            if batch_idx + 1 >= max_batches:
                break
    return float(sum(scores) / max(len(scores), 1))


def run_training_pipeline(config: ExperimentConfig) -> tuple[dict[str, float], TrainArtifacts]:
    """Train model, run evaluation, and generate outputs."""
    output_dirs = _create_output_dirs(Path(config.project.output_dir))
    log_file = output_dirs["logs"] / "train.log"
    configure_logging(config.project.log_level, log_file=log_file)
    logger = get_logger("embodied_motion_flow.training")

    device = resolve_device(config.device.preference)
    logger.info("Active config: %s", json.dumps(config_to_dict(config), indent=2))
    logger.info("Selected device: %s", device)

    data_splits = build_dataloaders(config)
    model = TemporalTransformerDenoiser(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        time_embedding_dim=config.model.time_embedding_dim,
    ).to(device)
    scheduler = DDPMScheduler(
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule=config.diffusion.beta_schedule,
    ).to(device)
    lower_limits = torch.tensor(config.data.joint_limits.lower, dtype=torch.float32, device=device)
    upper_limits = torch.tensor(config.data.joint_limits.upper, dtype=torch.float32, device=device)
    biomechanical_loss = BiomechanicalConsistencyLoss(
        lower_joint_limits=lower_limits,
        upper_joint_limits=upper_limits,
        acceleration_weight=config.loss.acceleration_weight,
        joint_limit_weight=config.loss.joint_limit_weight,
        temporal_jitter_weight=config.loss.temporal_jitter_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    use_amp = bool(config.training.mixed_precision and device.type == "cuda")
    scaler = GradScaler(device="cuda", enabled=use_amp)
    history = _init_history()

    for epoch in range(config.training.epochs):
        model.train()
        total_loss_sum = 0.0
        rec_loss_sum = 0.0
        phys_loss_sum = 0.0
        joint_sum = 0.0
        accel_sum = 0.0
        jitter_sum = 0.0
        batch_count = 0

        pbar = tqdm(
            data_splits.train_loader,
            desc=f"Epoch {epoch + 1}/{config.training.epochs}",
            leave=False,
        )
        for batch in pbar:
            x0 = batch["motion"].to(device)
            timesteps = scheduler.sample_timesteps(batch_size=x0.shape[0], device=device)
            noise = torch.randn_like(x0)
            xt = scheduler.add_noise(x0=x0, noise=noise, timesteps=timesteps)

            optimizer.zero_grad(set_to_none=True)
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with autocast(device_type=autocast_device, enabled=use_amp):
                pred_noise = model(xt, timesteps)
                reconstruction_loss = F.mse_loss(pred_noise, noise)
                x0_hat = scheduler.predict_x0_from_noise(xt=xt, pred_noise=pred_noise, timesteps=timesteps)
                phys = biomechanical_loss(x0_hat)
                total_loss = (
                    config.loss.reconstruction_weight * reconstruction_loss
                    + config.loss.lambda_phys * phys["physical_loss"]
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss_sum += float(total_loss.item())
            rec_loss_sum += float(reconstruction_loss.item())
            phys_loss_sum += float(phys["physical_loss"].item())
            joint_sum += float(phys["joint_limit_loss"].item())
            accel_sum += float(phys["acceleration_loss"].item())
            jitter_sum += float(phys["temporal_jitter_loss"].item())
            batch_count += 1

            pbar.set_postfix(
                total=f"{total_loss_sum / batch_count:.4f}",
                recon=f"{rec_loss_sum / batch_count:.4f}",
                phys=f"{phys_loss_sum / batch_count:.4f}",
            )

        history["train_total_loss"].append(total_loss_sum / max(batch_count, 1))
        history["train_reconstruction_loss"].append(rec_loss_sum / max(batch_count, 1))
        history["train_physical_loss"].append(phys_loss_sum / max(batch_count, 1))
        history["train_joint_limit_loss"].append(joint_sum / max(batch_count, 1))
        history["train_acceleration_loss"].append(accel_sum / max(batch_count, 1))
        history["train_temporal_jitter_loss"].append(jitter_sum / max(batch_count, 1))

        val_smooth = _validation_smoothness(
            model=model,
            scheduler=scheduler,
            val_loader=data_splits.val_loader,
            device=device,
            reconstruction_steps=config.evaluation.reconstruction_steps,
        )
        history["val_temporal_smoothness"].append(val_smooth)
        logger.info(
            "Epoch %d summary | total=%.6f recon=%.6f phys=%.6f val_smooth=%.6f",
            epoch + 1,
            history["train_total_loss"][-1],
            history["train_reconstruction_loss"][-1],
            history["train_physical_loss"][-1],
            history["val_temporal_smoothness"][-1],
        )

        if (epoch + 1) % config.training.save_every_epochs == 0:
            epoch_ckpt = output_dirs["checkpoints"] / f"model_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history,
                    "config": config_to_dict(config),
                },
                epoch_ckpt,
            )
            logger.info("Saved epoch checkpoint to %s", epoch_ckpt)

    checkpoint_path = output_dirs["checkpoints"] / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "config": config_to_dict(config),
        },
        checkpoint_path,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)

    plot_paths = save_training_plots(history=history, plot_dir=output_dirs["plots"])

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

    artifacts = TrainArtifacts(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        training_loss_plot=plot_paths["training_loss_plot"],
        biomechanical_loss_plot=plot_paths["biomechanical_loss_plot"],
        smoothness_plot=plot_paths["smoothness_plot"],
        animation_gif=animation_paths["gif_path"],
        animation_mp4=animation_paths["mp4_path"],
        log_file=log_file,
    )
    return eval_outputs.metrics, artifacts
