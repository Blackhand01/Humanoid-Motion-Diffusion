"""Plotting utilities for training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_plots(history: dict[str, list[float]], plot_dir: Path) -> dict[str, Path]:
    """Save training loss, biomechanical loss, and smoothness plots."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_total_loss"]) + 1))

    training_loss_path = plot_dir / "training_loss.png"
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_total_loss"], label="Total Loss", linewidth=2)
    plt.plot(epochs, history["train_reconstruction_loss"], label="Reconstruction Loss", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(training_loss_path, dpi=150)
    plt.close()

    biomechanical_path = plot_dir / "biomechanical_loss.png"
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_physical_loss"], label="Physical Loss", linewidth=2)
    plt.plot(epochs, history["train_joint_limit_loss"], label="Joint Limit", linewidth=1.5)
    plt.plot(epochs, history["train_acceleration_loss"], label="Acceleration", linewidth=1.5)
    plt.plot(epochs, history["train_temporal_jitter_loss"], label="Temporal Jitter", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Biomechanical Loss Components")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(biomechanical_path, dpi=150)
    plt.close()

    smoothness_path = plot_dir / "smoothness.png"
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["val_temporal_smoothness"], label="Validation Smoothness", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Velocity")
    plt.title("Temporal Smoothness")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(smoothness_path, dpi=150)
    plt.close()

    return {
        "training_loss_plot": training_loss_path,
        "biomechanical_loss_plot": biomechanical_path,
        "smoothness_plot": smoothness_path,
    }
