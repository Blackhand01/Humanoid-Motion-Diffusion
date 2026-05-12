"""Dual-mode long-form showcase rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from embodied_motion_flow.rendering import smpl_renderer as sr


def _to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _open_writer(path: Path, fps: int) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1)


def _audio_joint_heatmap(motion: np.ndarray, audio_context: np.ndarray) -> np.ndarray:
    """Approximate audio-to-joint saliency using per-frame correlation."""
    joint_energy = np.linalg.norm(motion.reshape(motion.shape[0], 24, 3), axis=-1)
    audio_energy = np.linalg.norm(audio_context[:, : min(audio_context.shape[-1], 12)], axis=-1)
    heatmap = np.zeros((24, audio_context.shape[-1]), dtype=np.float32)
    for joint_idx in range(24):
        joint_signal = joint_energy[:, joint_idx]
        for audio_idx in range(audio_context.shape[-1]):
            audio_signal = audio_context[:, audio_idx] if audio_idx < audio_context.shape[-1] else audio_energy
            if np.std(joint_signal) < 1e-6 or np.std(audio_signal) < 1e-6:
                heatmap[joint_idx, audio_idx] = 0.0
            else:
                heatmap[joint_idx, audio_idx] = float(np.corrcoef(joint_signal, audio_signal)[0, 1])
    return np.nan_to_num(heatmap, nan=0.0).astype(np.float32)


def render_viral_motion(
    motion: np.ndarray | torch.Tensor,
    mp4_path: str | Path,
    title: str,
    fps: int = 30,
    dpi: int = 160,
    max_frames: int | None = None,
) -> Path:
    """Render a cinematic dark-background social preview."""
    motion_np = np.asarray(_to_numpy(motion), dtype=np.float32)
    joints = sr._as_joint_sequence(motion_np)
    frame_ids = sr._frame_indices(joints.shape[0], max_frames or joints.shape[0])
    bounds = sr._bounds_for_sequences([joints])
    ground_y = float(joints.reshape(-1, 3)[:, 1].min() - 0.06)
    path = Path(mp4_path)

    with _open_writer(path, fps=fps) as writer:
        for idx, frame_idx in enumerate(frame_ids):
            fig = plt.figure(figsize=(9.0, 16.0), dpi=dpi, facecolor="#050608")
            axis = fig.add_subplot(111, projection="3d", facecolor="#050608")
            axis.set_title(title, color="#f6f4ee", fontsize=16, fontweight="bold", pad=14)
            axis.set_axis_off()
            lower, upper = bounds
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[2], upper[2])
            axis.set_zlim(lower[1], upper[1])
            axis.view_init(elev=14.0 + 4.0 * np.sin(idx / max(len(frame_ids), 1) * np.pi), azim=-58.0 + idx * 0.05)
            sr._draw_floor(axis, bounds, ground_y=ground_y)
            sr._draw_shadow(axis, joints[int(frame_idx)], ground_y)
            sr._draw_skeleton(axis, joints[int(frame_idx)], alpha=1.0, linewidth=5.2)
            writer.append_data(sr._capture_frame(fig))
            plt.close(fig)
    return path


def render_research_motion(
    motion: np.ndarray | torch.Tensor,
    audio_context: np.ndarray | torch.Tensor,
    beat_indicator: np.ndarray | torch.Tensor,
    mp4_path: str | Path,
    title: str,
    fps: int = 30,
    dpi: int = 160,
    max_frames: int | None = None,
    diffusion_steps: int | None = None,
) -> Path:
    """Render a portfolio/research diagnostic view with HUD and heatmap."""
    motion_np = np.asarray(_to_numpy(motion), dtype=np.float32)
    audio_np = np.asarray(_to_numpy(audio_context), dtype=np.float32)
    beat_np = np.asarray(_to_numpy(beat_indicator), dtype=np.float32)
    joints = sr._as_joint_sequence(motion_np)
    metrics = sr._metric_series(motion_np, beat_np)
    frame_ids = sr._frame_indices(joints.shape[0], max_frames or joints.shape[0])
    bounds = sr._bounds_for_sequences([joints])
    ground_y = float(joints.reshape(-1, 3)[:, 1].min() - 0.06)
    heatmap = _audio_joint_heatmap(motion_np, audio_np)
    total_steps = int(diffusion_steps or 0)
    path = Path(mp4_path)

    with _open_writer(path, fps=fps) as writer:
        for frame_idx in frame_ids:
            fig = plt.figure(figsize=(14.0, 8.0), dpi=dpi, facecolor="#f4f4f1")
            fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
            grid = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.4], height_ratios=[4.8, 1.1], hspace=0.12, wspace=0.10)
            motion_axis = fig.add_subplot(grid[0, 0], projection="3d")
            heat_axis = fig.add_subplot(grid[0, 1])
            beat_axis = fig.add_subplot(grid[1, :])
            sr._set_axis_style(motion_axis, "EMA + CFG Generated Motion", bounds)
            sr._draw_floor(motion_axis, bounds, ground_y=ground_y)
            sr._draw_skeleton(motion_axis, joints[int(frame_idx)])
            sr._draw_hud(motion_axis, metrics, int(frame_idx), sr.RenderThresholds(), label="Generated")
            if total_steps > 0:
                motion_axis.text2D(
                    0.02,
                    0.88,
                    f"DDPM t=0 | CFG/EMA | source steps={total_steps}",
                    transform=motion_axis.transAxes,
                    fontsize=8,
                    color="#242424",
                )
            heat_axis.imshow(heatmap, aspect="auto", cmap="magma", vmin=-1.0, vmax=1.0)
            heat_axis.set_title("Audio -> Joint Saliency", fontsize=10, fontweight="bold")
            heat_axis.set_xlabel("audio feature")
            heat_axis.set_ylabel("SMPL joint")
            heat_axis.tick_params(labelsize=7)
            sr._draw_beat_panel(beat_axis, beat_np, int(frame_idx), joints.shape[0], fps)
            writer.append_data(sr._capture_frame(fig))
            plt.close(fig)
    return path
