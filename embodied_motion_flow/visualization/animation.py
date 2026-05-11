"""Animation utilities for noisy-to-denoised humanoid trajectories."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _chain_points(origin: np.ndarray, angles: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    points = [origin]
    theta = 0.0
    current = origin.copy()
    for angle, length in zip(angles, lengths, strict=True):
        theta += float(angle)
        current = current + np.array([np.sin(theta), -np.cos(theta)], dtype=np.float32) * float(length)
        points.append(current.copy())
    return points


def _humanoid_segments(joints: np.ndarray, root_offset: np.ndarray | None = None) -> list[np.ndarray]:
    render_joints = np.clip(joints * 1.6, -1.8, 1.8)
    offset = np.zeros(2, dtype=np.float32) if root_offset is None else root_offset.astype(np.float32)
    pelvis = np.array([0.0, 0.0], dtype=np.float32) + offset
    neck = np.array([0.0, 1.0], dtype=np.float32)
    left_hip = pelvis + np.array([-0.18, 0.0], dtype=np.float32)
    right_hip = pelvis + np.array([0.18, 0.0], dtype=np.float32)
    left_shoulder = neck + np.array([-0.22, 0.0], dtype=np.float32)
    right_shoulder = neck + np.array([0.22, 0.0], dtype=np.float32)

    left_leg = _chain_points(left_hip, render_joints[[0, 1, 2]], np.array([0.55, 0.55, 0.20], dtype=np.float32))
    right_leg = _chain_points(right_hip, render_joints[[3, 4, 5]], np.array([0.55, 0.55, 0.20], dtype=np.float32))
    left_arm = _chain_points(left_shoulder, render_joints[[6, 7, 8]], np.array([0.45, 0.35, 0.14], dtype=np.float32))
    right_arm = _chain_points(right_shoulder, render_joints[[9, 10, 11]], np.array([0.45, 0.35, 0.14], dtype=np.float32))

    torso = np.stack([pelvis, neck], axis=0)
    hip_line = np.stack([left_hip, right_hip], axis=0)
    shoulder_line = np.stack([left_shoulder, right_shoulder], axis=0)
    return [
        torso,
        hip_line,
        shoulder_line,
        np.stack(left_leg, axis=0),
        np.stack(right_leg, axis=0),
        np.stack(left_arm, axis=0),
        np.stack(right_arm, axis=0),
    ]


def save_denoising_animation(
    noisy_trajectory: np.ndarray,
    denoised_trajectory: np.ndarray,
    gif_path: Path,
    mp4_path: Path,
    fps: int,
    dpi: int,
    max_frames: int,
) -> dict[str, Path]:
    """Save side-by-side humanoid animation from noise to denoised motion."""
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = min(int(noisy_trajectory.shape[0]), int(denoised_trajectory.shape[0]), max_frames)
    frame_indices = np.linspace(0, frame_count - 1, frame_count, dtype=np.int64)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
    titles = ["Noisy Trajectory", "Denoised Trajectory"]
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title)
        axis.set_xlim(-2.0, 2.0)
        axis.set_ylim(-1.6, 1.6)
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.axhline(-1.28, color="0.35", linewidth=1.0)

    line_sets: list[list[plt.Line2D]] = []
    for axis in axes:
        lines = []
        for _ in range(7):
            (line,) = axis.plot([], [], "-o", linewidth=2, markersize=4)
            lines.append(line)
        line_sets.append(lines)

    frames: list[np.ndarray] = []
    for display_idx, frame_idx in enumerate(frame_indices):
        progress = 0.0 if frame_count <= 1 else display_idx / float(frame_count - 1)
        stride_bob = 0.045 * np.sin(2.0 * np.pi * progress * 12.0)
        root_offset = np.array([-0.9 + 1.8 * progress, stride_bob], dtype=np.float32)
        noisy_segments = _humanoid_segments(noisy_trajectory[frame_idx], root_offset=root_offset)
        denoised_segments = _humanoid_segments(denoised_trajectory[frame_idx], root_offset=root_offset)
        for line, segment in zip(line_sets[0], noisy_segments, strict=True):
            line.set_data(segment[:, 0], segment[:, 1])
        for line, segment in zip(line_sets[1], denoised_segments, strict=True):
            line.set_data(segment[:, 0], segment[:, 1])
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        frames.append(frame.copy())

    plt.close(fig)
    imageio.mimsave(gif_path, frames, duration=1.0 / max(fps, 1), loop=0)
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)

    return {"gif_path": gif_path, "mp4_path": mp4_path}
