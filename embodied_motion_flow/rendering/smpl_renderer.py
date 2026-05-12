"""Research-grade SMPL 24-joint motion renderer.

The renderer is intentionally Matplotlib/ImageIO based. It runs on local Mac,
Colab, and Kaggle without GPU-specific graphics dependencies, while still
producing publication-quality diagnostic previews with floor grids, HUD metrics,
beat alignment overlays, comparison views, and denoising ghosting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from embodied_motion_flow.evaluation.metrics import (
    beat_alignment_score,
    default_smpl_joint_limits,
    joint_limit_violation_rate,
    temporal_smoothness_index,
)
from embodied_motion_flow.visualization.smpl_render import SMPL_JOINT_NAMES, SMPL_PARENTS, smpl_motion_to_joints


BODY_GROUPS: dict[str, tuple[int, ...]] = {
    "spine": (0, 3, 6, 9, 12, 15),
    "left_leg": (0, 1, 4, 7, 10),
    "right_leg": (0, 2, 5, 8, 11),
    "left_arm": (12, 13, 16, 18, 20, 22),
    "right_arm": (12, 14, 17, 19, 21, 23),
}
BODY_COLORS = {
    "spine": "#2f4f4f",
    "left_leg": "#2878b5",
    "right_leg": "#6f4aa8",
    "left_arm": "#c4493d",
    "right_arm": "#d7922d",
}
SEGMENT_LOOKUP: dict[tuple[int, int], str] = {}
for group_name, chain in BODY_GROUPS.items():
    for parent, child in zip(chain[:-1], chain[1:], strict=True):
        SEGMENT_LOOKUP[(parent, child)] = group_name
        SEGMENT_LOOKUP[(child, parent)] = group_name


@dataclass(frozen=True)
class RenderThresholds:
    """Critical diagnostic thresholds used by the HUD."""

    tsi: float = 0.25
    jlvr: float = 0.02
    bas_min: float = 0.15


@dataclass(frozen=True)
class _MetricSeries:
    tsi: np.ndarray
    jlvr: np.ndarray
    bas: np.ndarray


def _to_numpy(array: np.ndarray | torch.Tensor | Sequence[float] | None) -> np.ndarray | None:
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _sequence_id(source: Any, fallback: str) -> str:
    if source is None:
        return fallback
    stem = Path(str(source)).stem
    return stem if stem else fallback


def _as_joint_sequence(
    motion: np.ndarray | torch.Tensor,
    translation: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Convert a motion sequence to approximate joint centers ``[T,24,3]``."""
    arr = np.asarray(_to_numpy(motion), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (24, 3):
        joints = arr.copy()
    elif arr.ndim == 2 and arr.shape[-1] == 72:
        joints = smpl_motion_to_joints(arr)
    else:
        raise ValueError(f"Expected motion [T,72] or joints [T,24,3], received {arr.shape}")

    trans = _to_numpy(translation)
    if trans is not None:
        trans_arr = np.asarray(trans, dtype=np.float32)
        if trans_arr.ndim != 2 or trans_arr.shape[-1] != 3:
            raise ValueError(f"translation must be [T,3], received {trans_arr.shape}")
        count = min(joints.shape[0], trans_arr.shape[0])
        joints = joints[:count] + trans_arr[:count, None, :]
    return joints.astype(np.float32)


def _metric_series(
    motion: np.ndarray | torch.Tensor,
    beat_indicator: np.ndarray | torch.Tensor | None,
    lower_limits: torch.Tensor | None = None,
    upper_limits: torch.Tensor | None = None,
) -> _MetricSeries:
    """Compute cumulative per-frame HUD metrics for a motion sequence."""
    motion_np = np.asarray(_to_numpy(motion), dtype=np.float32)
    if motion_np.ndim == 3 and motion_np.shape[-2:] == (24, 3):
        flat = motion_np.reshape(motion_np.shape[0], 72)
    elif motion_np.ndim == 2:
        flat = motion_np
    else:
        raise ValueError(f"Expected motion [T,72] or [T,24,3], received {motion_np.shape}")

    tensor = torch.tensor(flat[None], dtype=torch.float32)
    if lower_limits is None or upper_limits is None:
        lower_limits, upper_limits = default_smpl_joint_limits()
    total_frames = flat.shape[0]
    tsi_values = np.zeros(total_frames, dtype=np.float32)
    jlvr_values = np.zeros(total_frames, dtype=np.float32)
    bas_values = np.full(total_frames, np.nan, dtype=np.float32)

    beats_np = _to_numpy(beat_indicator)
    beat_tensor: torch.Tensor | None = None
    if beats_np is not None:
        beat_arr = np.asarray(beats_np, dtype=np.float32)
        if beat_arr.ndim == 1:
            beat_tensor = torch.tensor(beat_arr[None], dtype=torch.float32)
        elif beat_arr.ndim == 2:
            beat_tensor = torch.tensor(beat_arr[:1], dtype=torch.float32)

    for frame_idx in range(total_frames):
        end = frame_idx + 1
        window = tensor[:, :end]
        if end >= 3:
            tsi_values[frame_idx] = float(temporal_smoothness_index(window, reduce="mean").item())
        jlvr_values[frame_idx] = float(joint_limit_violation_rate(window, lower_limits, upper_limits, reduce="mean").item())
        if beat_tensor is not None and end >= 3:
            bas_values[frame_idx] = float(beat_alignment_score(window, beat_tensor[:, :end], reduce="mean").item())
    return _MetricSeries(tsi=tsi_values, jlvr=jlvr_values, bas=bas_values)


def _frame_indices(total_frames: int, max_frames: int) -> np.ndarray:
    frame_count = min(total_frames, max(1, max_frames))
    return np.linspace(0, total_frames - 1, frame_count, dtype=np.int64)


def _set_axis_style(axis: Any, title: str, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    """Apply consistent 3D camera, limits, pane colors, and labels."""
    axis.set_title(title, fontsize=11, fontweight="bold", pad=8)
    axis.set_xlabel("x", labelpad=4)
    axis.set_ylabel("depth", labelpad=4)
    axis.set_zlabel("height", labelpad=4)
    lower, upper = bounds
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[2], upper[2])
    axis.set_zlim(lower[1], upper[1])
    axis.view_init(elev=18.0, azim=-64.0)
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.96, 0.96, 0.94, 1.0))
        pane.set_edgecolor((0.82, 0.82, 0.80, 1.0))


def _draw_floor(axis: Any, bounds: tuple[np.ndarray, np.ndarray], ground_y: float) -> None:
    """Draw a neutral floor plane grid in x-depth coordinates."""
    lower, upper = bounds
    xs = np.linspace(lower[0], upper[0], 9)
    zs = np.linspace(lower[2], upper[2], 9)
    for x in xs:
        axis.plot([x, x], [lower[2], upper[2]], [ground_y, ground_y], color="0.78", linewidth=0.55, alpha=0.7)
    for z in zs:
        axis.plot([lower[0], upper[0]], [z, z], [ground_y, ground_y], color="0.78", linewidth=0.55, alpha=0.7)


def _draw_shadow(axis: Any, joints: np.ndarray, ground_y: float) -> None:
    shadow = joints.copy()
    shadow[:, 1] = ground_y
    axis.scatter(shadow[:, 0], shadow[:, 2], shadow[:, 1], s=8, color="0.1", alpha=0.10, depthshade=False)


def _draw_skeleton(axis: Any, joints: np.ndarray, alpha: float = 1.0, linewidth: float = 3.0) -> None:
    """Draw one SMPL skeleton with segment-specific colors."""
    _draw_shadow(axis, joints, ground_y=float(np.min(joints[:, 1]) - 0.03))
    for child_idx, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx < 0:
            continue
        segment = joints[[parent_idx, child_idx]]
        group = SEGMENT_LOOKUP.get((int(parent_idx), int(child_idx)), "spine")
        axis.plot(
            segment[:, 0],
            segment[:, 2],
            segment[:, 1],
            "-",
            color=BODY_COLORS[group],
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )
    axis.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=16, color="#1f1f1f", alpha=min(1.0, alpha + 0.1), depthshade=True)


def _draw_ghosts(axis: Any, ghost_joints: Sequence[np.ndarray], frame_idx: int, max_ghosts: int = 5) -> None:
    if not ghost_joints:
        return
    selected = list(ghost_joints)[-max_ghosts:]
    for ghost_idx, joints in enumerate(selected):
        alpha = 0.08 + 0.08 * ghost_idx
        _draw_skeleton(axis, joints[frame_idx], alpha=alpha, linewidth=1.2)


def _hud_color(value: float, threshold: float, higher_is_bad: bool = True) -> str:
    if np.isnan(value):
        return "#4a4a4a"
    failed = value > threshold if higher_is_bad else value < threshold
    return "#c1272d" if failed else "#1f4e3d"


def _draw_hud(axis: Any, metrics: _MetricSeries, frame_idx: int, thresholds: RenderThresholds, label: str) -> None:
    tsi = float(metrics.tsi[frame_idx])
    jlvr = float(metrics.jlvr[frame_idx])
    bas = float(metrics.bas[frame_idx])
    axis.text2D(0.02, 0.96, label, transform=axis.transAxes, fontsize=10, fontweight="bold", color="#222222")
    axis.text2D(0.02, 0.90, f"TSI {tsi:0.3f}", transform=axis.transAxes, fontsize=8.5, color=_hud_color(tsi, thresholds.tsi))
    axis.text2D(0.02, 0.85, f"JLVR {jlvr:0.3f}", transform=axis.transAxes, fontsize=8.5, color=_hud_color(jlvr, thresholds.jlvr))
    bas_text = "BAS n/a" if np.isnan(bas) else f"BAS {bas:0.3f}"
    axis.text2D(0.02, 0.80, bas_text, transform=axis.transAxes, fontsize=8.5, color=_hud_color(bas, thresholds.bas_min, higher_is_bad=False))


def _draw_beat_panel(axis: Any, beat_indicator: np.ndarray | None, frame_idx: int, total_frames: int, fps: int) -> None:
    axis.set_facecolor("#f8f8f6")
    axis.set_xlim(0, max(total_frames - 1, 1))
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlabel("time (s)")
    ticks = np.linspace(0, max(total_frames - 1, 1), 6)
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{tick / max(fps, 1):.1f}" for tick in ticks])
    axis.grid(axis="x", alpha=0.20)
    if beat_indicator is not None:
        beats = np.asarray(beat_indicator).reshape(-1)
        beat_ids = np.flatnonzero(beats[:total_frames] > 0.0)
        for beat in beat_ids:
            axis.axvline(int(beat), color="#d7922d", linewidth=1.2, alpha=0.65)
    else:
        axis.text(0.5, 0.5, "beat-line unavailable", transform=axis.transAxes, ha="center", va="center", color="0.35")
    axis.axvline(frame_idx, color="#1f1f1f", linewidth=2.0)


def _capture_frame(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()


def _write_frames(frames: Sequence[np.ndarray], gif_path: Path | None, mp4_path: Path | None, fps: int) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if gif_path is not None:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, list(frames), duration=1.0 / max(fps, 1), loop=0)
        outputs["gif_path"] = gif_path
    if mp4_path is not None:
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(mp4_path, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        outputs["mp4_path"] = mp4_path
    return outputs


def _bounds_for_sequences(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_joints = np.concatenate([sequence.reshape(-1, 3) for sequence in sequences], axis=0)
    lower = all_joints.min(axis=0)
    upper = all_joints.max(axis=0)
    center = (lower + upper) * 0.5
    radius = max(float((upper - lower).max()) * 0.58, 0.8)
    floor_y = float(lower[1] - 0.06)
    lower = center - radius
    upper = center + radius
    lower[1] = min(lower[1], floor_y)
    return lower.astype(np.float32), upper.astype(np.float32)


def render_motion_preview(
    motion: np.ndarray | torch.Tensor,
    gif_path: str | Path | None = None,
    mp4_path: str | Path | None = None,
    translation: np.ndarray | torch.Tensor | None = None,
    beat_indicator: np.ndarray | torch.Tensor | None = None,
    title: str = "SMPL Motion Preview",
    thresholds: RenderThresholds = RenderThresholds(),
    fps: int = 24,
    max_frames: int = 180,
    dpi: int = 120,
) -> dict[str, Path]:
    """Render one SMPL sequence with floor grid, HUD, and beat-line panel."""
    joints = _as_joint_sequence(motion, translation=translation)
    metrics = _metric_series(motion, beat_indicator=beat_indicator)
    beat_np = _to_numpy(beat_indicator)
    frame_ids = _frame_indices(joints.shape[0], max_frames)
    bounds = _bounds_for_sequences([joints])
    ground_y = float(joints.reshape(-1, 3)[:, 1].min() - 0.06)

    frames: list[np.ndarray] = []
    for frame_idx in frame_ids:
        fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi, facecolor="#f4f4f1")
        grid = fig.add_gridspec(2, 1, height_ratios=[5.0, 1.0], hspace=0.08)
        axis = fig.add_subplot(grid[0], projection="3d")
        beat_axis = fig.add_subplot(grid[1])
        _set_axis_style(axis, title, bounds)
        _draw_floor(axis, bounds, ground_y=ground_y)
        _draw_skeleton(axis, joints[frame_idx])
        _draw_hud(axis, metrics, int(frame_idx), thresholds, label=Path(title).stem)
        _draw_beat_panel(beat_axis, beat_np, int(frame_idx), joints.shape[0], fps)
        frames.append(_capture_frame(fig))
        plt.close(fig)

    return _write_frames(
        frames,
        gif_path=Path(gif_path) if gif_path is not None else None,
        mp4_path=Path(mp4_path) if mp4_path is not None else None,
        fps=fps,
    )


def render_comparison(
    ground_truth: np.ndarray | torch.Tensor,
    generated: np.ndarray | torch.Tensor,
    gif_path: str | Path | None = None,
    mp4_path: str | Path | None = None,
    ground_truth_translation: np.ndarray | torch.Tensor | None = None,
    generated_translation: np.ndarray | torch.Tensor | None = None,
    beat_indicator: np.ndarray | torch.Tensor | None = None,
    denoising_history: Sequence[np.ndarray | torch.Tensor] | None = None,
    title: str = "Ground Truth vs Generated Motion",
    thresholds: RenderThresholds = RenderThresholds(),
    fps: int = 24,
    max_frames: int = 180,
    dpi: int = 120,
) -> dict[str, Path]:
    """Render split-screen Ground Truth vs Generated Motion with ghosting.

    Args:
        denoising_history: Optional sequence of generated intermediate motions
            ``[T,72]`` or joints ``[T,24,3]``. Earlier entries are rendered as
            translucent ghosts behind the final generated pose.
    """
    gt_joints = _as_joint_sequence(ground_truth, translation=ground_truth_translation)
    gen_joints = _as_joint_sequence(generated, translation=generated_translation)
    total_frames = min(gt_joints.shape[0], gen_joints.shape[0])
    gt_joints = gt_joints[:total_frames]
    gen_joints = gen_joints[:total_frames]
    ghost_joints = [
        _as_joint_sequence(history_item)[:total_frames]
        for history_item in (denoising_history or [])
    ]
    gt_metrics = _metric_series(ground_truth, beat_indicator=beat_indicator)
    gen_metrics = _metric_series(generated, beat_indicator=beat_indicator)
    beat_np = _to_numpy(beat_indicator)
    frame_ids = _frame_indices(total_frames, max_frames)
    bounds = _bounds_for_sequences([gt_joints, gen_joints, *ghost_joints])
    ground_y = float(min(gt_joints.reshape(-1, 3)[:, 1].min(), gen_joints.reshape(-1, 3)[:, 1].min()) - 0.06)

    frames: list[np.ndarray] = []
    for frame_idx in frame_ids:
        fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi, facecolor="#f4f4f1")
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        grid = fig.add_gridspec(2, 2, height_ratios=[5.0, 1.0], hspace=0.10, wspace=0.04)
        gt_axis = fig.add_subplot(grid[0, 0], projection="3d")
        gen_axis = fig.add_subplot(grid[0, 1], projection="3d")
        beat_axis = fig.add_subplot(grid[1, :])
        for axis, panel_title in ((gt_axis, "Ground Truth"), (gen_axis, "Generated")):
            _set_axis_style(axis, panel_title, bounds)
            _draw_floor(axis, bounds, ground_y=ground_y)
        _draw_skeleton(gt_axis, gt_joints[frame_idx])
        _draw_ghosts(gen_axis, ghost_joints, int(frame_idx))
        _draw_skeleton(gen_axis, gen_joints[frame_idx])
        _draw_hud(gt_axis, gt_metrics, int(frame_idx), thresholds, label="GT")
        _draw_hud(gen_axis, gen_metrics, int(frame_idx), thresholds, label="Generated")
        _draw_beat_panel(beat_axis, beat_np, int(frame_idx), total_frames, fps)
        frames.append(_capture_frame(fig))
        plt.close(fig)

    return _write_frames(
        frames,
        gif_path=Path(gif_path) if gif_path is not None else None,
        mp4_path=Path(mp4_path) if mp4_path is not None else None,
        fps=fps,
    )


def render_batch_previews(
    motions: np.ndarray | torch.Tensor,
    output_dir: str | Path,
    sequence_ids: Sequence[str] | None = None,
    generated_motions: np.ndarray | torch.Tensor | None = None,
    beat_indicators: np.ndarray | torch.Tensor | None = None,
    fps: int = 24,
    max_frames: int = 180,
    dpi: int = 120,
    save_mp4: bool = True,
) -> list[dict[str, Path]]:
    """Render GIF/MP4 previews for an entire test batch.

    Files are named with the AIST++ sequence id, for example
    ``gBR_sBM_cAll_d04_mBR0_ch01_preview.gif``.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    motion_np = np.asarray(_to_numpy(motions), dtype=np.float32)
    if motion_np.ndim != 3:
        raise ValueError(f"motions must be [B,T,D], received {motion_np.shape}")
    generated_np = np.asarray(_to_numpy(generated_motions), dtype=np.float32) if generated_motions is not None else None
    beat_np = _to_numpy(beat_indicators)
    outputs: list[dict[str, Path]] = []
    for batch_idx in range(motion_np.shape[0]):
        sequence_id = _sequence_id(sequence_ids[batch_idx] if sequence_ids is not None else None, f"sequence_{batch_idx:04d}")
        gif_path = output_root / f"{sequence_id}_preview.gif"
        mp4_path = output_root / f"{sequence_id}_preview.mp4" if save_mp4 else None
        beats = None if beat_np is None else beat_np[batch_idx]
        if generated_np is not None:
            outputs.append(
                render_comparison(
                    ground_truth=motion_np[batch_idx],
                    generated=generated_np[batch_idx],
                    gif_path=gif_path,
                    mp4_path=mp4_path,
                    beat_indicator=beats,
                    title=f"{sequence_id} Preview",
                    fps=fps,
                    max_frames=max_frames,
                    dpi=dpi,
                )
            )
        else:
            outputs.append(
                render_motion_preview(
                    motion=motion_np[batch_idx],
                    gif_path=gif_path,
                    mp4_path=mp4_path,
                    beat_indicator=beats,
                    title=f"{sequence_id} Preview",
                    fps=fps,
                    max_frames=max_frames,
                    dpi=dpi,
                )
            )
    return outputs
