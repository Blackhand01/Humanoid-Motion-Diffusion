"""Tests for research-grade SMPL rendering utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from embodied_motion_flow.rendering.smpl_renderer import render_comparison, render_motion_preview


def test_render_motion_preview_writes_gif(tmp_path: Path) -> None:
    motion = np.zeros((8, 72), dtype=np.float32)
    beat_indicator = np.zeros((8,), dtype=np.float32)
    beat_indicator[[2, 6]] = 1.0
    paths = render_motion_preview(
        motion=motion,
        beat_indicator=beat_indicator,
        gif_path=tmp_path / "preview.gif",
        mp4_path=None,
        max_frames=3,
        dpi=60,
        fps=6,
    )
    assert paths["gif_path"].exists()


def test_render_comparison_writes_gif_with_ghosting(tmp_path: Path) -> None:
    gt = np.zeros((8, 72), dtype=np.float32)
    generated = gt.copy()
    generated[:, 20 * 3] = np.linspace(0.0, 0.3, 8)
    noisy = generated + 0.2
    paths = render_comparison(
        ground_truth=gt,
        generated=generated,
        denoising_history=[noisy],
        gif_path=tmp_path / "comparison.gif",
        mp4_path=None,
        max_frames=3,
        dpi=60,
        fps=6,
    )
    assert paths["gif_path"].exists()
