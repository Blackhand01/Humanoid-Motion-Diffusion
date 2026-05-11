"""Tests for DDPM scheduler construction."""

from __future__ import annotations

import torch

from embodied_motion_flow.models.diffusion import DDPMScheduler


def test_linear_scheduler_construction() -> None:
    scheduler = DDPMScheduler(timesteps=100, beta_start=1e-4, beta_end=2e-2, schedule="linear")
    assert scheduler.betas.shape == (100,)
    assert torch.all(scheduler.betas[1:] >= scheduler.betas[:-1])
    assert torch.all(scheduler.alphas > 0.0)
    assert torch.all(scheduler.alphas_cumprod[1:] <= scheduler.alphas_cumprod[:-1])
    assert torch.isclose(scheduler.alphas_cumprod_prev[0], torch.tensor(1.0))
