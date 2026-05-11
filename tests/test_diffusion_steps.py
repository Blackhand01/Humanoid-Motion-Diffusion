"""Tests for forward and reverse diffusion steps."""

from __future__ import annotations

import torch
from torch import nn

from embodied_motion_flow.models.diffusion import DDPMScheduler


class ZeroNoiseModel(nn.Module):
    """Simple model predicting zero noise for determinism tests."""

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return torch.zeros_like(x)


def test_forward_noising_step_is_deterministic_with_fixed_noise() -> None:
    scheduler = DDPMScheduler(timesteps=50, beta_start=1e-4, beta_end=2e-2)
    x0 = torch.randn(4, 32, 12)
    noise = torch.randn_like(x0)
    t = torch.tensor([5, 10, 15, 20], dtype=torch.long)

    out1 = scheduler.add_noise(x0=x0, noise=noise, timesteps=t)
    out2 = scheduler.add_noise(x0=x0, noise=noise, timesteps=t)
    assert torch.allclose(out1, out2)


def test_reverse_denoising_step_shape_and_determinism_given_noise() -> None:
    scheduler = DDPMScheduler(timesteps=50, beta_start=1e-4, beta_end=2e-2)
    model = ZeroNoiseModel()
    xt = torch.randn(3, 24, 12)
    t = torch.tensor([30, 30, 30], dtype=torch.long)
    fixed_noise = torch.ones_like(xt) * 0.5

    out1 = scheduler.p_sample(model=model, xt=xt, timesteps=t, noise=fixed_noise)
    out2 = scheduler.p_sample(model=model, xt=xt, timesteps=t, noise=fixed_noise)
    assert out1.shape == xt.shape
    assert torch.allclose(out1, out2)
