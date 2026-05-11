"""DDPM scheduler and forward/reverse diffusion operations."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class DDPMScheduler(nn.Module):
    """DDPM scheduler with linear beta schedule."""

    def __init__(self, timesteps: int, beta_start: float, beta_end: float, schedule: str = "linear") -> None:
        super().__init__()
        if timesteps <= 1:
            raise ValueError("timesteps must be > 1")
        if schedule != "linear":
            raise ValueError(f"Unsupported beta schedule: {schedule}")

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    @staticmethod
    def _validate_sample_shape(x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"Expected motion tensor [B, T, J], got {tuple(x.shape)}")

    @staticmethod
    def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        values = buffer.gather(dim=0, index=timesteps)
        return values.view(-1, 1, 1).expand(target_shape)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample integer timesteps in [0, timesteps)."""
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        self._validate_sample_shape(x0)
        if noise.shape != x0.shape:
            raise ValueError(f"Noise shape mismatch: expected {tuple(x0.shape)}, got {tuple(noise.shape)}")
        if timesteps.ndim != 1 or timesteps.shape[0] != x0.shape[0]:
            raise ValueError("timesteps must be [B] and align with batch dimension")

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def predict_x0_from_noise(self, xt: torch.Tensor, pred_noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Estimate x0 from noisy sample and predicted noise."""
        self._validate_sample_shape(xt)
        if pred_noise.shape != xt.shape:
            raise ValueError("pred_noise must match xt shape")
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, xt.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, xt.shape)
        return (xt - sqrt_one_minus * pred_noise) / (sqrt_alpha_bar + 1e-8)

    def p_mean(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Mean of reverse step p(x_{t-1} | x_t)."""
        pred_noise = model(xt, timesteps)
        beta_t = self._extract(self.betas, timesteps, xt.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, xt.shape)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, timesteps, xt.shape)
        return sqrt_recip_alpha * (xt - (beta_t / (sqrt_one_minus + 1e-8)) * pred_noise)

    def p_sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single reverse denoising step."""
        self._validate_sample_shape(xt)
        if timesteps.ndim != 1 or timesteps.shape[0] != xt.shape[0]:
            raise ValueError("timesteps must be [B] and align with batch dimension")

        model_mean = self.p_mean(model, xt, timesteps)
        variance = self._extract(self.posterior_variance, timesteps, xt.shape)
        if noise is None:
            noise = torch.randn_like(xt)
        nonzero_mask = (timesteps > 0).float().view(-1, 1, 1)
        return model_mean + nonzero_mask * torch.sqrt(torch.clamp_min(variance, 1e-12)) * noise

    @torch.no_grad()
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate trajectories from Gaussian noise."""
        xt = torch.randn(shape, device=device, dtype=torch.float32)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            xt = self.p_sample(model, xt, t)
        return xt

    @torch.no_grad()
    def reconstruct(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        start_timestep: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct clean trajectories from noisy trajectories.

        Returns:
            noisy_xt: noised input at start_timestep.
            reconstructed_x0: denoised sample after reverse steps to 0.
        """
        self._validate_sample_shape(x0)
        if start_timestep >= self.timesteps:
            raise ValueError("start_timestep must be < scheduler timesteps")

        t = torch.full((x0.shape[0],), start_timestep, device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.add_noise(x0, noise, t)
        noisy_xt = xt.clone()

        for step in reversed(range(start_timestep + 1)):
            step_t = torch.full((x0.shape[0],), step, device=x0.device, dtype=torch.long)
            # deterministic evaluation path
            xt = self.p_sample(model, xt, step_t, noise=torch.zeros_like(xt))
        return noisy_xt, xt
