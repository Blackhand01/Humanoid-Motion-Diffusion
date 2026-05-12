"""Classifier-free guidance and long-form sliding-window sampling."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from embodied_motion_flow.models.cross_attention_diffusion import AudioConditionedTransformerDenoiser
from embodied_motion_flow.models.diffusion import DDPMScheduler


def classifier_free_guidance_model_fn(
    model: nn.Module,
    audio_context: torch.Tensor,
    guidance_scale: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a DDPM model function using dual-pass classifier-free guidance."""
    scale = float(guidance_scale)

    def guided(sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if not isinstance(model, AudioConditionedTransformerDenoiser):
            return model(sample, timesteps)
        cond = model(sample, timesteps, audio_context)
        if scale == 1.0:
            return cond
        uncond = model(sample, timesteps, torch.zeros_like(audio_context))
        return uncond + scale * (cond - uncond)

    return guided


@torch.no_grad()
def sample_with_cfg(
    model: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple[int, int, int],
    audio_context: torch.Tensor,
    guidance_scale: float,
    device: torch.device,
    steps: int | None = None,
) -> torch.Tensor:
    """Generate one motion window using classifier-free guidance."""
    model_fn = classifier_free_guidance_model_fn(model, audio_context, guidance_scale)
    reverse_steps = scheduler.timesteps if steps is None else min(int(steps), scheduler.timesteps)
    xt = torch.randn(shape, device=device, dtype=torch.float32)
    for step in reversed(range(reverse_steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        xt = scheduler.p_sample(model_fn, xt, t)
    return xt


@torch.no_grad()
def _sample_window_with_prefix(
    model: nn.Module,
    scheduler: DDPMScheduler,
    window_shape: tuple[int, int, int],
    audio_context: torch.Tensor,
    guidance_scale: float,
    device: torch.device,
    prefix: torch.Tensor | None,
    steps: int | None,
) -> torch.Tensor:
    """Sample a window while clamping a clean prefix through reverse diffusion."""
    model_fn = classifier_free_guidance_model_fn(model, audio_context, guidance_scale)
    reverse_steps = scheduler.timesteps if steps is None else min(int(steps), scheduler.timesteps)
    xt = torch.randn(window_shape, device=device, dtype=torch.float32)
    prefix_noise = torch.randn_like(prefix) if prefix is not None else None
    prefix_len = 0 if prefix is None else int(prefix.shape[1])
    if prefix is not None:
        start_t = torch.full((window_shape[0],), reverse_steps - 1, device=device, dtype=torch.long)
        xt[:, :prefix_len] = scheduler.add_noise(prefix, prefix_noise, start_t)

    for step in reversed(range(reverse_steps)):
        t = torch.full((window_shape[0],), step, device=device, dtype=torch.long)
        xt = scheduler.p_sample(model_fn, xt, t)
        if prefix is not None:
            if step > 0:
                next_t = torch.full((window_shape[0],), step - 1, device=device, dtype=torch.long)
                xt[:, :prefix_len] = scheduler.add_noise(prefix, prefix_noise, next_t)
            else:
                xt[:, :prefix_len] = prefix
    return xt


@torch.no_grad()
def generate_sliding_window(
    model: nn.Module,
    scheduler: DDPMScheduler,
    audio_context: torch.Tensor,
    output_frames: int,
    motion_dim: int,
    window_frames: int,
    prefix_frames: int,
    guidance_scale: float,
    device: torch.device,
    steps: int | None = None,
) -> torch.Tensor:
    """Generate long motion with overlapping denoising windows.

    The last ``prefix_frames`` of each window are reused as a clean fixed prefix
    for the next window. The returned tensor has shape ``[1, output_frames, D]``.
    """
    if audio_context.ndim != 3 or audio_context.shape[0] != 1:
        raise ValueError("audio_context must be [1, T, C] for showcase generation")
    if output_frames <= 0:
        raise ValueError("output_frames must be positive")
    if window_frames <= prefix_frames:
        raise ValueError("window_frames must be greater than prefix_frames")
    if audio_context.shape[1] < output_frames:
        raise ValueError("audio_context must cover the requested output_frames")

    model.eval()
    generated_chunks: list[torch.Tensor] = []
    total_generated = 0
    prefix: torch.Tensor | None = None
    hop = window_frames - prefix_frames

    while total_generated < output_frames:
        if prefix is None:
            new_frames = min(window_frames, output_frames - total_generated)
            audio_start = 0
            audio_end = new_frames
            window_audio = audio_context[:, audio_start:audio_end]
            window_shape = (1, new_frames, motion_dim)
            window = _sample_window_with_prefix(
                model=model,
                scheduler=scheduler,
                window_shape=window_shape,
                audio_context=window_audio,
                guidance_scale=guidance_scale,
                device=device,
                prefix=None,
                steps=steps,
            )
            generated_chunks.append(window)
            total_generated += new_frames
            prefix = window[:, -min(prefix_frames, window.shape[1]) :].detach()
            continue

        new_frames = min(hop, output_frames - total_generated)
        effective_window = prefix.shape[1] + new_frames
        audio_start = max(total_generated - prefix.shape[1], 0)
        audio_end = audio_start + effective_window
        window_audio = audio_context[:, audio_start:audio_end]
        window = _sample_window_with_prefix(
            model=model,
            scheduler=scheduler,
            window_shape=(1, effective_window, motion_dim),
            audio_context=window_audio,
            guidance_scale=guidance_scale,
            device=device,
            prefix=prefix,
            steps=steps,
        )
        generated_chunks.append(window[:, prefix.shape[1] :])
        total_generated += new_frames
        full_so_far = torch.cat(generated_chunks, dim=1)
        prefix = full_so_far[:, -prefix_frames:].detach()

    return torch.cat(generated_chunks, dim=1)[:, :output_frames]
