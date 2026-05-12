"""Tests for CFG sampling and EMA utilities."""

from __future__ import annotations

import torch
from torch import nn

from embodied_motion_flow.generation.sampling import classifier_free_guidance_model_fn, generate_sliding_window
from embodied_motion_flow.models.cross_attention_diffusion import AudioConditionedTransformerDenoiser
from embodied_motion_flow.models.diffusion import DDPMScheduler
from embodied_motion_flow.training.ema import ExponentialMovingAverage


class TinyCondModel(AudioConditionedTransformerDenoiser):
    """Small audio-conditioned model with deterministic output."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor, audio_context: torch.Tensor) -> torch.Tensor:
        del timesteps
        return sample * self.weight + audio_context.mean(dim=-1, keepdim=True)


def test_ema_context_swaps_and_restores_parameters() -> None:
    model = nn.Linear(2, 2, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.5)
    original = model.weight.detach().clone()
    model.weight.data.add_(2.0)
    ema.update(model)

    trained = model.weight.detach().clone()
    with ema.average_parameters(model):
        assert not torch.allclose(model.weight, trained)
    assert torch.allclose(model.weight, trained)
    assert not torch.allclose(model.weight, original)


def test_classifier_free_guidance_dual_pass() -> None:
    model = TinyCondModel()
    sample = torch.zeros(2, 4, 1)
    timesteps = torch.zeros(2, dtype=torch.long)
    audio = torch.ones(2, 4, 3)
    guided = classifier_free_guidance_model_fn(model, audio, guidance_scale=2.0)
    output = guided(sample, timesteps)
    assert torch.allclose(output, torch.full_like(output, 2.0))


def test_sliding_window_generation_shape() -> None:
    torch.manual_seed(7)
    model = TinyCondModel()
    scheduler = DDPMScheduler(timesteps=4, beta_start=1e-4, beta_end=2e-2)
    audio = torch.zeros(1, 12, 3)
    generated = generate_sliding_window(
        model=model,
        scheduler=scheduler,
        audio_context=audio,
        output_frames=12,
        motion_dim=1,
        window_frames=6,
        prefix_frames=2,
        guidance_scale=1.0,
        device=torch.device("cpu"),
    )
    assert generated.shape == (1, 12, 1)
    assert torch.isfinite(generated).all()
