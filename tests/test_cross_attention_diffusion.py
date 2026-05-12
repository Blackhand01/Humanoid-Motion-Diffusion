"""Tests for audio-conditioned cross-attention denoiser."""

from __future__ import annotations

import torch

from embodied_motion_flow.models.cross_attention_diffusion import AudioConditionedTransformerDenoiser


def test_audio_conditioned_denoiser_shape() -> None:
    model = AudioConditionedTransformerDenoiser(
        input_dim=72,
        audio_dim=14,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        time_embedding_dim=32,
    )
    x = torch.randn(2, 16, 72)
    audio = torch.randn(2, 16, 14)
    timesteps = torch.tensor([3, 7], dtype=torch.long)
    output = model(x, timesteps, audio)
    assert output.shape == x.shape
