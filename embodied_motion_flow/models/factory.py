"""Model factory for configured denoisers."""

from __future__ import annotations

from torch import nn

from embodied_motion_flow.config import ExperimentConfig
from embodied_motion_flow.models.cross_attention_diffusion import AudioConditionedTransformerDenoiser
from embodied_motion_flow.models.transformer_denoiser import TemporalTransformerDenoiser


def build_denoiser(config: ExperimentConfig) -> nn.Module:
    """Build the denoising model specified by config.model.name."""
    if config.model.name == "cross_attention_transformer_diffusion":
        return AudioConditionedTransformerDenoiser(
            input_dim=config.model.input_dim,
            audio_dim=config.model.audio_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            time_embedding_dim=config.model.time_embedding_dim,
        )
    return TemporalTransformerDenoiser(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        time_embedding_dim=config.model.time_embedding_dim,
    )
