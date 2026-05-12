"""Cross-attention diffusion denoiser for music-conditioned SMPL motion."""

from __future__ import annotations

import torch
from torch import nn

from embodied_motion_flow.models.transformer_denoiser import PositionalEncoding, sinusoidal_timestep_embedding


class CrossAttentionEncoderBlock(nn.Module):
    """Temporal self-attention plus audio cross-attention block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, motion_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Apply self-attention over motion and cross-attention to audio context."""
        self_out, _ = self.self_attn(motion_tokens, motion_tokens, motion_tokens, need_weights=False)
        motion_tokens = self.norm_self(motion_tokens + self.dropout(self_out))
        cross_out, _ = self.cross_attn(motion_tokens, audio_tokens, audio_tokens, need_weights=False)
        motion_tokens = self.norm_cross(motion_tokens + self.dropout(cross_out))
        ffn_out = self.ffn(motion_tokens)
        return self.norm_ffn(motion_tokens + self.dropout(ffn_out))


class AudioConditionedTransformerDenoiser(nn.Module):
    """Transformer DDPM denoiser conditioned on aligned audio features.

    Args:
        input_dim: Motion feature dimension, 72 for SMPL 24x3 axis-angle.
        audio_dim: Per-frame audio feature dimension, e.g. 14 from audio_processor.

    Forward shapes:
        x: [batch, motion_time, input_dim]
        timesteps: [batch]
        audio_context: [batch, audio_time, audio_dim]
        output: [batch, motion_time, input_dim]
    """

    def __init__(
        self,
        input_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        time_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.audio_dim = audio_dim
        self.time_embedding_dim = time_embedding_dim

        self.motion_projection = nn.Linear(input_dim, hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.motion_position = PositionalEncoding(hidden_dim)
        self.audio_position = PositionalEncoding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [CrossAttentionEncoderBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, audio_context: torch.Tensor) -> torch.Tensor:
        """Predict diffusion noise conditioned on audio context."""
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,T,D], received {tuple(x.shape)}")
        if audio_context.ndim != 3:
            raise ValueError(f"Expected audio_context [B,Ta,C], received {tuple(audio_context.shape)}")
        if x.shape[0] != audio_context.shape[0]:
            raise ValueError("x and audio_context batch sizes must match")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected motion dim {self.input_dim}, received {x.shape[-1]}")
        if audio_context.shape[-1] != self.audio_dim:
            raise ValueError(f"Expected audio dim {self.audio_dim}, received {audio_context.shape[-1]}")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("timesteps must be [B] and match batch size")

        motion_tokens = self.motion_position(self.motion_projection(x))
        audio_tokens = self.audio_position(self.audio_projection(audio_context))
        time_tokens = self.time_mlp(sinusoidal_timestep_embedding(timesteps, self.time_embedding_dim)).unsqueeze(1)
        motion_tokens = motion_tokens + time_tokens

        for block in self.blocks:
            motion_tokens = block(motion_tokens, audio_tokens)
        return self.output_projection(motion_tokens)
