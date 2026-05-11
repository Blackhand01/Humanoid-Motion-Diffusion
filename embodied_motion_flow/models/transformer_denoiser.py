"""1D Transformer denoiser for temporal trajectory diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal embeddings for integer timesteps.

    Args:
        timesteps: Tensor [batch] of integer timesteps.
        embedding_dim: Embedding dimension.
        max_period: Controls minimum frequency.
    """
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be 1D [batch], got shape {tuple(timesteps.shape)}")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half_dim, device=timesteps.device, dtype=torch.float32
    ) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    args = timesteps.float()[:, None] * frequencies[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences shaped [B, T, C]."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, C], got {tuple(x.shape)}")
        return x + self.pe[:, : x.shape[1], :]


class TemporalEncoderBlock(nn.Module):
    """Transformer encoder block with explicit multi-head self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TemporalTransformerDenoiser(nn.Module):
    """Temporal denoiser for trajectories shaped [batch, time, joints]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        time_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_encoding = PositionalEncoding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.encoder_layers = nn.ModuleList(
            [TemporalEncoderBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.latent_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def encode_latent(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return latent temporal representation [B, T, hidden]."""
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, J], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final dim {self.input_dim}, got {x.shape[-1]}")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected timesteps [B] aligned with batch size {x.shape[0]}, got {tuple(timesteps.shape)}"
            )

        h = self.input_projection(x)
        h = self.position_encoding(h)
        t_emb = sinusoidal_timestep_embedding(timesteps, self.time_embedding_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        h = h + t_emb
        for layer in self.encoder_layers:
            h = layer(h)
        return self.latent_projection(h)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict diffusion noise with output shape [B, T, J]."""
        latent = self.encode_latent(x, timesteps)
        return self.output_projection(latent)
