"""
Enhancement: Rotary Position Embeddings (RoPE)

Why RoPE?
In standard transformers (like GPT-2), learned absolute positional embeddings are simply added
to token embeddings: x = tok_emb + pos_emb.
However, attention cares about *relative* distance between tokens (e.g. word 5 vs word 7 is a distance of 2),
not just their absolute indices.

RoPE rotates the Query and Key vectors in 2D coordinate pairs by angles proportional to their positions.
Property:
    <R_m q, R_n k> = q^T R_{n - m} k
The inner product depends solely on relative distance (m - n), enabling superior length generalization.
"""

from typing import Tuple
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module implemented from scratch.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"Head dimension ({dim}) must be even for RoPE rotation"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Frequency bands: theta^(-2(i-1)/dim) for i in [0, dim // 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin caches
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        # Outer product: (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate each frequency to match full head_dim: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cos, sin) slices corresponding to the sequence length starting from start_pos.
        Output shape: (1, 1, seq_len, dim)
        """
        seq_len = x.shape[-2]
        end_pos = start_pos + seq_len
        if end_pos > self.cos_cached.shape[0]:
            self._build_cache(max(end_pos, self.cos_cached.shape[0] * 2))

        cos = self.cos_cached[start_pos:end_pos].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[start_pos:end_pos].unsqueeze(0).unsqueeze(0)
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D rotation to Query and Key tensors.
    Formula: x_rot = (x * cos) + (rotate_half(x) * sin)
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
