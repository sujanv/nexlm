"""
Milestone 39: YaRN (Yet another RoPE extensioN) Context Scaling

YaRN (Peng et al., 2023) is a context window extension method that scales
Rotary Position Embeddings (RoPE) up to 16x-64x without fine-tuning:
1. Frequency Bands:
   - High frequencies (short wavelengths < beta_slow * L): not interpolated (preserve local syntax).
   - Low frequencies (long wavelengths > beta_fast * L): fully interpolated by scale s.
   - Intermediate frequencies: smooth ramp blending between extrapolation and interpolation.
2. Attention Temperature Scaling:
   - Counteracts the entropy increase in attention distributions over long sequences:
     t = 0.1 * ln(s) + 1.0
     cos, sin scaled by 1 / sqrt(t)
"""

import math
from typing import Tuple
import torch
import torch.nn as nn


def find_correction_dim(num_rotations: float, dim: int, base: float = 10000.0, max_seq_len: int = 256) -> float:
    return (dim * math.log(max_seq_len / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float = 10000.0, max_seq_len: int = 256
) -> Tuple[int, int]:
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp = torch.clamp(linear_func, 0.0, 1.0)
    return ramp


class YaRNScaledRotaryEmbedding(nn.Module):
    """
    YaRN Rotary Position Embedding for long-context scaling.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        original_max_seq_len: int = 256,
        scale: float = 4.0,
        base: float = 10000.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        extrapolation_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = original_max_seq_len
        self.scale = scale
        self.base = base
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.extrapolation_factor = extrapolation_factor

        # 1. Base frequencies
        inv_freq_base = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # 2. Linear interpolation frequencies
        inv_freq_interp = inv_freq_base / scale

        # 3. Compute YaRN frequency blending
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_seq_len)
        half_dim = dim // 2
        inv_freq_mask = 1.0 - linear_ramp_mask(low / 2, high / 2, half_dim)

        # Blend: 0 = fully interpolated, 1 = un-interpolated
        inv_freq = inv_freq_interp * (1 - inv_freq_mask) + inv_freq_base * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 4. Temperature scaling factor for attention distribution
        # t = 0.1 * ln(s) + 1.0
        self.mscale = float(0.1 * math.log(scale) + 1.0) if scale > 1.0 else 1.0

        # Precompute cos and sin cached tables
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Apply mscale to counteract entropy increase
        self.register_buffer("cos_cached", (emb.cos() * self.mscale).unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:, :, :seq_len, :].to(device),
            self.sin_cached[:, :, :seq_len, :].to(device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_yarn_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies YaRN-scaled rotary embedding to queries and keys.
    Shapes:
        q, k: (batch_size, n_heads, seq_len, head_dim)
        cos, sin: (1, 1, seq_len, head_dim)
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
