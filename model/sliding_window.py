"""
Enhancement: Sliding Window Attention (SWA) with Rolling KV Cache (Mistral Architecture)

Why Sliding Window Attention?
In standard full causal attention, each token attends to all previous tokens:
    Complexity = O(T^2) compute, O(T) KV cache memory.
For long contexts (e.g. 32k or 128k tokens), the KV cache exhausts GPU VRAM.

Sliding Window Attention (Beltagy et al., 2020; Mistral 7B, 2023) restricts attention
to a local window of size W:
    Token at index i attends only to tokens in range [max(0, i - W + 1), i].

Key Benefits:
1. Receptive field compounding: In an L-layer model, the effective receptive field
   at the final layer is L * W tokens!
2. Fixed-size Rolling Buffer: The KV cache is implemented as a circular buffer of
   fixed size W, capping memory strictly at O(W) rather than unbounded O(T).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RollingKVCache:
    """
    Fixed-size circular buffer for Key and Value tensors.
    Memory footprint is constant O(W) regardless of generation length.
    """

    def __init__(self, window_size: int, n_heads: int, head_dim: int, batch_size: int = 1):
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.k_buffer: Optional[torch.Tensor] = None
        self.v_buffer: Optional[torch.Tensor] = None
        self.num_tokens = 0

    def init_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.k_buffer = torch.zeros(
            (self.batch_size, self.n_heads, self.window_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_buffer = torch.zeros(
            (self.batch_size, self.n_heads, self.window_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        self.num_tokens = 0

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inserts new single-step Key and Value into the circular buffer.
        Returns the active valid slice of Keys and Values.
        """
        if self.k_buffer is None:
            self.init_buffers(device=k_new.device, dtype=k_new.dtype)

        slot = self.num_tokens % self.window_size
        self.k_buffer[:, :, slot : slot + 1, :] = k_new
        self.v_buffer[:, :, slot : slot + 1, :] = v_new
        self.num_tokens += 1

        if self.num_tokens <= self.window_size:
            return self.k_buffer[:, :, : self.num_tokens, :], self.v_buffer[:, :, : self.num_tokens, :]
        else:
            # Reorder buffer so chronological order is preserved: oldest to newest
            start = self.num_tokens % self.window_size
            k_ordered = torch.cat([self.k_buffer[:, :, start:, :], self.k_buffer[:, :, :start, :]], dim=2)
            v_ordered = torch.cat([self.v_buffer[:, :, start:, :], self.v_buffer[:, :, :start, :]], dim=2)
            return k_ordered, v_ordered


class SlidingWindowAttention(nn.Module):
    """
    Causal Multi-Head Attention restricted to a sliding local window of size W.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rolling_cache: Optional[RollingKVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[RollingKVCache]]:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Handle circular rolling buffer for token-by-token generation
        if rolling_cache is not None and T == 1:
            k_active, v_active = rolling_cache.update(k, v)
            total_k_len = k_active.size(2)
            scores = torch.matmul(q, k_active.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Query at last position can attend to all keys within the active window
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            out = torch.matmul(weights, v_active)
        else:
            # Full sequence forward pass with sliding band mask
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Mask construction:
            # Allowed: j <= i (causal) AND j >= i - window_size + 1 (sliding window)
            q_idx = torch.arange(T, device=x.device).unsqueeze(1)
            k_idx = torch.arange(T, device=x.device).unsqueeze(0)
            causal_mask = k_idx > q_idx
            window_mask = k_idx < (q_idx - self.window_size + 1)
            full_mask = causal_mask | window_mask

            scores = scores.masked_fill(full_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out, rolling_cache
