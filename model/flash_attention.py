"""
Enhancement: Hardware-Accelerated Scaled Dot-Product Attention (FlashAttention / SDPA)

Why FlashAttention?
Standard attention materializes the entire (T x T) attention matrix in high-bandwidth memory (HBM).
For long sequences, memory access overhead (memory bandwidth) dominates runtime rather than compute FLOPs.

FlashAttention (Dao et al., 2022, 2023) computes exact attention in tiles using GPU SRAM,
fusing the softmax reduction and avoiding writing the intermediate attention scores to HBM.

This module provides FlashCausalSelfAttention with selectable backends:
1. "sdpa": Invokes PyTorch 2.0+ F.scaled_dot_product_attention (FlashAttention-2 / Memory-Efficient kernel).
2. "manual": Educational from-scratch matrix multiplication and explicit causal masking.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashCausalSelfAttention(nn.Module):
    """
    Causal Self-Attention module supporting both hardware-accelerated SDPA
    and from-scratch manual attention backends.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        bias: bool = False,
        backend: str = "sdpa",
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.backend = backend.lower()

        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Upper triangular mask for manual mode
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        backend_override: Optional[str] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        backend = (backend_override or self.backend).lower()

        # Step 1: Project Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 2: Attention Computation
        if backend == "sdpa" and hasattr(F, "scaled_dot_product_attention"):
            # Hardware-accelerated kernel (FlashAttention / Memory-Efficient / C++ vector engine)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual from-scratch fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            out = torch.matmul(weights, v)

        # Step 3: Output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out
