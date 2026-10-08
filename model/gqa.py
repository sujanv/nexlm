"""
Enhancement: Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)

Why GQA?
In standard Multi-Head Attention (MHA):
    n_heads_q == n_heads_kv
For long sequences, the KV cache becomes the primary memory bottleneck.

Grouped-Query Attention (Ainslie et al., 2023) partitions query heads into G groups.
Each group of (n_heads_q // n_heads_kv) query heads shares a single Key and Value head.

Memory Comparison:
    MHA (8 Q, 8 KV): 100% KV cache size
    GQA (8 Q, 2 KV):  25% KV cache size (4x reduction!)
    MQA (8 Q, 1 KV): 12.5% KV cache size (8x reduction!)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads across Query groups.
    x shape: (B, n_kv_heads, T, head_dim)
    output:  (B, n_kv_heads * n_rep, T, head_dim) == (B, n_q_heads, T, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, head_dim)
        .reshape(B, n_kv_heads * n_rep, T, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) module implemented from first principles.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Query projection: projects to full (n_heads * head_dim)
        self.w_q = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        # Key & Value projections: project only to (n_kv_heads * head_dim)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # Step 1: Project Q, K, V
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Step 2: Update KV cache with compact KV heads
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None
        total_kv_len = k.size(2)

        # Step 3: Expand KV heads to match Q heads for attention calculation
        k_expanded = repeat_kv(k, self.num_groups)
        v_expanded = repeat_kv(v, self.num_groups)

        # Step 4: Scaled dot-product attention
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if T == total_kv_len:
            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(mask, float("-inf"))
        elif T > 1:
            q_idx = torch.arange(total_kv_len - T, total_kv_len, device=x.device).unsqueeze(1)
            k_idx = torch.arange(total_kv_len, device=x.device).unsqueeze(0)
            scores = scores.masked_fill(k_idx > q_idx, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = torch.matmul(weights, v_expanded)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.w_o(out)), new_kv_cache
