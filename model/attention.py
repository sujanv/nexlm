"""
Milestones 2 & 3: Causal Self-Attention

Why Self-Attention?
In an autoregressive language model, every token needs to look back at relevant
preceding context to predict the next word.

1. Single-Head Causal Self-Attention (Milestone 2):
   - Q = X W_Q  (What this token is looking for)
   - K = X W_K  (What this token contains / offers)
   - V = X W_V  (What information is extracted if matched)
   - Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + Mask) V

2. Multi-Head Causal Self-Attention (Milestone 3):
   - Runs multiple attention heads in parallel so the model can simultaneously
     attend to syntactic relations, semantic connections, subject-verb agreements, etc.
   - Handcrafted without using PyTorch's nn.MultiheadAttention black box.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Milestone 2: Single-Head Causal Self-Attention
# =====================================================================

class SingleHeadCausalSelfAttention(nn.Module):
    """
    A single causal self-attention head implemented from first principles.
    
    Formula:
        Scores = (Q @ K^T) / sqrt(d_k)
        Scores = Scores.masked_fill(mask == 0, -inf)
        Weights = softmax(Scores, dim=-1)
        Output = Weights @ V
    """

    def __init__(self, d_in: int, d_out: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_out = d_out
        self.max_seq_len = max_seq_len

        # Linear projections for Query, Key, Value
        self.w_q = nn.Linear(d_in, d_out, bias=False)
        self.w_k = nn.Linear(d_in, d_out, bias=False)
        self.w_v = nn.Linear(d_in, d_out, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Causal mask: lower triangle is 1 (allowed), upper triangle is 0 (blocked)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
        Returns:
            out: Contextualized output of shape (batch_size, seq_len, d_out)
            attn_weights: Attention score matrix of shape (batch_size, seq_len, seq_len)
        """
        B, T, C = x.shape

        # Step 1: Linear projections
        q = self.w_q(x)  # (B, T, d_out)
        k = self.w_k(x)  # (B, T, d_out)
        v = self.w_v(x)  # (B, T, d_out)

        # Step 2: Compute raw attention affinities (Q @ K^T) / sqrt(d_k)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_out)  # (B, T, T)

        # Step 3: Apply causal mask so token i cannot attend to any j > i
        # Replace 0s in lower triangular mask with -infinity
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))

        # Step 4: Softmax to convert scores into probabilities summing to 1.0 per row
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 5: Weighted combination of value vectors
        out = torch.bmm(attn_weights, v)  # (B, T, d_out)
        return out, attn_weights


# =====================================================================
# Milestone 3: Multi-Head Causal Self-Attention
# =====================================================================

class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention implemented from scratch.
    
    Instead of running separate Linear layers in a loop, we project
    all heads simultaneously in a single matrix multiplication,
    then reshape to (B, n_heads, T, head_dim) for maximum parallel throughput.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Single projection for Q, K, V combined: (B, T, 3 * d_model)
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Final projection layer W_O: (B, T, d_model) -> (B, T, d_model)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Upper triangular mask (elements above diagonal are True -> masked)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # Compute Q, K, V in one projection
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Optional KV-cache concatenation for fast autoregressive decode
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None
        total_kv_len = k.size(2)

        # Scaled dot-product: (B, n_heads, T, total_kv_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal masking
        if T == total_kv_len:
            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(mask, float("-inf"))
        elif T > 1:
            q_idx = torch.arange(total_kv_len - T, total_kv_len, device=x.device).unsqueeze(1)
            k_idx = torch.arange(total_kv_len, device=x.device).unsqueeze(0)
            chunk_mask = k_idx > q_idx
            scores = scores.masked_fill(chunk_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Multiply values and concatenate heads: (B, T, d_model)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection W_O
        out = self.resid_dropout(self.c_proj(out))
        return out, new_kv_cache
