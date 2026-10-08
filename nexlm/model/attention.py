import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with optional KV-cache support.
    
    Implements:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V
    where M is a causal mask ensuring tokens only attend to past and current tokens.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len

        # Combined projection for query, key, value for computational efficiency
        self.c_attn = nn.Linear(dim, 3 * dim, bias=bias)

        # Output projection
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: upper triangular elements set to 1 (to be masked with -inf)
        # Register as buffer so it moves with the module to device
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            kv_cache: Optional tuple of (past_key, past_value), each of shape
                      (batch_size, n_heads, past_seq_len, head_dim)
            use_cache: Whether to return updated (key, value) cache
            
        Returns:
            output: Attention output tensor of shape (batch_size, seq_len, dim)
            new_kv_cache: Updated (key, value) tuple if use_cache is True, else None
        """
        B, T, C = x.shape

        # Compute Q, K, V
        # Shape: (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Handle KV Cache
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None
        total_kv_len = k.size(2)

        # Scaled dot-product attention
        # Scores: (B, n_heads, T, total_kv_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal masking
        # When generating with KV cache, T is typically 1 and total_kv_len is current_step + 1.
        # Query token at index (total_kv_len - T + i) can attend to keys up to (total_kv_len - T + i).
        if T == total_kv_len:
            # Full sequence (standard training or prefill)
            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(mask, float("-inf"))
        elif T > 1:
            # Chunked sequence with prefix cache
            # Row i (0 <= i < T) represents query at position (total_kv_len - T + i)
            # Cannot attend to keys at positions > (total_kv_len - T + i)
            q_indices = torch.arange(total_kv_len - T, total_kv_len, device=x.device).unsqueeze(1)
            k_indices = torch.arange(total_kv_len, device=x.device).unsqueeze(0)
            chunk_mask = k_indices > q_indices
            scores = scores.masked_fill(chunk_mask, float("-inf"))
        # If T == 1, query at last position can attend to all keys up to total_kv_len - 1,
        # so no mask is needed!

        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: (B, n_heads, T, head_dim)
        out = torch.matmul(attn_weights, v)

        # Reassemble all heads side by side: (B, T, dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.resid_dropout(self.c_proj(out))

        return out, new_kv_cache
