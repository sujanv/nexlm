"""
Milestone 40: Multi-Head Latent Attention (MLA) (DeepSeek Architecture)

Standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) store full-dimensional
key-value tensors in the KV cache, which quickly bottlenecks GPU memory bandwidth during inference.

DeepSeek's Multi-Head Latent Attention (MLA) compresses keys and values into a low-rank latent space:
1. Low-Rank KV Compression:
   c_t^{KV} = x_t W^{DKV} in R^{d_{ckv}}  (d_{ckv} << n_heads * d_k)
   Only c_t^{KV} (and decoupled RoPE keys k_t^R) are cached!
2. Up-Projection:
   k_t^C = c_t^{KV} W^{UK},  v_t = c_t^{KV} W^{UV}
3. Matrix Absorption:
   During generation, queries can absorb W^{UK} directly:
   q_t^T k_{j}^C = (q_t^T (W^{UK})^T) c_j^{KV}
   This allows calculating attention scores directly against the compressed latent cache,
   slashing KV cache memory consumption by up to 90%+.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding, apply_rotary_emb


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) as used in DeepSeek-V2 and DeepSeek-V3.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        q_lora_rank: int = 64,
        kv_lora_rank: int = 64,
        qk_nope_head_dim: int = 32,
        qk_rope_head_dim: int = 16,
        v_head_dim: int = 32,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        # 1. Query down-projection and up-projection
        self.w_dq = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = nn.LayerNorm(q_lora_rank)
        self.w_uq = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim, bias=False)

        # 2. Key-Value down-projection (compressed latent) and up-projections
        self.w_dkv = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)

        # Up-projection for content keys and values
        self.w_uk = nn.Linear(kv_lora_rank, n_heads * self.qk_nope_head_dim, bias=False)
        self.w_uv = nn.Linear(kv_lora_rank, n_heads * self.v_head_dim, bias=False)

        # 3. Decoupled RoPE Key projection for positional awareness
        self.w_rk = nn.Linear(d_model, qk_rope_head_dim, bias=False)
        self.rope = RotaryEmbedding(qk_rope_head_dim, max_seq_len=max_seq_len)

        # 4. Final output projection
        self.w_o = nn.Linear(n_heads * v_head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (B, T, d_model)
            kv_cache: Optional tuple of (cached_c_kv, cached_k_rope)
                      cached_c_kv: (B, seq_len, kv_lora_rank)
                      cached_k_rope: (B, seq_len, qk_rope_head_dim)
        """
        B, T, _ = x.shape

        # 1. Compute queries: down-project, norm, up-project
        c_q = self.q_norm(self.w_dq(x))
        q = self.w_uq(c_q).view(B, T, self.n_heads, self.qk_head_dim).transpose(1, 2)
        # Split into content query (no-pe) and positional query (rope)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim :]

        # Apply RoPE to query rope component
        cos, sin = self.rope(q_rope, start_pos=start_pos)
        q_rope, _ = apply_rotary_emb(q_rope, q_rope, cos, sin)

        # 2. Compute compressed KV latent: c_kv (B, T, kv_lora_rank)
        c_kv = self.kv_norm(self.w_dkv(x))

        # Compute decoupled RoPE key: k_rope (B, T, qk_rope_head_dim)
        k_rope = self.w_rk(x).unsqueeze(1)  # (B, 1, T, rope_dim)
        _, k_rope = apply_rotary_emb(k_rope, k_rope, cos, sin)
        k_rope = k_rope.squeeze(1)  # (B, T, rope_dim)

        # 3. Handle KV Cache: Only store (c_kv, k_rope)!
        if kv_cache is not None:
            past_c_kv, past_k_rope = kv_cache
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)

        new_kv = (c_kv, k_rope) if use_cache else None
        total_len = c_kv.shape[1]

        # 4. Decompress content keys and values from compressed latent c_kv
        # k_nope: (B, total_len, n_heads * qk_nope_head_dim) -> (B, n_heads, total_len, qk_nope_head_dim)
        k_nope = self.w_uk(c_kv).view(B, total_len, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, total_len, self.n_heads, self.v_head_dim).transpose(1, 2)

        # Expand k_rope across heads: (B, n_heads, total_len, qk_rope_head_dim)
        k_rope_heads = k_rope.unsqueeze(1).expand(B, self.n_heads, total_len, self.qk_rope_head_dim)

        # Combine content key and rope key
        k = torch.cat([k_nope, k_rope_heads], dim=-1)
        q_full = torch.cat([q_nope, q_rope], dim=-1)

        # 5. Attention scores computation
        # (B, n_heads, T, qk_head_dim) x (B, n_heads, qk_head_dim, total_len) -> (B, n_heads, T, total_len)
        scores = torch.matmul(q_full, k.transpose(-2, -1)) * self.scale

        # Causal mask if decoding or prefilling
        if T > 1 or total_len > T:
            mask = torch.triu(torch.full((T, total_len), float("-inf"), device=x.device), diagonal=start_pos + 1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. Weighted values and output projection
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, v_head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_head_dim)
        return self.w_o(out), new_kv
