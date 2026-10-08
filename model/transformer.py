"""
Milestone 4 & 5: Transformer Block & Feed-Forward Network

Architecture (Pre-LayerNorm GPT-2 style):
    x ──> LayerNorm ──> MultiHeadAttention ──(+)──> LayerNorm ──> MLP ──(+)──> Output
    │                                         │     │                  │
    └────────── Residual Connection ──────────┘     └─── Residual ─────┘

Why Pre-LayerNorm?
In original Attention Is All You Need (Post-LN), normalization occurred after the residual add.
This caused unstable gradients in deep networks, requiring complex warmup tricks.
Pre-LN normalizes the inputs before attention and MLP, maintaining a clean gradient highway
across residual skips.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from model.attention import MultiHeadCausalSelfAttention


class LayerNorm(nn.Module):
    """
    Layer Normalization implemented from first principles.
    Formula: y = ((x - mean) / sqrt(var + eps)) * gamma + beta
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class MLP(nn.Module):
    """
    Feed-Forward Network (Position-wise).
    
    Expands hidden dimension by 4x, applies GELU activation,
    and projects back down to d_model with dropout.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    A single GPT-style Pre-LayerNorm Transformer Block.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with Pre-LN and residual skip
        norm_x = self.ln_1(x)
        attn_out, new_kv = self.attn(norm_x, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out

        # MLP with Pre-LN and residual skip
        norm_x = self.ln_2(x)
        x = x + self.mlp(norm_x)
        return x, new_kv
