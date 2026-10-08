import torch
import torch.nn as nn
from typing import Optional, Tuple

from nexlm.model.attention import CausalSelfAttention
from nexlm.model.normalization import get_norm_layer


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Expands hidden dimension by 4x, applies GELU activation,
    and projects back down to model dimension with dropout.
    """

    def __init__(self, dim: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Decoder Block (GPT-2 style).
    
    Architecture:
        x = x + Attention(Norm1(x))
        x = x + MLP(Norm2(x))
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = True,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln_1 = get_norm_layer(norm_type, dim, eps=norm_eps, bias=bias)
        self.attn = CausalSelfAttention(
            dim=dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = get_norm_layer(norm_type, dim, eps=norm_eps, bias=bias)
        self.mlp = MLP(dim=dim, dropout=dropout, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Attention residual
        norm_x = self.ln_1(x)
        attn_out, new_kv_cache = self.attn(norm_x, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out

        # Pre-LN MLP residual
        norm_x = self.ln_2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x, new_kv_cache
