"""
Enhancement: SwiGLU Gated Feed-Forward Network

Why SwiGLU?
Shazeer (2020) demonstrated that Gated Linear Unit (GLU) variants consistently outperform
standard ReLU/GELU activations in Transformer architectures.
Today, nearly every top-tier open-weights LLM (Llama 2/3, Mistral, Gemma, DeepSeek) uses SwiGLU.

Formula:
    SwiGLU(x) = ( SiLU(x @ W_gate) * (x @ W_up) ) @ W_down
where SiLU(z) = z * sigmoid(z).

To preserve parameter count parity with a standard 4 * d_model MLP, the hidden dimension
is typically scaled to approx (8/3) * d_model, rounded to a multiple of 64.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) MLP module.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            # Standard LLaMA convention: (8/3) * d_model, rounded to multiple of 64
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        # Gate path with SiLU activation
        gate = F.silu(self.w_gate(x))
        # Up-projection linear path
        up = self.w_up(x)
        # Elementwise gating product
        gated = gate * up
        # Down-projection back to d_model
        out = self.w_down(gated)
        return self.dropout(out)
