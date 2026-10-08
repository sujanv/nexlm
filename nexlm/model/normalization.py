import torch
import torch.nn as nn
from typing import Optional


class LayerNorm(nn.Module):
    """
    Layer Normalization implemented from scratch.
    
    Formula:
        y = ((x - mean) / sqrt(var + eps)) * weight + bias
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        out = x_norm * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) implemented from scratch.
    
    Formula:
        RMS(x) = sqrt(mean(x^2) + eps)
        y = (x / RMS(x)) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def get_norm_layer(norm_type: str, dim: int, eps: float = 1e-5, bias: bool = True) -> nn.Module:
    """Factory helper to obtain the configured normalization module."""
    if norm_type.lower() == "layernorm":
        return LayerNorm(dim=dim, eps=eps, bias=bias)
    elif norm_type.lower() == "rmsnorm":
        return RMSNorm(dim=dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Choose 'layernorm' or 'rmsnorm'.")
