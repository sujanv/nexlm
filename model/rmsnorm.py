"""
Milestone 32: Root Mean Square Normalization (RMSNorm)

RMSNorm simplifies standard LayerNorm by removing the mean-centering step,
focusing purely on scaling invariance via root-mean-square:
    RMS(x) = sqrt( (1 / d) * sum(x_i^2) + eps )
    y = (x / RMS(x)) * gamma

Adopted by modern open-weight LLMs (LLaMA 1/2/3, Mistral, Gemma, DeepSeek, Qwen)
for improved computational throughput and gradient flow without loss of fidelity.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Args:
        dim: Embedding dimension (d_model).
        eps: Small epsilon constant for numerical stability (default: 1e-6).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability across fp16/bf16
        input_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        return normed.to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (..., dim)
        Returns:
            Normalized and scaled tensor of shape (..., dim)
        """
        return self._norm(x) * self.weight

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
