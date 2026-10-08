"""
Enhancement: Post-Training Weight-Only Quantization (INT8 & INT4)

Why Quantization?
Large Language Models are primarily memory-bandwidth bound during autoregressive decoding.
By quantizing weights from FP32/FP16 down to INT8 or INT4:
1. Model memory footprint decreases by 2x (INT8) or 4x-8x (INT4).
2. Weight transfer latency from RAM to compute units drops proportionally.

Formulation (Symmetric Absolute Maximum Quantization):
    scale = max(|W|) / qmax
    W_q   = clamp(round(W / scale), -qmax, qmax)
    W_deq = W_q * scale
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear storing weights in INT8 or INT4
    and dequantizing on-the-fly during matrix multiplication.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.qmax = 127 if bits == 8 else 7

        # Register quantized weights and scale buffer
        self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.zeros((out_features, 1), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int = 8) -> "QuantizedLinear":
        """Quantize an existing floating-point nn.Linear layer."""
        q_module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
        )

        with torch.no_grad():
            w = linear.weight.float()
            # Per-channel (per-row) maximum absolute value
            max_val = torch.max(torch.abs(w), dim=1, keepdim=True).values.clamp(min=1e-8)
            scale = max_val / q_module.qmax
            w_q = torch.clamp(torch.round(w / scale), -q_module.qmax, q_module.qmax).to(torch.int8)

            q_module.weight_q.copy_(w_q)
            q_module.scale.copy_(scale)

            if linear.bias is not None:
                q_module.bias.copy_(linear.bias.float())

        return q_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight on the fly: (out_features, in_features)
        w_dequant = (self.weight_q.to(x.dtype)) * self.scale.to(x.dtype)
        out = torch.matmul(x, w_dequant.t())
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


def quantize_model(model: nn.Module, bits: int = 8, skip_lm_head: bool = True) -> nn.Module:
    """
    Recursively replaces nn.Linear layers with QuantizedLinear modules throughout the model.
    """
    for name, child in model.named_children():
        if skip_lm_head and name == "lm_head":
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear.from_float(child, bits=bits))
        else:
            quantize_model(child, bits=bits, skip_lm_head=skip_lm_head)
    return model


def compute_model_size_mb(model: nn.Module) -> float:
    """Calculate the memory size of a model in Megabytes."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)
