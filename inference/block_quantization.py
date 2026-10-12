"""
Milestone 35: Fine-Grained Block-Wise Quantization (AWQ / GGUF Style)

Standard tensor-wide quantization suffers from outlier degradation: a single extreme weight
causes quantization noise across all other parameters in the row.

Block-wise quantization splits weight matrices into contiguous groups (default block_size=32):
    For each block of 32 weights:
        s = (max(w) - min(w)) / (2^b - 1)
        q = clamp(round((w - min(w)) / s), 0, 2^b - 1)
        w_hat = q * s + min(w)

Isolates outlier effects to a tiny group of weights, preserving high perplexity even at 4-bit quantization.
"""

from dataclasses import dataclass
import io
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockQuantizedLinear(nn.Module):
    """
    Linear layer with block-wise quantized weights (4-bit or 8-bit, block_size=32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        block_size: int = 32,
    ):
        super().__init__()
        assert bits in (4, 8), "Only 4-bit and 8-bit block quantization supported."
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        self.max_q = (1 << bits) - 1

        total_elements = out_features * in_features
        assert total_elements % block_size == 0, f"Total elements ({total_elements}) must be divisible by block_size ({block_size})"
        num_blocks = total_elements // block_size

        # Quantized indices stored as uint8
        self.register_buffer("qweight", torch.zeros(total_elements, dtype=torch.uint8))
        # Per-block scales and zero-points (fp16/fp32)
        self.register_buffer("scales", torch.ones(num_blocks, dtype=torch.float32))
        self.register_buffer("zeros", torch.zeros(num_blocks, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 4, block_size: int = 32) -> "BlockQuantizedLinear":
        """
        Quantizes a standard nn.Linear layer into a BlockQuantizedLinear module.
        """
        out_features, in_features = linear.weight.shape
        total_elements = out_features * in_features
        
        # Pad in_features if needed so total elements is divisible by block_size
        assert total_elements % block_size == 0, f"Weight shape ({out_features}, {in_features}) must divide block_size {block_size}"

        q_linear = cls(
            in_features=in_features,
            out_features=out_features,
            bias=linear.bias is not None,
            bits=bits,
            block_size=block_size,
        )

        w = linear.weight.detach().float().view(-1, block_size)
        num_blocks = w.shape[0]

        w_min = w.min(dim=1, keepdim=True).values
        w_max = w.max(dim=1, keepdim=True).values

        # Compute per-block scales and offsets
        scales = (w_max - w_min).clamp(min=1e-8) / q_linear.max_q
        zeros = w_min

        # Quantize to integer range [0, 2^b - 1]
        q = torch.clamp(torch.round((w - zeros) / scales), 0, q_linear.max_q).to(torch.uint8)

        q_linear.qweight.copy_(q.view(-1))
        q_linear.scales.copy_(scales.view(-1))
        q_linear.zeros.copy_(zeros.view(-1))

        if linear.bias is not None:
            q_linear.bias.data.copy_(linear.bias.detach())

        return q_linear

    def unpack_weights(self) -> torch.Tensor:
        """
        Dequantizes stored integer blocks back to float32 matrix.
        """
        q = self.qweight.float().view(-1, self.block_size)
        s = self.scales.unsqueeze(1)
        z = self.zeros.unsqueeze(1)
        w_dequant = (q * s + z).view(self.out_features, self.in_features)
        return w_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.unpack_weights().to(x.dtype)
        return F.linear(x, w, self.bias)

    def state_dict_bytes(self) -> int:
        """Computes memory footprint of the quantized parameters."""
        # 4-bit packs 2 values per byte (or uint8 stored directly here for simplicity)
        q_bytes = self.qweight.numel() if self.bits == 8 else (self.qweight.numel() // 2)
        scale_bytes = self.scales.numel() * 4
        zero_bytes = self.zeros.numel() * 4
        bias_bytes = (self.bias.numel() * 4) if self.bias is not None else 0
        return q_bytes + scale_bytes + zero_bytes + bias_bytes


def quantize_model_blockwise(model: nn.Module, bits: int = 4, block_size: int = 32) -> nn.Module:
    """
    Recursively replaces all nn.Linear layers in model with BlockQuantizedLinear.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            # Check divisibility
            total_elements = child.weight.numel()
            if total_elements % block_size == 0:
                q_layer = BlockQuantizedLinear.from_linear(child, bits=bits, block_size=block_size)
                setattr(model, name, q_layer)
        else:
            quantize_model_blockwise(child, bits=bits, block_size=block_size)
    return model


def save_quantized_model(model: nn.Module, filepath: str) -> None:
    """Saves block-quantized weights and metadata to .nexq format."""
    state = {
        "state_dict": model.state_dict(),
        "arch_type": "block_quantized",
    }
    torch.save(state, filepath)


def load_quantized_model(model: nn.Module, filepath: str) -> nn.Module:
    """Loads block-quantized weights from .nexq format into model."""
    state = torch.load(filepath, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    return model
