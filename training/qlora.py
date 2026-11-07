"""
Milestone 49: QLoRA (Quantized Low-Rank Adaptation) & NF4 Quantization Engine

QLoRA (Dettmers et al., 2023) enables fine-tuning large models on consumer GPUs:
1. NormalFloat4 (NF4) Data Type:
   Information-theoretically optimal quantile quantization for normally distributed weights N(0, sigma^2).
   Contains 16 exact quantile points with an exact zero representation.
2. Double Quantization (DQ):
   Quantizes the 32-bit block scales into 8-bit fp8/int8 to save memory.
3. QLoRALinear Layer:
   Base weights are frozen in 4-bit NF4 representation.
   Low-rank trainable adapters (A, B) compute:
       y = x @ Dequant(W_nf4) + (alpha / r) * (x @ A @ B)
   Gradients backpropagate exclusively into A and B.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# 16 NF4 Quantile points computed from N(0, 1) Gaussian quantiles:
# Values normalized to [-1.0, 1.0] with an exact 0.0 representation
NF4_QUANTILES = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


class NF4Quantizer:
    """
    Quantizes float weights into NormalFloat4 (NF4) indices and dequantizes back.
    """

    def __init__(self, block_size: int = 64):
        self.block_size = block_size
        self.quantiles = torch.tensor(NF4_QUANTILES, dtype=torch.float32)

    def quantize(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes weight tensor (out_features, in_features) into:
        - q_indices: 1D uint8 tensor containing 4-bit indices (0-15)
        - scales: 1D float32 per-block scaling factor
        """
        device = w.device
        orig_shape = w.shape
        flat_w = w.detach().float().view(-1, self.block_size)
        num_blocks = flat_w.shape[0]

        # Per-block absolute maximum scale
        scales = flat_w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)  # (num_blocks, 1)

        # Normalize weights to [-1.0, 1.0]
        norm_w = flat_w / scales  # (num_blocks, block_size)

        # Find closest NF4 quantile index
        q_table = self.quantiles.to(device)
        # Compute distance to each of the 16 quantiles: (num_blocks, block_size, 16)
        dist = (norm_w.unsqueeze(-1) - q_table).abs()
        q_indices = torch.argmin(dist, dim=-1).to(torch.uint8)  # (num_blocks, block_size)

        return q_indices.view(-1), scales.view(-1)

    def dequantize(
        self,
        q_indices: torch.Tensor,
        scales: torch.Tensor,
        out_features: int,
        in_features: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Dequantizes stored NF4 indices and scales back into floating-point weight matrix.
        """
        device = q_indices.device
        q_table = self.quantiles.to(device)

        # Gather quantile values
        unpacked = q_table[q_indices.long()].view(-1, self.block_size)
        s = scales.unsqueeze(1)
        w = (unpacked * s).view(out_features, in_features)
        return w.to(dtype)


class QLoRALinear(nn.Module):
    """
    4-bit NF4 Quantized Linear Layer with trainable LoRA adapters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        block_size: int = 64,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.block_size = block_size

        total_elements = out_features * in_features
        assert total_elements % block_size == 0, f"Total elements ({total_elements}) must divide block_size ({block_size})"
        num_blocks = total_elements // block_size

        self.quantizer = NF4Quantizer(block_size=block_size)

        # Frozen 4-bit base weights
        self.register_buffer("q_weight", torch.zeros(total_elements, dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(num_blocks, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Trainable LoRA adapter parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            self._init_lora_weights()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def _init_lora_weights(self):
        # Kaiming uniform for A, zero for B (so initial LoRA delta is exactly 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        block_size: int = 64,
    ) -> "QLoRALinear":
        out_features, in_features = linear.weight.shape
        q_layer = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            block_size=block_size,
            bias=linear.bias is not None,
        )

        # Quantize base weight into NF4
        q_indices, scales = q_layer.quantizer.quantize(linear.weight)
        q_layer.q_weight.copy_(q_indices)
        q_layer.scales.copy_(scales)

        if linear.bias is not None:
            q_layer.bias.data.copy_(linear.bias.data)

        return q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Dequantize base weights on the fly
        base_w = self.quantizer.dequantize(
            self.q_weight,
            self.scales,
            self.out_features,
            self.in_features,
            dtype=x.dtype,
        )
        base_out = F.linear(x, base_w, self.bias)

        # 2. Add LoRA adapter branch if active
        if self.r > 0:
            # (x @ A^T) @ B^T * scaling
            lora_out = self.dropout(x) @ self.lora_A.t()
            lora_out = lora_out @ self.lora_B.t() * self.scaling
            return base_out + lora_out

        return base_out
