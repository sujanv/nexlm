"""
Enhancement: Low-Rank Adaptation (LoRA) for Parameter-Efficient Fine-Tuning (PEFT)

Why LoRA?
Hu et al. (2021) demonstrated that weight updates during adaptation have a low "intrinsic dimension".
Instead of fine-tuning all parameters W0 in a linear layer:
    W = W0 + delta_W = W0 + (B @ A) * (alpha / r)
where:
    - W0 in R^(d_out x d_in) is completely frozen (requires_grad = False)
    - A in R^(r x d_in) is initialized with N(0, 1/r)
    - B in R^(d_out x r) is initialized to zeros
    - r << min(d_in, d_out) is the rank (e.g., r = 4, 8, 16)
    - alpha is a scaling constant (e.g., alpha = 16)

Advantages:
1. Drastically reduces trainable parameters (< 1% of the model).
2. Allows zero-overhead inference by merging: W_merged = W0 + (alpha / r) * (B @ A).
"""

import math
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps or replaces an nn.Linear layer with Low-Rank Adaptation matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        # Base frozen weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable low-rank adapter matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        self.merged = False

    def reset_parameters(self) -> None:
        """Initialize base weights (if needed) and LoRA parameters."""
        if self.r > 0:
            # Gaussian initialization for A, zero initialization for B
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """Converts an existing nn.Linear layer into a LoRALinear module."""
        lora_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            lora_layer.weight.copy_(linear.weight)
            lora_layer.weight.requires_grad = False
            if linear.bias is not None:
                lora_layer.bias.copy_(linear.bias)
                lora_layer.bias.requires_grad = False
        return lora_layer

    def merge(self) -> None:
        """Merge low-rank weights directly into base weight matrix for zero-overhead inference."""
        if self.r > 0 and not self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.weight.data.add_(delta_w)
                self.merged = True

    def unmerge(self) -> None:
        """Undo the merge if switching back to training."""
        if self.r > 0 and self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.weight.data.sub_(delta_w)
                self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear transformation
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0 and not self.merged:
            # Low-rank branch: (x @ A^T) @ B^T * scaling
            lora_out = self.lora_dropout(x)
            lora_out = nn.functional.linear(lora_out, self.lora_A)
            lora_out = nn.functional.linear(lora_out, self.lora_B)
            result = result + (lora_out * self.scaling)
        return result


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("c_attn", "w_q", "w_v", "w_gate", "w_up"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> List[str]:
    """
    Recursively replaces target linear modules in a model with LoRALinear layers.
    Freezes all non-LoRA parameters.
    """
    replaced = []
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and any(tgt in name for tgt in target_modules):
            setattr(model, name, LoRALinear.from_linear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(name)
        else:
            replaced.extend(apply_lora_to_model(child, target_modules, r, alpha, dropout))
    return replaced


def mark_only_lora_as_trainable(model: nn.Module) -> Tuple[int, int]:
    """
    Freezes all parameters except LoRA parameters (lora_A, lora_B).
    Returns (trainable_params, total_params).
    """
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora_" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    return trainable_params, total_params
