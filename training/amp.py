"""
Enhancement: Automatic Mixed Precision (AMP) & Gradient Scaler

Why AMP?
Deep learning models typically execute in 32-bit floating point (FP32).
Modern accelerators (NVIDIA Tensor Cores, Apple Silicon MPS) compute matrix multiplications
dramatically faster in 16-bit precision (FP16 or BF16) while halving VRAM requirements.

To prevent small gradients from underflowing to zero in FP16, a dynamic GradScaler scales
loss values before backprop, and un-scales them before the optimizer step.
"""

from contextlib import nullcontext
from typing import Optional
import torch
import torch.nn as nn


class MixedPrecisionManager:
    """
    Manages PyTorch AMP autocast context and gradient scaling across CUDA, MPS, and CPU.
    """

    def __init__(self, device: torch.device, precision: str = "fp16"):
        self.device = device
        self.device_type = "cuda" if "cuda" in device.type else ("mps" if "mps" in device.type else "cpu")
        self.precision = precision.lower()

        # Resolve dtype
        if self.precision == "bf16":
            self.dtype = torch.bfloat16
        elif self.precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # Enable scaler for fp16 on CUDA (bf16 has wider exponent range, rarely needs scaling)
        self.enabled = (self.precision in ("fp16", "bf16")) and (self.device_type in ("cuda", "cpu"))
        if self.device_type == "cuda" and self.precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def autocast_context(self):
        """Returns the appropriate autocast context manager."""
        if self.device_type in ("cuda", "cpu") and self.enabled:
            return torch.autocast(device_type=self.device_type, dtype=self.dtype)
        return nullcontext()

    def scale_and_backward(self, loss: torch.Tensor) -> None:
        """Backpropagates loss with dynamic gradient scaling if active."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer, model: Optional[nn.Module] = None, grad_clip: float = 1.0) -> None:
        """Performs optimizer step with unscaling and gradient clipping."""
        if self.scaler is not None:
            if model is not None and grad_clip > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if model is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
