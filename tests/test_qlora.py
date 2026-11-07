"""
Tests for Milestone 49: QLoRA & NF4 Quantization Engine.
"""

import pytest
import torch
import torch.nn as nn

from training.qlora import NF4Quantizer, QLoRALinear


def test_nf4_quantizer_fidelity():
    in_features, out_features = 64, 64
    linear = nn.Linear(in_features, out_features, bias=False)

    quantizer = NF4Quantizer(block_size=64)
    q_indices, scales = quantizer.quantize(linear.weight)

    assert q_indices.shape == (in_features * out_features,)
    # NF4 indices must be within [0, 15]
    assert (q_indices >= 0).all() and (q_indices <= 15).all()

    dequant_w = quantizer.dequantize(q_indices, scales, out_features, in_features)
    assert dequant_w.shape == linear.weight.shape

    # Check that quantization error (RMSE) is modest (< 0.15 for NF4)
    rmse = torch.sqrt(torch.mean((linear.weight - dequant_w) ** 2)).item()
    assert rmse < 0.15


def test_qlora_linear_forward_initialization():
    in_features, out_features = 64, 64
    linear = nn.Linear(in_features, out_features)
    x = torch.randn(2, 8, in_features)

    qlora_layer = QLoRALinear.from_linear(linear, r=4, lora_alpha=8.0, block_size=64)

    # Initial lora_B is all zeros, so QLoRA output should exactly equal dequantized base output
    base_w = qlora_layer.quantizer.dequantize(
        qlora_layer.q_weight, qlora_layer.scales, out_features, in_features
    )
    expected_base_out = torch.nn.functional.linear(x, base_w, qlora_layer.bias)

    actual_out = qlora_layer(x)
    assert torch.allclose(actual_out, expected_base_out, atol=1e-5)


def test_qlora_gradient_backprop():
    in_features, out_features = 64, 64
    linear = nn.Linear(in_features, out_features)
    qlora_layer = QLoRALinear.from_linear(linear, r=4, lora_alpha=8.0, block_size=64)

    x = torch.randn(2, 4, in_features)
    out = qlora_layer(x)
    loss = out.sum()
    loss.backward()

    # Base weight buffers should NOT receive gradients
    assert qlora_layer.q_weight.grad is None
    assert qlora_layer.scales.grad is None

    # LoRA adapters MUST receive gradients
    assert qlora_layer.lora_A.grad is not None
    assert qlora_layer.lora_B.grad is not None
    assert not torch.isnan(qlora_layer.lora_A.grad).any()
    assert not torch.isnan(qlora_layer.lora_B.grad).any()
