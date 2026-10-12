"""
Tests for Milestone 35: Fine-Grained Block-Wise Quantization.
"""

import os
import pytest
import torch
import torch.nn as nn

from model.gpt import GPT, GPTConfig
from inference.block_quantization import (
    BlockQuantizedLinear,
    quantize_model_blockwise,
    save_quantized_model,
    load_quantized_model,
)


def test_block_quantized_linear_fidelity():
    in_features = 64
    out_features = 64
    linear = nn.Linear(in_features, out_features)
    x = torch.randn(4, in_features)

    orig_out = linear(x)

    # 8-bit block quantization (block_size=32)
    q8 = BlockQuantizedLinear.from_linear(linear, bits=8, block_size=32)
    q8_out = q8(x)
    rmse_8 = torch.sqrt(torch.mean((orig_out - q8_out) ** 2)).item()
    assert rmse_8 < 0.05, f"8-bit RMSE too high: {rmse_8}"

    # 4-bit block quantization (block_size=32)
    q4 = BlockQuantizedLinear.from_linear(linear, bits=4, block_size=32)
    q4_out = q4(x)
    rmse_4 = torch.sqrt(torch.mean((orig_out - q4_out) ** 2)).item()
    assert rmse_4 < 0.25, f"4-bit RMSE too high: {rmse_4}"


def test_quantize_gpt_model_blockwise(tmp_path):
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=64,
        n_heads=2,
        n_layers=2,
        tie_weights=False,
    )
    model = GPT(config)
    model.eval()
    input_ids = torch.randint(0, 100, (2, 8))

    orig_logits, _, _ = model(input_ids)

    # Quantize to 4-bit blockwise
    q_model = quantize_model_blockwise(model, bits=4, block_size=32)

    # Check that MLP layers are now BlockQuantizedLinear
    assert isinstance(q_model.blocks[0].mlp.c_fc, BlockQuantizedLinear)
    assert isinstance(q_model.blocks[0].mlp.c_proj, BlockQuantizedLinear)

    q_logits, _, _ = q_model(input_ids)
    assert q_logits.shape == orig_logits.shape

    # Save and reload
    filepath = str(tmp_path / "model_block_q4.nexq")
    save_quantized_model(q_model, filepath)
    assert os.path.exists(filepath)

    reloaded_model = load_quantized_model(q_model, filepath)
    reloaded_logits, _, _ = reloaded_model(input_ids)
    assert torch.allclose(q_logits, reloaded_logits, atol=1e-5)
