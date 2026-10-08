import torch
import torch.nn as nn
import pytest

from inference.quantize import QuantizedLinear, quantize_model, compute_model_size_mb
from model.gpt import GPT, GPTConfig


def test_quantized_linear_fidelity():
    """Verify that INT8 quantized linear output closely tracks FP32 output."""
    torch.manual_seed(42)
    in_feat, out_feat = 64, 32
    linear_fp32 = nn.Linear(in_feat, out_feat, bias=True)
    x = torch.randn(4, 10, in_feat)

    # Convert to INT8
    linear_int8 = QuantizedLinear.from_float(linear_fp32, bits=8)

    y_fp32 = linear_fp32(x)
    y_int8 = linear_int8(x)

    # Check shapes
    assert y_int8.shape == y_fp32.shape

    # Cosine similarity between outputs should be extremely high (> 0.999)
    cos_sim = torch.nn.functional.cosine_similarity(y_fp32.flatten(), y_int8.flatten(), dim=0)
    assert cos_sim.item() > 0.999, f"Quantization distorted linear output: cos_sim={cos_sim.item()}"

    # Verify storage format
    assert linear_int8.weight_q.dtype == torch.int8


def test_full_model_quantization():
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=64,
        n_heads=2,
        n_layers=2,
    )
    model = GPT(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        orig_logits, _, _ = model(tokens)

    # Quantize model
    q_model = quantize_model(model, bits=8)
    with torch.no_grad():
        q_logits, _, _ = q_model(tokens)

    assert q_logits.shape == orig_logits.shape
    cos_sim = torch.nn.functional.cosine_similarity(orig_logits.flatten(), q_logits.flatten(), dim=0)
    assert cos_sim.item() > 0.99, f"Model quantization degraded output too much: {cos_sim.item()}"
