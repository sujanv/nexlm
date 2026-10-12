"""
Tests for Milestone 32: RMSNorm implementation and GPT integration.
"""

import pytest
import torch

from model.rmsnorm import RMSNorm
from model.transformer import TransformerBlock
from model.gpt import GPT, GPTConfig


def test_rmsnorm_shape_and_values():
    dim = 64
    norm = RMSNorm(dim=dim, eps=1e-6)
    x = torch.randn(2, 10, dim)

    out = norm(x)
    assert out.shape == x.shape

    # When weight == 1, mean of squares should be ~1
    rms_val = out.pow(2).mean(dim=-1)
    assert torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-3)


def test_rmsnorm_scale_invariance():
    dim = 32
    norm = RMSNorm(dim=dim, eps=1e-6)
    x = torch.randn(4, 16, dim)
    scale = 3.5

    # RMSNorm is scale-invariant: norm(scale * x) == norm(x)
    out1 = norm(x)
    out2 = norm(x * scale)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_rmsnorm_gradients():
    dim = 32
    norm = RMSNorm(dim=dim)
    x = torch.randn(2, 8, dim, requires_grad=True)

    out = norm(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(norm.weight.grad).any()


def test_transformer_block_with_rmsnorm():
    d_model = 64
    n_heads = 4
    max_seq_len = 32
    block = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        norm_type="rmsnorm",
    )

    x = torch.randn(2, 12, d_model)
    out, _ = block(x)
    assert out.shape == x.shape
    assert isinstance(block.ln_1, RMSNorm)
    assert isinstance(block.ln_2, RMSNorm)


def test_gpt_with_rmsnorm_forward_and_cache():
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=64,
        n_heads=2,
        n_layers=2,
        norm_type="rmsnorm",
    )
    model = GPT(config)
    assert isinstance(model.ln_f, RMSNorm)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0

    # Test with KV cache
    logits_cached, _, new_kv = model(input_ids, use_cache=True)
    assert len(new_kv) == config.n_layers
