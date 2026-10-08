import torch
import pytest

from nexlm.model.normalization import LayerNorm, RMSNorm


def test_layernorm_forward_and_backward():
    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C, requires_grad=True)
    ln = LayerNorm(dim=C, eps=1e-5)

    out = ln(x)
    assert out.shape == (B, T, C)

    # Check normalized statistics across last dimension
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)

    # Test backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_rmsnorm_forward_and_backward():
    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C, requires_grad=True)
    rms = RMSNorm(dim=C, eps=1e-5)

    out = rms(x)
    assert out.shape == (B, T, C)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert rms.weight.grad is not None
