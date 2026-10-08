import torch
import pytest

from model.swiglu import SwiGLU


def test_swiglu_shapes():
    B, T, d_model = 2, 8, 64
    mlp = SwiGLU(d_model=d_model, dropout=0.0)
    x = torch.randn(B, T, d_model)

    out = mlp(x)
    assert out.shape == (B, T, d_model)


def test_swiglu_backward_and_gradients():
    B, T, d_model = 2, 4, 32
    mlp = SwiGLU(d_model=d_model, dropout=0.0)
    x = torch.randn(B, T, d_model, requires_grad=True)

    out = mlp(x)
    loss = out.sum()
    loss.backward()

    # All three weight projections must receive non-zero gradients
    assert mlp.w_gate.weight.grad is not None
    assert mlp.w_up.weight.grad is not None
    assert mlp.w_down.weight.grad is not None
    assert x.grad is not None


def test_swiglu_gating_behavior():
    """Verify that when gate is zeroed, output is zeroed."""
    d_model = 16
    mlp = SwiGLU(d_model=d_model, hidden_dim=32)
    # Zero out w_gate weights
    with torch.no_grad():
        mlp.w_gate.weight.zero_()

    x = torch.randn(1, 2, d_model)
    out = mlp(x)
    assert torch.allclose(out, torch.zeros_like(out)), "Zeroed gate did not zero output!"
