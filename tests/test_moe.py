import torch
import pytest

from model.moe import TopKRouter, MoEFeedForward


def test_topk_router():
    B, T, d_model = 2, 6, 32
    num_experts, top_k = 4, 2
    router = TopKRouter(d_model=d_model, num_experts=num_experts, top_k=top_k)

    x = torch.randn(B, T, d_model)
    weights, indices, aux_loss = router(x)

    assert weights.shape == (B, T, top_k)
    assert indices.shape == (B, T, top_k)

    # Weights must sum to 1.0 per token
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    # Auxiliary loss must be a scalar >= 0
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0.0


def test_moe_feedforward_forward_and_backward():
    B, T, d_model = 2, 4, 32
    moe = MoEFeedForward(d_model=d_model, num_experts=4, top_k=2)

    x = torch.randn(B, T, d_model, requires_grad=True)
    out, aux_loss = moe(x)

    assert out.shape == (B, T, d_model)

    loss = out.sum() + 0.1 * aux_loss
    loss.backward()

    assert x.grad is not None
    assert moe.router.gate.weight.grad is not None
    # At least some experts should have non-zero gradients
    expert_grads = [exp.c_fc.weight.grad is not None for exp in moe.experts]
    assert any(expert_grads)
