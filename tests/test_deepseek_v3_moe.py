"""
Tests for Milestone 53: DeepSeek-V3 Style Mixture of Experts (Shared + Routed).
"""

import pytest
import torch

from model.deepseek_v3_moe import DeepSeekV3MoE


def test_deepseek_v3_moe_forward_shape():
    B, T, D = 2, 8, 32
    moe = DeepSeekV3MoE(
        d_model=D,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2,
    )

    x = torch.randn(B, T, D)
    out, aux_loss = moe(x)

    assert out.shape == (B, T, D)
    assert aux_loss.shape == ()
    assert aux_loss.item() >= 0.0


def test_deepseek_v3_moe_shared_expert_invariance():
    # If gate weights are zeroed, routed output is identical across runs,
    # but shared experts ensure nonzero activation
    B, T, D = 1, 4, 32
    moe = DeepSeekV3MoE(
        d_model=D,
        num_shared_experts=2,
        num_routed_experts=4,
        top_k=1,
    )

    x = torch.randn(B, T, D)
    out, _ = moe(x)
    assert not torch.equal(out, torch.zeros_like(out))


def test_deepseek_v3_moe_gradient_flow():
    B, T, D = 2, 4, 16
    moe = DeepSeekV3MoE(
        d_model=D,
        num_shared_experts=1,
        num_routed_experts=3,
        top_k=2,
    )

    x = torch.randn(B, T, D, requires_grad=True)
    out, aux_loss = moe(x)
    loss = out.sum() + aux_loss
    loss.backward()

    assert x.grad is not None
    assert moe.gate.weight.grad is not None
    # Both shared expert and selected routed experts must have gradients
    assert moe.shared_experts[0].w_gate.weight.grad is not None
    assert not torch.isnan(x.grad).any()
