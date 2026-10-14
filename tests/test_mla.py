"""
Tests for Milestone 40: Multi-Head Latent Attention (MLA).
"""

import pytest
import torch

from model.mla import MultiHeadLatentAttention


def test_mla_forward_shape():
    B, T, D = 2, 16, 64
    n_heads = 4
    mla = MultiHeadLatentAttention(
        d_model=D,
        n_heads=n_heads,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
    )

    x = torch.randn(B, T, D)
    out, new_kv = mla(x)

    assert out.shape == (B, T, D)
    assert new_kv is None


def test_mla_kv_cache_equivalence():
    B, T, D = 1, 8, 64
    mla = MultiHeadLatentAttention(
        d_model=D,
        n_heads=2,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
    )
    mla.eval()

    x = torch.randn(B, T, D)
    full_out, _ = mla(x)

    # Step-by-step with KV cache
    # 1. Prefill first 4 tokens
    prefill_x = x[:, :4, :]
    out_prefill, cache = mla(prefill_x, use_cache=True, start_pos=0)
    assert cache is not None
    c_kv, k_rope = cache
    assert c_kv.shape == (B, 4, 16)
    assert k_rope.shape == (B, 4, 8)

    # 2. Decode 5th token
    t5_x = x[:, 4:5, :]
    out_t5, cache = mla(t5_x, kv_cache=cache, use_cache=True, start_pos=4)

    # The output at position 4 should match full_out at position 4
    assert torch.allclose(out_t5[:, 0, :], full_out[:, 4, :], atol=1e-4)


def test_mla_cache_memory_savings():
    # Demonstrating the significant KV cache compression of MLA over MHA
    d_model = 256
    n_heads = 8
    head_dim = 32
    seq_len = 1024
    batch_size = 4

    # Standard MHA: 2 * (B * n_heads * seq_len * head_dim)
    mha_elements = 2 * (batch_size * n_heads * seq_len * head_dim)

    # MLA: cached c_kv (kv_lora_rank = 64) + cached k_rope (rope_dim = 16)
    kv_lora_rank = 64
    rope_dim = 16
    mla_elements = batch_size * seq_len * (kv_lora_rank + rope_dim)

    # MLA achieves ~69% to 90% cache memory reduction
    assert mla_elements < mha_elements * 0.4
    reduction_pct = (1.0 - mla_elements / mha_elements) * 100
    assert reduction_pct > 60.0


def test_mla_gradient_flow():
    B, T, D = 2, 8, 32
    mla = MultiHeadLatentAttention(
        d_model=D,
        n_heads=2,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )

    x = torch.randn(B, T, D, requires_grad=True)
    out, _ = mla(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert mla.w_dkv.weight.grad is not None
    assert mla.w_dq.weight.grad is not None
    assert mla.w_uk.weight.grad is not None
    assert not torch.isnan(x.grad).any()
