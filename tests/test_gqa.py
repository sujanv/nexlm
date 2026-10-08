import torch
import pytest

from model.gqa import GroupedQueryAttention, repeat_kv


def test_repeat_kv():
    B, n_kv_heads, T, head_dim = 2, 2, 4, 16
    n_rep = 3
    x = torch.randn(B, n_kv_heads, T, head_dim)
    repeated = repeat_kv(x, n_rep)

    assert repeated.shape == (B, n_kv_heads * n_rep, T, head_dim)
    # Verify values match across group
    assert torch.equal(repeated[:, 0, :, :], x[:, 0, :, :])
    assert torch.equal(repeated[:, 1, :, :], x[:, 0, :, :])
    assert torch.equal(repeated[:, 2, :, :], x[:, 0, :, :])
    assert torch.equal(repeated[:, 3, :, :], x[:, 1, :, :])


def test_gqa_shapes_and_cache_compression():
    B, T, d_model = 2, 8, 64
    n_heads = 8
    n_kv_heads = 2  # 4x compression in KV cache

    gqa = GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=16,
    )
    x = torch.randn(B, T, d_model)

    out, cache = gqa(x, use_cache=True)
    assert out.shape == (B, T, d_model)
    assert cache is not None

    k, v = cache
    # Verify KV cache only stores n_kv_heads (2), not n_heads (8)
    assert k.shape == (B, n_kv_heads, T, d_model // n_heads)
    assert v.shape == (B, n_kv_heads, T, d_model // n_heads)


def test_mqa_single_kv_head():
    """Multi-Query Attention: All query heads share exactly 1 KV head."""
    B, T, d_model = 1, 6, 32
    gqa = GroupedQueryAttention(
        d_model=d_model,
        n_heads=4,
        n_kv_heads=1,
        max_seq_len=16,
    )
    x = torch.randn(B, T, d_model)
    out, cache = gqa(x, use_cache=True)

    assert out.shape == (B, T, d_model)
    k, _ = cache
    assert k.shape == (B, 1, T, d_model // 4)
