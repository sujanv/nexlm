import torch
import pytest

from model.sliding_window import SlidingWindowAttention, RollingKVCache


def test_sliding_window_shapes():
    B, T, d_model = 2, 16, 32
    swa = SlidingWindowAttention(d_model=d_model, n_heads=4, window_size=8)
    x = torch.randn(B, T, d_model)

    out, _ = swa(x)
    assert out.shape == (B, T, d_model)


def test_rolling_kv_cache_bounded_memory():
    """Verify that buffer size never exceeds window_size regardless of step count."""
    window_size = 6
    n_heads = 2
    head_dim = 8
    cache = RollingKVCache(window_size=window_size, n_heads=n_heads, head_dim=head_dim, batch_size=1)

    # Simulate generating 20 tokens (much larger than window_size = 6)
    for step in range(20):
        k = torch.randn(1, n_heads, 1, head_dim)
        v = torch.randn(1, n_heads, 1, head_dim)
        k_act, v_act = cache.update(k, v)

        # Buffer length must be bounded by window_size
        assert k_act.shape[2] <= window_size
        assert v_act.shape[2] <= window_size

    assert cache.num_tokens == 20
    assert cache.k_buffer.shape[2] == window_size
