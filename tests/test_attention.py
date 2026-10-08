import torch
import pytest

from model.attention import SingleHeadCausalSelfAttention, MultiHeadCausalSelfAttention


def test_single_head_attention_shapes_and_weights():
    B, T, d_in, d_out = 2, 4, 16, 8
    attn = SingleHeadCausalSelfAttention(d_in=d_in, d_out=d_out, max_seq_len=8)
    x = torch.randn(B, T, d_in)

    out, weights = attn(x)

    # Output shape: (B, T, d_out)
    assert out.shape == (B, T, d_out)
    # Weights shape: (B, T, T)
    assert weights.shape == (B, T, T)

    # Weights must sum to 1.0 along the key dimension (rows)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Strictly check causal mask: token at row i cannot attend to any column j > i
    for i in range(T):
        for j in range(i + 1, T):
            assert torch.all(weights[:, i, j] == 0.0), f"Token {i} attended to future token {j}!"


def test_single_head_causal_no_future_leakage_gradient():
    """Verify backprop does not flow from future tokens into past positions."""
    B, T, d_in, d_out = 1, 5, 12, 12
    attn = SingleHeadCausalSelfAttention(d_in=d_in, d_out=d_out, max_seq_len=8)
    attn.eval()

    x = torch.randn(B, T, d_in, requires_grad=True)
    out, _ = attn(x)

    # Loss computed only from token position 1
    target_pos = 1
    loss = out[0, target_pos, :].sum()
    loss.backward()

    # Positions > target_pos MUST have zero gradient
    future_grads = x.grad[0, target_pos + 1 :, :]
    assert torch.all(future_grads == 0.0), "Future tokens leaked backward into earlier positions!"


def test_multi_head_attention_shapes_and_residuals():
    B, T, d_model = 2, 8, 32
    n_heads = 4
    mha = MultiHeadCausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=16)

    x = torch.randn(B, T, d_model)
    out, cache = mha(x, use_cache=True)

    assert out.shape == (B, T, d_model)
    assert cache is not None
    k, v = cache
    assert k.shape == (B, n_heads, T, d_model // n_heads)
    assert v.shape == (B, n_heads, T, d_model // n_heads)
