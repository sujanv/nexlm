import torch
import pytest

from nexlm.model.transformer import MLP, TransformerBlock


def test_mlp_shapes():
    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C)
    mlp = MLP(dim=C, dropout=0.0)
    out = mlp(x)
    assert out.shape == (B, T, C)


def test_transformer_block_forward():
    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C)
    block = TransformerBlock(dim=C, n_heads=4, max_seq_len=16, dropout=0.0)

    out, cache = block(x, use_cache=True)
    assert out.shape == (B, T, C)
    assert cache is not None
    assert cache[0].shape == (B, 4, T, 8)
