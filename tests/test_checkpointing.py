import torch
import torch.nn as nn
import pytest

from model.transformer import TransformerBlock
from training.checkpointing import CheckpointedTransformerBlockList


def test_gradient_checkpointing_forward_and_backward():
    d_model, n_heads, seq_len = 32, 2, 8
    blocks = nn.ModuleList([
        TransformerBlock(d_model=d_model, n_heads=n_heads, max_seq_len=16)
        for _ in range(2)
    ])

    wrapped = CheckpointedTransformerBlockList(blocks, enabled=True)
    wrapped.train()

    x = torch.randn(2, seq_len, d_model, requires_grad=True)
    out, _ = wrapped(x)

    assert out.shape == (2, seq_len, d_model)

    loss = out.sum()
    loss.backward()

    # All block parameters and input tensor must have valid gradients
    assert x.grad is not None
    for block in blocks:
        assert block.attn.c_attn.weight.grad is not None
        assert block.mlp.c_fc.weight.grad is not None
