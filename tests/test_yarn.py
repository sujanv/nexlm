"""
Tests for Milestone 39: YaRN RoPE Context Scaling.
"""

import pytest
import torch

from model.yarn import (
    YaRNScaledRotaryEmbedding,
    apply_yarn_rope,
    find_correction_range,
    linear_ramp_mask,
)


def test_find_correction_range():
    low, high = find_correction_range(32.0, 1.0, dim=64, base=10000.0, max_seq_len=256)
    assert 0 <= low <= high < 64


def test_linear_ramp_mask():
    ramp = linear_ramp_mask(2.0, 6.0, dim=10)
    assert ramp.shape == (10,)
    assert ramp[0] == 0.0
    assert ramp[-1] == 1.0
    assert (ramp >= 0.0).all() and (ramp <= 1.0).all()


def test_yarn_embedding_shape_and_scaling():
    dim = 32
    max_seq_len = 512
    orig_len = 128
    scale = 4.0

    yarn = YaRNScaledRotaryEmbedding(
        dim=dim,
        max_seq_len=max_seq_len,
        original_max_seq_len=orig_len,
        scale=scale,
    )

    cos, sin = yarn(seq_len=256, device=torch.device("cpu"))
    assert cos.shape == (1, 1, 256, dim)
    assert sin.shape == (1, 1, 256, dim)
    assert yarn.mscale > 1.0  # Temperature scaling factor applied


def test_apply_yarn_rope():
    B, H, T, D = 2, 4, 128, 32
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    yarn = YaRNScaledRotaryEmbedding(dim=D, max_seq_len=T, original_max_seq_len=64, scale=2.0)
    cos, sin = yarn(seq_len=T, device=q.device)

    q_rot, k_rot = apply_yarn_rope(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # Check that rotation altered tensor values without NaNs
    assert not torch.equal(q, q_rot)
    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()
