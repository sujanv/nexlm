"""
Tests for Milestone 42: Model Merging Engine (SLERP & DARE).
"""

import pytest
import torch

from model.gpt import GPT, GPTConfig
from training.model_merging import (
    slerp_tensor,
    merge_models_slerp,
    dare_task_vector,
    merge_models_dare,
)


def test_slerp_tensor():
    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])

    # t = 0.5 should yield 45-degree vector normalized [sqrt(2)/2, sqrt(2)/2]
    mid = slerp_tensor(v1, v2, t=0.5)
    expected = torch.tensor([2**0.5 / 2, 2**0.5 / 2])
    assert torch.allclose(mid, expected, atol=1e-4)

    # Norm of slerp output should equal 1.0
    assert torch.allclose(torch.norm(mid), torch.tensor(1.0), atol=1e-4)


def test_merge_models_slerp():
    config = GPTConfig(vocab_size=50, max_seq_len=32, d_model=32, n_heads=2, n_layers=2)
    m1 = GPT(config)
    m2 = GPT(config)

    merged = merge_models_slerp(m1, m2, t=0.5)
    assert isinstance(merged, GPT)

    # Check forward pass runs smoothly
    x = torch.randint(0, 50, (2, 8))
    logits, _, _ = merged(x)
    assert logits.shape == (2, 8, 50)


def test_dare_task_vector_unbiased():
    # Test that DARE is an unbiased estimator: E[DARE(v)] == v
    v = torch.ones(100000)
    drop_rate = 0.3

    dare_v = dare_task_vector(v, drop_rate=drop_rate)

    # Percentage of dropped (zeroed) elements should be approximately drop_rate
    num_zeros = (dare_v == 0.0).sum().item()
    zero_ratio = num_zeros / len(v)
    assert abs(zero_ratio - drop_rate) < 0.02

    # Expectation of difference should be close to 0
    mean_diff = torch.abs(torch.mean(dare_v - v)).item()
    assert mean_diff < 0.02


def test_merge_models_dare():
    config = GPTConfig(vocab_size=50, max_seq_len=32, d_model=32, n_heads=2, n_layers=2)
    base = GPT(config)
    ft1 = GPT(config)
    ft2 = GPT(config)

    merged = merge_models_dare(base, [ft1, ft2], weights=[0.5, 0.5], drop_rate=0.2)
    assert isinstance(merged, GPT)

    x = torch.randint(0, 50, (1, 6))
    logits, _, _ = merged(x)
    assert logits.shape == (1, 6, 50)
