"""
Tests for Milestone 48: Advanced Sampler (Min-P, Typical, Penalties).
"""

import pytest
import torch

from inference.advanced_sampler import (
    apply_frequency_presence_penalties,
    sample_min_p,
    sample_typical,
    advanced_sample,
)


def test_frequency_and_presence_penalties():
    logits = torch.zeros(1, 10)
    # Context has repeated token 3 twice, and token 5 once
    context = [3, 3, 5]

    penalized = apply_frequency_presence_penalties(
        logits,
        context_tokens=context,
        frequency_penalty=1.0,
        presence_penalty=0.5,
    )

    # Token 3 penalty: 1.0 * 2 + 0.5 * 1 = 2.5 -> logit should be -2.5
    assert torch.isclose(penalized[0, 3], torch.tensor(-2.5))
    # Token 5 penalty: 1.0 * 1 + 0.5 * 1 = 1.5 -> logit should be -1.5
    assert torch.isclose(penalized[0, 5], torch.tensor(-1.5))
    # Token 0 untouched: logit 0.0
    assert torch.isclose(penalized[0, 0], torch.tensor(0.0))


def test_min_p_sampling():
    # Token 0 is dominant (prob ~0.8)
    # Token 1 has prob ~0.18
    # Token 2 has tiny prob ~0.02
    logits = torch.tensor([[10.0, 8.5, 5.0]])

    # With min_p = 0.1, threshold = 0.1 * P(0)
    # P(2) is way below threshold and should never be sampled
    samples = [sample_min_p(logits, min_p=0.1, temperature=1.0).item() for _ in range(50)]
    assert 2 not in samples
    assert 0 in samples


def test_typical_sampling():
    logits = torch.tensor([[5.0, 4.0, 3.0, 0.5, 0.1]])
    sample = sample_typical(logits, typical_p=0.8, temperature=1.0)
    assert sample.shape == (1, 1)
    assert 0 <= sample.item() < 5


def test_advanced_sample_fallback_and_combinations():
    logits = torch.randn(2, 20)
    # 1. Greedy fallback
    greedy = advanced_sample(logits, temperature=0.0)
    assert greedy.shape == (2, 1)
    assert (greedy == torch.argmax(logits, dim=-1, keepdim=True)).all()

    # 2. Min-P combined with penalties
    min_p_sample = advanced_sample(
        logits,
        context_tokens=[1, 2, 3],
        temperature=0.8,
        min_p=0.05,
        frequency_penalty=0.5,
    )
    assert min_p_sample.shape == (2, 1)

    # 3. Typical combined with presence penalty
    typ_sample = advanced_sample(
        logits,
        context_tokens=[4, 5],
        temperature=0.8,
        typical_p=0.85,
        presence_penalty=0.5,
    )
    assert typ_sample.shape == (2, 1)
