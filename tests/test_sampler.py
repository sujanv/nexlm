import torch
import pytest

from nexlm.inference.sampler import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)


def test_temperature_scaling():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    scaled = apply_temperature(logits, temperature=0.5)
    assert torch.allclose(scaled, torch.tensor([[2.0, 4.0, 6.0]]))


def test_top_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = apply_top_k(logits, top_k=2)
    # The top 2 values are 5.0 and 4.0; all others should be -inf
    assert filtered[0, 1] == 5.0
    assert filtered[0, 4] == 4.0
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_top_p():
    # Large gap ensures clear probability partition
    logits = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    filtered = apply_top_p(logits, top_p=0.8)
    assert not torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_sample_next_token_greedy():
    logits = torch.tensor([[1.0, 10.0, 2.0]])
    token = sample_next_token(logits, temperature=0.0)
    assert token.item() == 1
