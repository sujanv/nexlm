"""
Tests for Milestone 34: Multi-Token Prediction (MTP).
"""

import pytest
import torch

from model.gpt import GPT, GPTConfig
from model.mtp import MultiTokenPredictionModel, MTPModule


@pytest.fixture
def base_gpt():
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )
    return GPT(config)


def test_mtp_module_shape():
    d_model = 32
    vocab_size = 100
    module = MTPModule(d_model=d_model, vocab_size=vocab_size)
    h = torch.randn(2, 8, d_model)
    out = module(h)
    assert out.shape == (2, 8, vocab_size)


def test_mtp_forward_and_loss(base_gpt):
    mtp_model = MultiTokenPredictionModel(base_model=base_gpt, num_future_tokens=2, aux_loss_weight=0.5)

    input_ids = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    logits, total_loss, loss_dict = mtp_model(input_ids, targets=targets)

    assert logits.shape == (2, 16, 100)
    assert total_loss is not None
    assert "primary_loss" in loss_dict
    assert "aux_loss_k2" in loss_dict
    assert "aux_loss_k3" in loss_dict
    assert total_loss.item() > 0


def test_mtp_gradient_backprop(base_gpt):
    mtp_model = MultiTokenPredictionModel(base_model=base_gpt, num_future_tokens=1)
    input_ids = torch.randint(0, 100, (2, 12))
    targets = torch.randint(0, 100, (2, 12))

    _, total_loss, _ = mtp_model(input_ids, targets=targets)
    total_loss.backward()

    # Check base model and auxiliary head receive gradients
    assert base_gpt.tok_emb.weight.grad is not None
    assert mtp_model.aux_heads[0].head.weight.grad is not None


def test_mtp_speculative_draft_predictions(base_gpt):
    mtp_model = MultiTokenPredictionModel(base_model=base_gpt, num_future_tokens=2)
    input_ids = torch.randint(0, 100, (1, 8))

    predictions = mtp_model.predict_multi_tokens(input_ids)
    # 1 primary token + 2 future tokens = 3 total tokens predicted
    assert len(predictions) == 3
    for pred in predictions:
        assert pred.shape == (1,)
        assert 0 <= pred.item() < 100
