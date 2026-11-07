"""
Tests for Milestone 46: Kahneman-Tversky Optimization (KTO) Alignment.
"""

import copy
import pytest
import torch

from model.gpt import GPT, GPTConfig
from training.kto import (
    KTOBatch,
    KTOLoss,
    KTOEngine,
    compute_sequence_log_probs,
)


@pytest.fixture
def tiny_gpt():
    config = GPTConfig(
        vocab_size=60,
        max_seq_len=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )
    return GPT(config)


def test_compute_sequence_log_probs(tiny_gpt):
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    completion = torch.tensor([[7, 8], [9, 10]])

    log_probs = compute_sequence_log_probs(tiny_gpt, prompt, completion)
    assert log_probs.shape == (2,)
    # Log probabilities are negative numbers
    assert (log_probs <= 0.0).all()


def test_kto_loss_behavior():
    loss_fn = KTOLoss(beta=0.1, desirable_weight=1.0, undesirable_weight=1.5)

    # Policy assigns higher probability to desirable sample (index 0)
    # and lower probability to undesirable sample (index 1)
    policy_log_probs = torch.tensor([-2.0, -10.0])
    ref_log_probs = torch.tensor([-5.0, -5.0])
    labels = torch.tensor([1, 0])

    loss, metrics = loss_fn(policy_log_probs, ref_log_probs, labels)

    assert loss.item() > 0.0
    assert "kto_loss" in metrics
    assert "loss_desirable" in metrics
    assert "loss_undesirable" in metrics
    assert "mean_reward" in metrics


def test_kto_engine_step(tiny_gpt):
    ref_gpt = copy.deepcopy(tiny_gpt)
    optimizer = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
    engine = KTOEngine(tiny_gpt, ref_gpt, optimizer, beta=0.1)

    batch = KTOBatch(
        prompt_ids=torch.randint(0, 50, (2, 4)),
        completion_ids=torch.randint(0, 50, (2, 4)),
        labels=torch.tensor([1, 0]),
    )

    metrics = engine.train_step(batch)
    assert "kto_loss" in metrics
    assert metrics["kto_loss"] > 0.0

    # Ensure reference model parameters remained frozen
    for p in ref_gpt.parameters():
        assert p.grad is None


def test_kto_all_desirable_or_undesirable():
    loss_fn = KTOLoss()
    p_lp = torch.tensor([-2.0, -3.0])
    r_lp = torch.tensor([-2.5, -3.5])

    # All desirable
    loss_des, m_des = loss_fn(p_lp, r_lp, torch.tensor([1, 1]))
    assert loss_des.item() > 0.0
    assert m_des["loss_undesirable"] == 0.0

    # All undesirable
    loss_undes, m_undes = loss_fn(p_lp, r_lp, torch.tensor([0, 0]))
    assert loss_undes.item() > 0.0
    assert m_undes["loss_desirable"] == 0.0
