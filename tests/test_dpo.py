import torch
import pytest

from model.gpt import GPT, GPTConfig
from training.dpo import compute_log_probs, dpo_loss, DPOTrainer


def test_dpo_loss_mechanics():
    # Synthetic log probs where policy prefers chosen heavily
    policy_chosen = torch.tensor([-2.0, -1.5])
    policy_rejected = torch.tensor([-6.0, -5.0])
    ref_chosen = torch.tensor([-3.0, -3.0])
    ref_rejected = torch.tensor([-3.0, -3.0])

    loss, chosen_r, rejected_r = dpo_loss(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected,
        beta=0.1,
    )

    # Chosen reward should be strictly greater than rejected reward
    assert torch.all(chosen_r > rejected_r)
    assert loss.item() > 0.0


def test_dpo_trainer_step():
    config = GPTConfig(vocab_size=30, max_seq_len=16, d_model=32, n_heads=2, n_layers=2)
    policy = GPT(config)
    ref = GPT(config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    trainer = DPOTrainer(policy_model=policy, ref_model=ref, optimizer=optimizer, beta=0.1)

    B, T = 2, 8
    chosen_ids = torch.randint(0, 30, (B, T))
    chosen_labels = chosen_ids.clone()
    rejected_ids = torch.randint(0, 30, (B, T))
    rejected_labels = rejected_ids.clone()

    metrics = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)

    assert "loss" in metrics
    assert "reward_margin" in metrics
    # Reference model parameters must remain completely frozen
    for p in ref.parameters():
        assert p.grad is None
