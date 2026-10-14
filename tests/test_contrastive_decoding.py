"""
Tests for Milestone 41: Contrastive Decoding & Classifier-Free Guidance (CFG).
"""

import pytest
import torch

from model.gpt import GPT, GPTConfig
from inference.contrastive_decoding import (
    apply_contrastive_penalty,
    apply_cfg_guidance,
    generate_contrastive,
    generate_cfg,
)


def test_apply_contrastive_penalty():
    vocab_size = 10
    expert_logits = torch.tensor([[10.0, 5.0, 1.0, 0.0, -2.0, -5.0, 0.0, 0.0, 0.0, 0.0]])
    amateur_logits = torch.tensor([[5.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    penalized = apply_contrastive_penalty(expert_logits, amateur_logits, alpha=0.5, beta=0.01)
    assert penalized.shape == expert_logits.shape

    # Token 0: expert was 10, amateur was 5 -> 10 - 2.5 = 7.5
    # Token 1: expert was 5, amateur was 8 -> 5 - 4.0 = 1.0
    # Token 0 is favored much more strongly now!
    assert penalized[0, 0] > penalized[0, 1]


def test_apply_cfg_guidance():
    cond_logits = torch.tensor([[5.0, 2.0]])
    uncond_logits = torch.tensor([[1.0, 1.0]])

    # Scale = 1.0 returns cond_logits
    res_1 = apply_cfg_guidance(cond_logits, uncond_logits, cfg_scale=1.0)
    assert torch.allclose(res_1, cond_logits)

    # Scale = 2.0 pushes conditioned difference: uncond + 2 * (cond - uncond)
    # [1 + 2*(5-1), 1 + 2*(2-1)] = [9.0, 3.0]
    res_2 = apply_cfg_guidance(cond_logits, uncond_logits, cfg_scale=2.0)
    assert torch.allclose(res_2, torch.tensor([[9.0, 3.0]]))


def test_generate_contrastive_end_to_end():
    config = GPTConfig(vocab_size=50, max_seq_len=64, d_model=32, n_heads=2, n_layers=2)
    expert = GPT(config)
    amateur = GPT(config)

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    output = generate_contrastive(expert, amateur, prompt, max_new_tokens=6, alpha=0.3)

    assert output.shape == (1, 9)
    assert (output[:, :3] == prompt).all()


def test_generate_cfg_end_to_end():
    config = GPTConfig(vocab_size=50, max_seq_len=64, d_model=32, n_heads=2, n_layers=2)
    model = GPT(config)

    prompt = torch.tensor([[4, 5, 6]], dtype=torch.long)
    output = generate_cfg(model, prompt, cfg_scale=1.5, max_new_tokens=5)

    assert output.shape == (1, 8)
    assert (output[:, :3] == prompt).all()
