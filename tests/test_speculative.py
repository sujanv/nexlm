import torch
import pytest

from model.gpt import GPT, GPTConfig
from inference.speculative import speculative_generate


def test_speculative_decoding_generation():
    torch.manual_seed(42)
    vocab_size = 50

    # Target model: 4 layers, d_model=64
    target_config = GPTConfig(vocab_size=vocab_size, max_seq_len=64, d_model=64, n_heads=4, n_layers=4)
    target_model = GPT(target_config)

    # Draft model: 2 layers, d_model=64 (fast approximation)
    draft_config = GPTConfig(vocab_size=vocab_size, max_seq_len=64, d_model=64, n_heads=4, n_layers=2)
    draft_model = GPT(draft_config)

    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    max_new_tokens = 12

    out_tokens, accept_rate = speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        input_ids=prompt,
        max_new_tokens=max_new_tokens,
        gamma=3,
        temperature=1.0,
    )

    # Assert new tokens were appended
    assert out_tokens.shape[1] > prompt.shape[1]
    assert 0.0 <= accept_rate <= 1.0
