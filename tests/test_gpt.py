import torch
import pytest

from nexlm.config import NexLMConfig
from nexlm.model.gpt import NexLM


def test_nexlm_forward_and_loss():
    config = NexLMConfig(
        vocab_size=100,
        max_seq_len=32,
        dim=32,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
    )
    model = NexLM(config)

    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    # Without targets
    logits, loss, _ = model(input_ids)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None

    # With targets
    logits, loss, _ = model(input_ids, targets=targets)
    assert loss is not None
    assert loss.item() > 0.0

    # Test backward pass
    loss.backward()
    assert model.lm_head.weight.grad is not None


def test_nexlm_weight_tying():
    config = NexLMConfig(
        vocab_size=100,
        max_seq_len=32,
        dim=32,
        n_layers=2,
        n_heads=2,
        tie_word_embeddings=True,
    )
    model = NexLM(config)
    assert model.lm_head.weight is model.embeddings.token_embedding.weight
