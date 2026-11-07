"""
Tests for Milestone 50: Chunked Prefill & Attention Sinks (StreamingLLM).
"""

import pytest
import torch

from model.gpt import GPT, GPTConfig
from inference.chunked_prefill import chunked_prefill, AttentionSinkKVCache


@pytest.fixture
def tiny_gpt():
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=256,
        d_model=64,
        n_heads=2,
        n_layers=2,
    )
    return GPT(config)


def test_chunked_prefill_equivalence(tiny_gpt):
    tiny_gpt.eval()
    prompt = torch.randint(0, 100, (1, 64))

    # Standard full forward pass
    with torch.no_grad():
        full_logits, _, full_kv = tiny_gpt(prompt, use_cache=True)
    expected_last_logits = full_logits[:, -1:, :]

    # Chunked prefill with chunk_size = 16 (4 chunks total)
    chunked_logits, chunked_kv = chunked_prefill(tiny_gpt, prompt, chunk_size=16)

    # Logits at the end of prompt must match full pass
    assert torch.allclose(expected_last_logits, chunked_logits, atol=1e-4)

    # Total tokens in KV cache must match full length (64)
    for layer_k, layer_v in chunked_kv:
        assert layer_k.size(2) == 64
        assert layer_v.size(2) == 64


def test_attention_sink_kv_pruning():
    sink_manager = AttentionSinkKVCache(num_sink_tokens=2, window_size=4)

    # Fake cache with 10 tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Sinks: [0, 1]
    # Recent 4: [6, 7, 8, 9]
    # Max allowed = 2 + 4 = 6 tokens
    B, H, T, D = 1, 2, 10, 16
    k = torch.arange(T).float().view(1, 1, T, 1).expand(B, H, T, D)
    v = k.clone()
    caches = [(k, v)]

    pruned = sink_manager.evict_and_retain(caches)
    pruned_k, pruned_v = pruned[0]

    # Length must be 2 + 4 = 6
    assert pruned_k.size(2) == 6
    assert pruned_v.size(2) == 6

    # Verify retained token indices: [0, 1] + [6, 7, 8, 9]
    retained_indices = pruned_k[0, 0, :, 0].tolist()
    assert retained_indices == [0.0, 1.0, 6.0, 7.0, 8.0, 9.0]


def test_attention_sink_no_op_when_under_capacity():
    sink_manager = AttentionSinkKVCache(num_sink_tokens=4, window_size=16)

    B, H, T, D = 1, 2, 10, 16  # 10 < 4 + 16 (20)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    caches = [(k, v)]

    pruned = sink_manager.evict_and_retain(caches)
    assert pruned[0][0].shape == (B, H, 10, D)
