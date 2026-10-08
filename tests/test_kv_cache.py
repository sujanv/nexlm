import torch
import pytest

from nexlm.config import NexLMConfig
from nexlm.model.gpt import NexLM
from nexlm.inference.generate import generate


def test_kv_cache_equivalence():
    """
    Greedy decoding with KV cache MUST produce the exact same
    token sequence as greedy decoding without KV cache.
    """
    torch.manual_seed(42)
    config = NexLMConfig(
        vocab_size=64,
        max_seq_len=64,
        dim=32,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
    )
    model = NexLM(config)
    model.eval()

    prompt = torch.tensor([[1, 5, 12, 8]], dtype=torch.long)
    max_new_tokens = 10

    # 1. Generate without KV cache
    tokens_no_cache = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_kv_cache=False,
    )

    # 2. Generate with KV cache
    tokens_with_cache = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_kv_cache=True,
    )

    # Assert exact match
    assert torch.equal(tokens_no_cache, tokens_with_cache), (
        f"Mismatch between cache and no-cache generation!\n"
        f"Without cache: {tokens_no_cache.tolist()}\n"
        f"With cache:    {tokens_with_cache.tolist()}"
    )
