import torch
import pytest

from model.gpt import GPT, GPTConfig
from inference.prefix_cache import PrefixCacheManager, prefill_with_prefix_cache, hash_token_prefix


def test_prefix_cache_lru_and_hashing():
    manager = PrefixCacheManager(max_cached_prefixes=2)

    tokens1 = [1, 2, 3]
    tokens2 = [4, 5, 6]
    tokens3 = [7, 8, 9]

    dummy_cache = [(torch.zeros(1), torch.zeros(1))]

    manager.put(tokens1, dummy_cache)
    manager.put(tokens2, dummy_cache)

    assert manager.get(tokens1) is not None
    assert manager.hits == 1

    # Adding a 3rd should evict tokens2 (since tokens1 was recently accessed via get)
    manager.put(tokens3, dummy_cache)
    assert manager.get(tokens2) is None
    assert manager.get(tokens1) is not None
    assert manager.get(tokens3) is not None


def test_prefill_prefix_cache_hit_and_miss():
    config = GPTConfig(vocab_size=50, max_seq_len=32, d_model=32, n_heads=2, n_layers=2)
    model = GPT(config)
    manager = PrefixCacheManager(max_cached_prefixes=4)

    system_prefix = [1, 2, 3, 4]
    user_query1 = system_prefix + [10, 11]
    user_query2 = system_prefix + [20, 21]

    prompt1 = torch.tensor([user_query1], dtype=torch.long)
    prompt2 = torch.tensor([user_query2], dtype=torch.long)

    # 1. First query: cache miss (must prefill and cache prefix)
    logits1, cache1, hit1 = prefill_with_prefix_cache(
        model, prompt1, manager, system_prefix_len=len(system_prefix)
    )
    assert not hit1

    # 2. Second query: cache hit (reuses prefix KV cache)
    logits2, cache2, hit2 = prefill_with_prefix_cache(
        model, prompt2, manager, system_prefix_len=len(system_prefix)
    )
    assert hit2
    assert manager.hits == 1
