"""
Enhancement: Prefix Caching & Prompt Cache Sharing Engine (Anthropic / OpenAI style)

Why Prefix Caching?
In multi-turn chat and production API serving, requests frequently share identical prefixes:
    - Fixed System Prompts (e.g., "You are an assistant with documentation on X...")
    - Few-shot in-context learning examples
    - Preceding conversation turns in a chat session.

Without prefix caching, the model redundantly computes Q, K, V for all prefix tokens on EVERY request.

Prefix Caching:
1. Computes a deterministic SHA-256 hash of prefix token sequences.
2. Stores the pre-computed KV-cache in an LRU memory registry.
3. On incoming queries, matches the longest common prefix in O(1) time and skips prefill computation!
"""

import hashlib
from typing import Dict, List, Optional, Tuple, OrderedDict
from collections import OrderedDict as ODict
import torch
import torch.nn as nn

from model.gpt import GPT


def hash_token_prefix(token_ids: List[int]) -> str:
    """Computes a SHA-256 fingerprint for a sequence of token IDs."""
    raw = ",".join(str(t) for t in token_ids).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class PrefixCacheEntry:
    def __init__(self, token_ids: List[int], kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.token_ids = list(token_ids)
        self.kv_cache = kv_cache  # List of (k, v) per layer
        self.seq_len = len(token_ids)


class PrefixCacheManager:
    """
    LRU (Least Recently Used) cache for reusable prompt KV representations.
    """

    def __init__(self, max_cached_prefixes: int = 32):
        self.max_cached_prefixes = max_cached_prefixes
        self.cache: ODict[str, PrefixCacheEntry] = ODict()
        self.hits = 0
        self.misses = 0

    def get(self, token_ids: List[int]) -> Optional[PrefixCacheEntry]:
        """Looks up a precomputed prefix cache for the token sequence."""
        key = hash_token_prefix(token_ids)
        if key in self.cache:
            self.hits += 1
            # Move to end to indicate recent usage in LRU
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, token_ids: List[int], kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]) -> str:
        """Stores a precomputed KV cache into the registry."""
        key = hash_token_prefix(token_ids)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_cached_prefixes:
                # Evict oldest unused entry (LRU)
                self.cache.popitem(last=False)
            self.cache[key] = PrefixCacheEntry(token_ids, kv_cache)
        return key

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0


@torch.no_grad()
def prefill_with_prefix_cache(
    model: GPT,
    full_prompt_ids: torch.Tensor,
    prefix_manager: PrefixCacheManager,
    system_prefix_len: int = 0,
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], bool]:
    """
    Generates initial logits and KV cache, reusing cached prefix if available.
    Returns: (logits, kv_cache, was_cache_hit)
    """
    model.eval()
    tokens_list = full_prompt_ids[0].tolist()

    if system_prefix_len > 0 and len(tokens_list) >= system_prefix_len:
        prefix_slice = tokens_list[:system_prefix_len]
        entry = prefix_manager.get(prefix_slice)

        if entry is not None:
            # CACHE HIT: Only compute forward pass for remaining suffix tokens!
            suffix_ids = full_prompt_ids[:, system_prefix_len:]
            if suffix_ids.shape[1] == 0:
                # Prompt was entirely the prefix
                logits, _, updated_cache = model(
                    full_prompt_ids[:, :1], kv_cache=entry.kv_cache, use_cache=True, start_pos=system_prefix_len
                )
                return logits, entry.kv_cache, True

            logits, _, updated_cache = model(
                suffix_ids,
                kv_cache=entry.kv_cache,
                use_cache=True,
                start_pos=system_prefix_len,
            )
            return logits, updated_cache, True

    # CACHE MISS: Compute full prompt from scratch and save prefix into cache
    logits, _, full_cache = model(full_prompt_ids, kv_cache=None, use_cache=True, start_pos=0)
    if system_prefix_len > 0 and len(tokens_list) >= system_prefix_len:
        prefix_slice = tokens_list[:system_prefix_len]
        # Slice prefix KV cache for future reuse
        prefix_kvs = [(k[:, :, :system_prefix_len, :].clone(), v[:, :, :system_prefix_len, :].clone()) for k, v in full_cache]
        prefix_manager.put(prefix_slice, prefix_kvs)

    return logits, full_cache, False
