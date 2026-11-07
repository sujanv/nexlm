"""
Milestone 50: Long-Context Chunked Prefill & Attention Sink Engine (StreamingLLM)

1. Chunked Prefill (Sarathi / vLLM):
   Splits large prompts into fixed-size chunks (e.g. 128 or 256 tokens).
   Prefills sequentially into the KV cache, preventing OOM spikes from O(T^2) attention
   and interleaving prefill with decoding to minimize Inter-Token Latency (ITL).

2. Attention Sinks (StreamingLLM - Xiao et al., 2023):
   Language models concentrate massive attention weights on the first few tokens (attention sinks)
   regardless of semantic relevance.
   StreamingKVCache retains:
       [Sink Tokens (first k tokens)] + [Recent Rolling Window (last W tokens)]
   Allows infinite continuous generation without perplexity explosion or recomputation.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from model.gpt import GPT


def chunked_prefill(
    model: GPT,
    prompt_ids: torch.Tensor,
    chunk_size: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Prefills prompt tokens in small chunks of size chunk_size into the model's KV cache.
    Args:
        model: GPT model
        prompt_ids: (1, seq_len)
        chunk_size: maximum tokens to process in a single forward pass
    Returns:
        final_logits: (1, 1, vocab_size) logits for the very last prompt token
        kv_cache: fully accumulated KV cache across all layers
    """
    device = device or prompt_ids.device
    model.eval()
    total_tokens = prompt_ids.size(1)

    kv_cache = None
    start_pos = 0

    with torch.no_grad():
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk = prompt_ids[:, start_idx:end_idx].to(device)

            logits, _, new_kv = model(
                chunk,
                kv_cache=kv_cache,
                use_cache=True,
                start_pos=start_pos,
            )
            kv_cache = new_kv
            start_pos += chunk.size(1)

    last_logits = logits[:, -1:, :]
    return last_logits, kv_cache


class AttentionSinkKVCache:
    """
    KV Cache manager implementing StreamingLLM Attention Sinks.
    Maintains [first num_sink_tokens] + [most recent window_size tokens].
    """

    def __init__(self, num_sink_tokens: int = 4, window_size: int = 256):
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size

    def evict_and_retain(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prunes past keys and values if total tokens exceed num_sink_tokens + window_size.
        Input shapes per layer: (k, v) where k, v are (batch_size, n_heads, seq_len, head_dim)
        """
        if not kv_caches:
            return kv_caches

        # Check sequence length from the first layer's key
        sample_k = kv_caches[0][0]
        seq_len = sample_k.size(2)
        max_allowed = self.num_sink_tokens + self.window_size

        if seq_len <= max_allowed:
            return kv_caches

        # Prune middle tokens: keep [0 : sink] and [-window_size :]
        new_caches = []
        for k, v in kv_caches:
            sink_k = k[:, :, : self.num_sink_tokens, :]
            recent_k = k[:, :, -self.window_size :, :]
            pruned_k = torch.cat([sink_k, recent_k], dim=2)

            sink_v = v[:, :, : self.num_sink_tokens, :]
            recent_v = v[:, :, -self.window_size :, :]
            pruned_v = torch.cat([sink_v, recent_v], dim=2)

            new_caches.append((pruned_k, pruned_v))

        return new_caches
