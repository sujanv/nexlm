from typing import Optional, List, Generator
import torch
import torch.nn as nn

from nexlm.inference.sampler import sample_next_token
from nexlm.inference.kv_cache import KVCache


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """
    Autoregressive text generation supporting KV caching and various sampling strategies.

    Args:
        model: NexLM model instance
        input_ids: Tensor of initial prompt token IDs, shape (B, T)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (1.0 = normal, 0.0 = greedy)
        top_k: Top-k filtering threshold
        top_p: Nucleus filtering threshold
        eos_token_id: Token ID indicating end of sequence (stops early)
        use_kv_cache: Whether to use KV caching for fast generation

    Returns:
        Tensor of shape (B, T + generated_tokens)
    """
    model.eval()
    device = input_ids.device
    tokens = input_ids.clone()
    B = input_ids.shape[0]

    if use_kv_cache:
        # 1. Prefill phase: process prompt tokens and populate initial KV cache
        # If prompt is longer than max_seq_len, truncate from left
        if tokens.shape[1] > model.config.max_seq_len:
            tokens = tokens[:, -model.config.max_seq_len:]

        logits, _, layer_caches = model(
            tokens,
            kv_cache=None,
            use_cache=True,
            start_pos=0,
        )

        cache = KVCache(model.config.n_layers)
        cache.update_all(layer_caches)

        # Sample first generated token from the last prompt position
        next_token_logits = logits[:, -1, :]
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            return tokens

        # 2. Decode phase: pass single token each step with accumulated KV cache
        for _ in range(max_new_tokens - 1):
            current_pos = cache.current_seq_len
            if current_pos >= model.config.max_seq_len:
                break

            logits, _, layer_caches = model(
                next_token,
                kv_cache=cache.cache,
                use_cache=True,
                start_pos=current_pos,
            )
            cache.update_all(layer_caches)

            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens = torch.cat([tokens, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return tokens

    else:
        # Standard generation without KV cache (full sequence forward pass each step)
        for _ in range(max_new_tokens):
            idx_cond = (
                tokens
                if tokens.size(1) <= model.config.max_seq_len
                else tokens[:, -model.config.max_seq_len:]
            )
            logits, _, _ = model(idx_cond, kv_cache=None, use_cache=False)
            next_token_logits = logits[:, -1, :]

            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens = torch.cat([tokens, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return tokens


@torch.no_grad()
def generate_stream(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,
    use_kv_cache: bool = True,
) -> Generator[int, None, None]:
    """
    Stream generated tokens one at a time.
    """
    model.eval()
    B = input_ids.shape[0]
    assert B == 1, "Streaming currently supports batch size 1"

    if use_kv_cache:
        tokens = input_ids.clone()
        if tokens.shape[1] > model.config.max_seq_len:
            tokens = tokens[:, -model.config.max_seq_len:]

        logits, _, layer_caches = model(tokens, kv_cache=None, use_cache=True, start_pos=0)
        cache = KVCache(model.config.n_layers)
        cache.update_all(layer_caches)

        next_token_logits = logits[:, -1, :]
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        token_id = next_token.item()
        yield token_id

        if eos_token_id is not None and token_id == eos_token_id:
            return

        for _ in range(max_new_tokens - 1):
            current_pos = cache.current_seq_len
            if current_pos >= model.config.max_seq_len:
                break

            logits, _, layer_caches = model(
                next_token,
                kv_cache=cache.cache,
                use_cache=True,
                start_pos=current_pos,
            )
            cache.update_all(layer_caches)

            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            token_id = next_token.item()
            yield token_id

            if eos_token_id is not None and token_id == eos_token_id:
                break
    else:
        tokens = input_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = (
                tokens
                if tokens.size(1) <= model.config.max_seq_len
                else tokens[:, -model.config.max_seq_len:]
            )
            logits, _, _ = model(idx_cond, kv_cache=None, use_cache=False)
            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            token_id = next_token.item()
            tokens = torch.cat([tokens, next_token], dim=1)
            yield token_id

            if eos_token_id is not None and token_id == eos_token_id:
                break
