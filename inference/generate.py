"""
Milestones 8 & 9: Generation, Sampling (Temperature, Top-K, Top-P) & KV Caching

Sampling controls:
1. Greedy (temperature = 0):
   next_token = logits[:, -1].argmax()

2. Temperature scaling:
   logits = logits / temperature
   (T < 1 -> sharp/deterministic; T > 1 -> flat/creative)

3. Top-k filtering:
   Keeps only the top k tokens with highest logits.

4. Top-p (Nucleus) filtering:
   Keeps smallest cumulative probability set exceeding p (e.g. 0.9).
"""

import argparse
import time
from typing import Optional, List
import torch
import torch.nn.functional as F

from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import ByteLevelBPETokenizer, CharTokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Sample the next token ID from output logits."""
    # 1. Greedy decoding
    if temperature <= 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature scaling
    logits = logits / temperature

    # 3. Top-k filtering
    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        v, _ = torch.topk(logits, top_k)
        min_top_v = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_v, torch.full_like(logits, float("-inf")), logits)

    # 4. Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """Autoregressive text generation with optional KV caching."""
    model.eval()
    tokens = input_ids.clone()

    if use_kv_cache:
        # Step 1: Prefill prompt and obtain initial cache
        logits, _, kv_caches = model(tokens, kv_cache=None, use_cache=True, start_pos=0)
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            return tokens

        # Step 2: Decode step-by-step passing only 1 token with cached past keys and values
        for _ in range(max_new_tokens - 1):
            current_pos = tokens.size(1) - 1
            if current_pos >= model.config.max_seq_len:
                break

            logits, _, kv_caches = model(
                next_token,
                kv_cache=kv_caches,
                use_cache=True,
                start_pos=current_pos,
            )
            next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
            tokens = torch.cat([tokens, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
    else:
        # Standard generation without KV cache (full sequence forward pass)
        for _ in range(max_new_tokens):
            idx_cond = tokens[:, -model.config.max_seq_len:]
            logits, _, _ = model(idx_cond, use_cache=False)
            next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
            tokens = torch.cat([tokens, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained NexLM checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cache", action="store_true", help="Disable KV caching")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running inference on {device}...")

    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer_path)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    t0 = time.time()
    out_ids = generate(
        model=model,
        input_ids=prompt_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_kv_cache=not args.no_cache,
    )
    elapsed = time.time() - t0

    generated_text = tokenizer.decode(out_ids[0].tolist())
    print("\n" + "=" * 50)
    print("PROMPT:   ", args.prompt)
    print("OUTPUT:   ", generated_text)
    print("=" * 50)
    print(f"Generated {out_ids.shape[1] - prompt_ids.shape[1]} tokens in {elapsed:.3f}s ({(out_ids.shape[1] - prompt_ids.shape[1]) / max(1e-4, elapsed):.1f} tok/s)")


if __name__ == "__main__":
    main()
