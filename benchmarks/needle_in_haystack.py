"""
Enhancement: Needle In A Haystack (NIAH) Context Retrieval Benchmark

Why Needle In A Haystack?
Popularized by Greg Kamradt, NIAH tests an LLM's effective context retrieval capability.
A specific secret piece of information ("needle") is inserted into a body of unrelated
background text ("haystack") at varying positions (depths from 0% to 100%).
The benchmark evaluates whether the attention heads can retrieve the needle accurately.
"""

import argparse
import random
from typing import Dict, List, Tuple
import torch

from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import ByteLevelBPETokenizer
from inference.generate import generate


HAYSTACK_FILLER = [
    "The green trees in the park swayed gently under the afternoon breeze. ",
    "A small river flowed peacefully along the edge of the quiet town. ",
    "Children were laughing and playing games near the community center. ",
    "The sun was shining brightly in the clear blue summer sky. ",
    "Birds were chirping softly on the branches of the ancient oak tree. ",
    "A friendly dog barked happily as it chased a tennis ball across the lawn. ",
]


def construct_haystack(target_tokens: int, tokenizer: ByteLevelBPETokenizer) -> str:
    """Builds a haystack string of approximately target_tokens length."""
    words = []
    current_tokens = 0
    while current_tokens < target_tokens:
        sentence = random.choice(HAYSTACK_FILLER)
        words.append(sentence)
        current_tokens += len(tokenizer.encode(sentence))
    return "".join(words)


def insert_needle(haystack: str, needle: str, depth_fraction: float) -> str:
    """Inserts the needle at the specified fractional depth (0.0 to 1.0)."""
    sentences = haystack.split(". ")
    insert_idx = max(0, min(len(sentences), int(len(sentences) * depth_fraction)))
    sentences.insert(insert_idx, needle)
    return ". ".join(sentences)


def run_needle_test(
    model: GPT,
    tokenizer: ByteLevelBPETokenizer,
    context_length: int = 128,
    depths: List[float] = [0.1, 0.5, 0.9],
) -> Dict[float, bool]:
    """
    Evaluates needle retrieval across given context depths.
    """
    secret_code = "blue-42"
    needle = f"The secret password for the gate is {secret_code}."
    query_prompt = "\nQuestion: What is the secret password for the gate?\nAnswer: The secret password is "

    results = {}
    device = next(model.parameters()).device

    print("\n" + "=" * 65)
    print(f"Needle-In-A-Haystack Evaluation | Context Length: ~{context_length} tokens")
    print("=" * 65)
    print(f"{'Depth':>10} | {'Found Needle?':>15} | {'Generated Output'}")
    print("-" * 65)

    for depth in depths:
        base_haystack = construct_haystack(context_length, tokenizer)
        full_text = insert_needle(base_haystack, needle, depth) + query_prompt

        tokens = torch.tensor([tokenizer.encode(full_text)], dtype=torch.long, device=device)
        # Truncate if exceeding max_seq_len
        if tokens.shape[1] > model.config.max_seq_len:
            tokens = tokens[:, -model.config.max_seq_len:]

        out = generate(model, tokens, max_new_tokens=15, temperature=0.1, use_kv_cache=True)
        gen_tokens = out[0, tokens.shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_tokens).strip()

        success = secret_code in gen_text
        results[depth] = success
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{depth * 100:9.0f}% | {status:>15} | \"{gen_text[:40]}\"")

    print("=" * 65)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_len", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = ByteLevelBPETokenizer(vocab_size=512)
    sample_text = "".join(HAYSTACK_FILLER * 10)
    tokenizer.train(sample_text, verbose=False)

    config = GPTConfig(vocab_size=tokenizer.vocab_size, max_seq_len=256, d_model=128, n_heads=4, n_layers=4)
    model = GPT(config).to(device)
    model.eval()

    run_needle_test(model, tokenizer, context_length=args.context_len)
