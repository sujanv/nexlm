import time
import torch

from nexlm.config import NexLMConfig
from nexlm.model.gpt import NexLM
from nexlm.inference.generate import generate


def benchmark_kv_cache():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Running KV Cache Benchmark on device: {device}")

    # Use a realistic small configuration for the benchmark
    config = NexLMConfig(
        vocab_size=2048,
        max_seq_len=512,
        dim=256,
        n_layers=6,
        n_heads=8,
    )
    model = NexLM(config).to(device)
    model.eval()

    test_lengths = [32, 64, 128, 200]
    prompt_len = 16
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    print("\n" + "=" * 75)
    print(f"{'New Tokens':>12} | {'No KV Cache (s)':>16} | {'With KV Cache (s)':>18} | {'Speedup':>10}")
    print("=" * 75)

    # Warmup
    _ = generate(model, prompt, max_new_tokens=10, use_kv_cache=False)
    _ = generate(model, prompt, max_new_tokens=10, use_kv_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    for gen_len in test_lengths:
        # Measure without KV cache
        t0 = time.perf_counter()
        _ = generate(model, prompt, max_new_tokens=gen_len, use_kv_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        time_no_cache = time.perf_counter() - t0

        # Measure with KV cache
        t0 = time.perf_counter()
        _ = generate(model, prompt, max_new_tokens=gen_len, use_kv_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        time_with_cache = time.perf_counter() - t0

        speedup = time_no_cache / max(1e-6, time_with_cache)
        print(
            f"{gen_len:12d} | "
            f"{time_no_cache:16.4f} | "
            f"{time_with_cache:18.4f} | "
            f"{speedup:9.2f}x"
        )

    print("=" * 75)
    print("Benchmark complete! As generation sequence grows, KV cache shows significant speedup.")


if __name__ == "__main__":
    benchmark_kv_cache()
