import argparse
import os
import sys
import time
import torch

from nexlm.config import NexLMConfig
from nexlm.model.gpt import NexLM
from nexlm.tokenizer.bpe import ByteLevelBPETokenizer
from nexlm.inference.generate import generate_stream, generate
from nexlm.training.dataset import prepare_dataset_from_text
from nexlm.training.trainer import NexLMTrainer, TrainingConfig
from benchmarks.bench_kv_cache import benchmark_kv_cache


def run_quick_train_and_demo():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("=" * 60)
    print("           NexLM: End-to-End Quickstart Demo")
    print("=" * 60)
    print(f"Device: {device}")

    # 1. Prepare sample dataset
    stories = []
    names = ["Spot", "Mia", "Leo", "Lily", "Tim", "Sam"]
    friends = ["a friendly bird", "a gentle rabbit", "a playful dog", "a cheerful cat"]
    objects = ["a shiny red ball", "a bright blue kite", "a sweet red apple", "a golden leaf"]
    for i in range(120):
        n = names[i % len(names)]
        f = friends[i % len(friends)]
        o = objects[i % len(objects)]
        stories.append(
            f"Once upon a time, {n} went to the green park. "
            f"On the grass, {n} discovered {o}. "
            f"Suddenly, {f} arrived and wanted to play together. "
            f"'{n}, look how wonderful this day is!' said the happy friend. "
            f"They spent the whole afternoon playing with {o} and sharing stories under the sun. "
            f"When evening arrived, {n} went home feeling warm and joyful. "
            f"Everyone fell asleep dreaming of fun adventures.\n"
        )
    sample_text = "".join(stories)

    print("\n[Step 1/4] Training Byte-level BPE Tokenizer from scratch...")
    tokenizer = ByteLevelBPETokenizer(vocab_size=512)
    tokenizer.train(sample_text, verbose=False)
    print(f"✓ Tokenizer ready. Vocab size: {tokenizer.vocab_size}")

    # 2. Build model
    print("\n[Step 2/4] Initializing NexLM Transformer architecture...")
    config = NexLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        dim=128,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
    )
    model = NexLM(config).to(device)
    print(f"✓ Model created with {model.get_num_params():,} parameters.")

    # 3. Train for quick demo (50 steps)
    print("\n[Step 3/4] Training model for 50 quick steps...")
    train_ds, val_ds = prepare_dataset_from_text(sample_text, tokenizer, seq_len=config.max_seq_len)
    trainer = NexLMTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=TrainingConfig(
            max_steps=60,
            batch_size=8,
            learning_rate=1e-3,
            eval_interval=20,
            checkpoint_dir="checkpoints/demo",
            device=str(device),
        ),
    )
    trainer.train()

    # 4. Generate story
    print("\n[Step 4/4] Generating story completions...")
    prompt = "Once upon a time, Spot found"
    print(f"Prompt: '{prompt}'\nGenerated text (streaming with KV cache):")
    print(f"\033[92m{prompt}\033[0m", end="", flush=True)

    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for token_id in generate_stream(
        model,
        prompt_ids,
        max_new_tokens=60,
        temperature=0.7,
        top_k=30,
        top_p=0.9,
        use_kv_cache=True,
    ):
        word = tokenizer.decode([token_id])
        print(word, end="", flush=True)
        time.sleep(0.02)
    print("\n\n✓ Demo completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="NexLM Demo Runner")
    parser.add_argument("--bench", action="store_true", help="Run KV Cache benchmark")
    parser.add_argument("--train", action="store_true", help="Run quick train and generate demo")
    args = parser.parse_args()

    if args.bench:
        benchmark_kv_cache()
    else:
        run_quick_train_and_demo()


if __name__ == "__main__":
    main()
