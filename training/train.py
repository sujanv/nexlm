"""
Milestones 7, 8 & 10: Training Loop, Toy Data Overfitting & Checkpointing

Key concepts:
1. Shifted Language Modeling Task:
   x = tokens[:, :-1]
   y = tokens[:, 1:]
   Loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

2. Toy Overfitting Check (Milestone 8):
   Before running on giant datasets, prove the model and training loop work
   by driving loss to near zero on a small corpus.

3. Checkpointing & Resuming (Milestone 10):
   Save model weights, optimizer state, step counter, and config so training
   can be resumed seamlessly.
"""

import argparse
import math
import os
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import ByteLevelBPETokenizer, CharTokenizer


class TextChunkDataset(Dataset):
    """Slices a 1D token tensor into input (x) and next-token target (y) pairs."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def get_lr_cosine(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def main():
    parser = argparse.ArgumentParser(description="Train NexLM GPT model")
    parser.add_argument("--toy", action="store_true", help="Run toy dataset overfit test")
    parser.add_argument("--data", type=str, default="data/tinystories.txt", help="Training corpus text file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt to resume from")
    parser.add_argument("--steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Prepare text data
    if args.toy:
        print("\n[Mode] Running Toy Dataset Overfit Test (Milestone 8)...")
        raw_text = "hello world. the cat sat on the mat. " * 80
        tokenizer = CharTokenizer(raw_text)
        seq_len = 32
        d_model = 64
        n_heads = 2
        n_layers = 2
    else:
        if not os.path.exists(args.data):
            print(f"Data file '{args.data}' not found. Generating sample story text...")
            os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
            raw_text = (
                "Once upon a time, Lily found a red ball in the sunny garden. "
                "Lily was very happy. Her dog Sam ran across the grass to play. "
                "They played together until the sun set behind the trees.\n"
            ) * 200
            with open(args.data, "w", encoding="utf-8") as f:
                f.write(raw_text)
        else:
            with open(args.data, "r", encoding="utf-8") as f:
                raw_text = f.read()

        tokenizer = ByteLevelBPETokenizer(vocab_size=1024)
        tokenizer.train(raw_text[:50_000], verbose=False)
        os.makedirs("data", exist_ok=True)
        tokenizer.save("data/tokenizer.json")
        seq_len = 128
        d_model = 128
        n_heads = 4
        n_layers = 4

    encoded_ids = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    print(f"Total tokens in dataset: {len(encoded_ids):,} (Vocab size: {tokenizer.vocab_size})")

    # Dataset & DataLoader
    split_idx = int(0.9 * len(encoded_ids))
    train_tokens = encoded_ids[:split_idx]
    val_tokens = encoded_ids[split_idx:]
    if len(val_tokens) <= seq_len:
        val_tokens = train_tokens

    train_ds = TextChunkDataset(train_tokens, seq_len=seq_len)
    val_ds = TextChunkDataset(val_tokens, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    loader_iter = iter(train_loader)

    # Initialize model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    start_step = 0
    best_val_loss = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"✓ Resumed training from {args.resume} at step {start_step}")

    print(f"Model parameters: {model.get_num_params():,}")
    print("Beginning training loop...")

    t0 = time.time()
    for step in range(start_step, args.steps):
        model.train()
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        # LR cosine schedule
        lr = get_lr_cosine(step, warmup_steps=20, max_steps=args.steps, max_lr=args.lr, min_lr=args.lr * 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            tok_per_sec = ((step + 1) * args.batch_size * seq_len) / max(1e-4, elapsed)
            print(f"Step {step + 1:4d}/{args.steps} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:.0f} tok/s")

        # Periodic Evaluation and Checkpointing
        if (step + 1) % 50 == 0 or (step + 1) == args.steps:
            model.eval()
            with torch.no_grad():
                val_losses = []
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    _, vloss, _ = model(vx, targets=vy)
                    val_losses.append(vloss.item())
                    if len(val_losses) >= 10:
                        break
                avg_val_loss = sum(val_losses) / max(1, len(val_losses))
                val_ppl = math.exp(min(avg_val_loss, 20.0))
                print(f"  >>> Eval @ Step {step + 1}: Val Loss = {avg_val_loss:.4f} | Perplexity = {val_ppl:.2f}")

                # Save checkpoint
                ckpt_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "config": config.__dict__,
                    "best_val_loss": best_val_loss,
                    "val_loss": avg_val_loss,
                }
                torch.save(ckpt_data, os.path.join(args.checkpoint_dir, "checkpoint.pt"))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(ckpt_data, os.path.join(args.checkpoint_dir, "best.pt"))
                    print(f"  ✓ Saved new best model checkpoint to {args.checkpoint_dir}/best.pt")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
