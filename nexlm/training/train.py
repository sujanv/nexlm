import argparse
import os
import sys

import torch

from nexlm.config import NexLMConfig
from nexlm.model.gpt import NexLM
from nexlm.tokenizer.bpe import ByteLevelBPETokenizer
from nexlm.training.dataset import prepare_dataset_from_text
from nexlm.training.trainer import NexLMTrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train a NexLM transformer model")
    parser.add_argument("--data", type=str, default="data/sample.txt", help="Path to text data file")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--config_file", type=str, default=None, help="Path to custom config JSON")
    parser.add_argument("--vocab_size", type=int, default=1024, help="Tokenizer vocabulary size")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json")
    parser.add_argument("--steps", type=int, default=300, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per micro-step")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify or generate sample data if file does not exist
    if not os.path.exists(args.data):
        print(f"Data file '{args.data}' not found. Generating sample training data...")
        os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
        sample_text = (
            "Once upon a time, there was a little girl named Lily. She loved exploring the woods.\n"
            "One sunny morning, Lily saw a colorful butterfly flutter by. 'Hello butterfly!' said Lily.\n"
            "The butterfly landed on a bright yellow flower. Lily watched quietly and smiled.\n"
            "Then, a friendly puppy named Sam ran into the garden. Sam wagged his tail happily.\n"
            "Lily and Sam played hide and seek until the sun began to set behind the hills.\n"
            "Her mother called her for warm dinner. Lily hugged Sam and walked inside happily.\n"
        ) * 500  # repeat to create a small corpus
        with open(args.data, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"Sample data created at {args.data} ({len(sample_text)} characters).")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    # Train or load Tokenizer
    if os.path.exists(args.tokenizer_path):
        print(f"Loading existing tokenizer from {args.tokenizer_path}...")
        tokenizer = ByteLevelBPETokenizer.load(args.tokenizer_path)
    else:
        print(f"Training Byte-level BPE tokenizer (target vocab size: {args.vocab_size})...")
        tokenizer = ByteLevelBPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(text[:100_000])  # train on prefix for speed
        os.makedirs(os.path.dirname(args.tokenizer_path) or ".", exist_ok=True)
        tokenizer.save(args.tokenizer_path)
        print(f"Tokenizer saved to {args.tokenizer_path} (vocab size: {tokenizer.vocab_size})")

    # Load Model Configuration
    if args.config_file is not None and os.path.exists(args.config_file):
        config = NexLMConfig.from_json(args.config_file)
    elif args.preset == "small":
        config = NexLMConfig.small()
    elif args.preset == "medium":
        config = NexLMConfig.medium()
    else:
        config = NexLMConfig.tiny()

    # Adjust config vocab_size to match tokenizer
    config.vocab_size = tokenizer.vocab_size
    print(f"Model Configuration:\n  vocab_size: {config.vocab_size}\n  seq_len: {config.max_seq_len}\n  dim: {config.dim}\n  layers: {config.n_layers}\n  heads: {config.n_heads}")

    # Build Model
    model = NexLM(config)
    print(f"Total model parameters: {model.get_num_params(non_embedding=False):,}")

    # Prepare Datasets
    print("Preparing tokenized datasets...")
    train_ds, val_ds = prepare_dataset_from_text(
        text=text,
        tokenizer=tokenizer,
        seq_len=config.max_seq_len,
        val_split=0.1,
    )
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")

    # Configure Trainer
    train_cfg = TrainingConfig(
        learning_rate=args.lr,
        max_steps=args.steps,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        eval_interval=max(10, args.steps // 5),
        checkpoint_interval=max(20, args.steps // 2),
    )

    trainer = NexLMTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=train_cfg,
    )

    trainer.train()


if __name__ == "__main__":
    main()
