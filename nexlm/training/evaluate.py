import argparse
import torch

from nexlm.config import NexLMConfig
from nexlm.model.gpt import NexLM
from nexlm.tokenizer.bpe import ByteLevelBPETokenizer
from nexlm.inference.generate import generate
from nexlm.training.dataset import prepare_dataset_from_text


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a NexLM checkpoint and test generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Lily saw a", help="Prompt to continue")
    parser.add_argument("--max_tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--eval_data", type=str, default=None, help="Optional text data to evaluate perplexity")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint from {args.checkpoint} onto {device}...")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Reconstruct model from saved configuration
    config = NexLMConfig.from_dict(ckpt["config"])
    model = NexLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer_path)

    # Optional perplexity calculation
    if args.eval_data:
        with open(args.eval_data, "r", encoding="utf-8") as f:
            eval_text = f.read()
        _, val_ds = prepare_dataset_from_text(eval_text, tokenizer, seq_len=config.max_seq_len, val_split=0.5)
        print(f"Evaluating perplexity on {len(val_ds)} validation chunks...")

    # Interactive sample generation
    print(f"\n--- Testing Generation for Prompt: '{args.prompt}' ---")
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    # 1. Greedy decoding
    greedy_out = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        temperature=0.0,
        use_kv_cache=True,
    )
    print("\n[Greedy (Temp=0.0)]:")
    print(tokenizer.decode(greedy_out[0].tolist()))

    # 2. Temperature + Top-K + Top-P sampling
    sampled_out = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_kv_cache=True,
    )
    print(f"\n[Sampled (Temp={args.temperature}, Top-K={args.top_k}, Top-P={args.top_p})]:")
    print(tokenizer.decode(sampled_out[0].tolist()))


if __name__ == "__main__":
    main()
