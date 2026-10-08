import argparse
import math
import torch
from torch.utils.data import DataLoader

from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import ByteLevelBPETokenizer
from inference.generate import generate
from training.train import TextChunkDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate NexLM model on validation data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data", type=str, default="data/tinystories.txt")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time, Lily")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint {args.checkpoint} on {device}...")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer_path)

    # Optional dataset evaluation
    import os
    if os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        ds = TextChunkDataset(ids, seq_len=config.max_seq_len)
        loader = DataLoader(ds, batch_size=16)

        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                _, loss, _ = model(x, targets=y)
                total_loss += loss.item()
                count += 1
                if count >= 20:
                    break
        avg_loss = total_loss / max(1, count)
        ppl = math.exp(min(avg_loss, 20.0))
        print(f"Validation Loss: {avg_loss:.4f} | Validation Perplexity: {ppl:.2f}")

    # Generate sample
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = generate(model, prompt_ids, max_new_tokens=60, temperature=0.8, top_k=40, top_p=0.9)
    print("\nSample Generation:")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
