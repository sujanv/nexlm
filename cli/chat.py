"""
Enhancement: Interactive Terminal Chat Interface (CLI REPL)

Features:
1. Multi-turn conversation history buffer.
2. Real-time ANSI colorized streaming output.
3. Interactive runtime commands: /clear, /temp, /tokens, /exit.
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Optional
import torch

from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import ByteLevelBPETokenizer
from inference.generate import generate_stream


# ANSI Color codes for polished terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ConversationSession:
    """Manages multi-turn conversation state and token limits."""

    def __init__(self, system_prompt: str = "You are NexLM, a helpful and imaginative assistant."):
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})

    def clear(self) -> None:
        self.history.clear()

    def format_prompt(self) -> str:
        """Formats the conversation history into a unified autoregressive prompt string."""
        formatted = [f"System: {self.system_prompt}"]
        for turn in self.history:
            role_name = "User" if turn["role"] == "user" else "Assistant"
            formatted.append(f"{role_name}: {turn['content']}")
        formatted.append("Assistant: ")
        return "\n".join(formatted)


def run_chat_repl(
    model: GPT,
    tokenizer: ByteLevelBPETokenizer,
    device: torch.device,
    default_temp: float = 0.7,
    default_top_k: int = 40,
    default_top_p: int = 0.9,
) -> None:
    session = ConversationSession()
    temperature = default_temp

    print(f"\n{BOLD}{CYAN}==================================================={RESET}")
    print(f"{BOLD}{CYAN}        ⚡ NexLM Interactive Terminal Chat{RESET}")
    print(f"{BOLD}{CYAN}==================================================={RESET}")
    print(f"{DIM}Commands: /clear (reset), /temp <val>, /tokens, /exit{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}{CYAN}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        # In-chat slash commands
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            if cmd in ("/exit", "/quit"):
                print(f"{DIM}Goodbye!{RESET}")
                break
            elif cmd == "/clear":
                session.clear()
                print(f"{YELLOW}✓ Conversation history cleared.{RESET}\n")
                continue
            elif cmd == "/temp":
                if len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        print(f"{YELLOW}✓ Temperature set to {temperature:.2f}{RESET}\n")
                    except ValueError:
                        print(f"{YELLOW}Invalid temperature value.{RESET}\n")
                else:
                    print(f"{YELLOW}Current temperature: {temperature:.2f}{RESET}\n")
                continue
            elif cmd == "/tokens":
                prompt_text = session.format_prompt()
                toks = tokenizer.encode(prompt_text)
                print(f"{YELLOW}Current context tokens: {len(toks)} / {model.config.max_seq_len}{RESET}\n")
                continue
            else:
                print(f"{YELLOW}Unknown command: {cmd}{RESET}\n")
                continue

        session.add_user_message(user_input)
        prompt_text = session.format_prompt()
        prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

        # Truncate left if exceeding context window
        if prompt_ids.shape[1] > model.config.max_seq_len - 60:
            prompt_ids = prompt_ids[:, -(model.config.max_seq_len - 60):]

        print(f"\n{BOLD}{GREEN}NexLM > {RESET}", end="", flush=True)

        generated_token_ids = []
        for token_id in generate_stream(
            model=model,
            input_ids=prompt_ids,
            max_new_tokens=80,
            temperature=temperature,
            top_k=default_top_k,
            top_p=default_top_p,
            use_kv_cache=True,
        ):
            generated_token_ids.append(token_id)
            tok_str = tokenizer.decode([token_id])
            # If newline encountered after generation started or end of message, stop
            if "\nUser:" in tok_str or "<|endoftext|>" in tok_str:
                break
            print(tok_str, end="", flush=True)

        print("\n")
        full_reply = tokenizer.decode(generated_token_ids)
        session.add_assistant_message(full_reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NexLM Terminal Chat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    if os.path.exists(args.tokenizer_path):
        tokenizer = ByteLevelBPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = ByteLevelBPETokenizer(vocab_size=256)
        tokenizer.train("Once upon a time", verbose=False)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        config = GPTConfig(vocab_size=tokenizer.vocab_size, max_seq_len=256, d_model=128, n_heads=4, n_layers=4)
        model = GPT(config).to(device)

    model.eval()
    run_chat_repl(model, tokenizer, device, default_temp=args.temperature)
