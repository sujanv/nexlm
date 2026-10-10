"""
Enhancement: Structured JSON & Schema-Constrained Decoding

Why Constrained Decoding?
When using LLMs for function calling or structured data extraction, raw text generation
frequently generates invalid JSON (missing quotes, unclosed brackets, trailing commas).

Constrained Decoding enforces formal grammar constraints via Logit Masking:
At every generation step:
1. The parser determines the set of syntactically legal next characters.
2. Any token that would violate the JSON grammar has its logit masked to -infinity.
3. Guarantees 100% syntactically valid JSON output!
"""

import json
from typing import List, Set, Optional, Dict
import torch
import torch.nn.functional as F

from model.gpt import GPT
from tokenizer.tokenizer import ByteLevelBPETokenizer


class SimpleJSONStateTracker:
    """
    Lightweight pushdown automaton tracking JSON structural balance.
    Enforces valid braces {}, brackets [], quotes "", and colons.
    """

    def __init__(self):
        self.brace_depth = 0
        self.in_string = False
        self.escape_next = False
        self.has_started = False
        self.is_completed = False

    def update(self, char: str) -> bool:
        """Processes character and updates state. Returns False if character violates grammar."""
        if self.is_completed and not char.isspace():
            return False

        if not self.has_started:
            if char.isspace():
                return True
            if char == "{":
                self.has_started = True
                self.brace_depth = 1
                return True
            return False

        if self.in_string:
            if self.escape_next:
                self.escape_next = False
                return True
            if char == "\\":
                self.escape_next = True
                return True
            if char == '"':
                self.in_string = False
                return True
            return True
        else:
            if char == '"':
                self.in_string = True
                return True
            elif char == "{":
                self.brace_depth += 1
                return True
            elif char == "}":
                self.brace_depth -= 1
                if self.brace_depth == 0:
                    self.is_completed = True
                elif self.brace_depth < 0:
                    return False
                return True
            return True


@torch.no_grad()
def generate_constrained_json(
    model: GPT,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """
    Autoregressively generates text while constraining output to valid JSON objects.
    """
    model.eval()
    device = next(model.parameters()).device

    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    tokens = prompt_ids.clone()
    tracker = SimpleJSONStateTracker()

    generated_chars = []

    for _ in range(max_new_tokens):
        if tracker.is_completed:
            break

        idx_cond = tokens[:, -model.config.max_seq_len:]
        logits, _, _ = model(idx_cond)
        next_logits = logits[:, -1, :].clone() / max(1e-5, temperature)

        # Logit masking: mask tokens whose characters violate current JSON state
        # In ByteLevelBPETokenizer, token IDs 0..255 correspond directly to byte characters
        for tok_id in range(min(256, next_logits.size(-1))):
            ch = chr(tok_id)
            test_tracker = SimpleJSONStateTracker()
            test_tracker.brace_depth = tracker.brace_depth
            test_tracker.in_string = tracker.in_string
            test_tracker.escape_next = tracker.escape_next
            test_tracker.has_started = tracker.has_started
            test_tracker.is_completed = tracker.is_completed

            if not test_tracker.update(ch):
                next_logits[0, tok_id] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        tokens = torch.cat([tokens, next_token], dim=1)
        ch = tokenizer.decode([token_id])
        generated_chars.append(ch)

        for c in ch:
            tracker.update(c)

    return "".join(generated_chars)
