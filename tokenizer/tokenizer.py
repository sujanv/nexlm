"""
Milestone 1: Tokenizer Pipeline

Why start with the tokenizer?
An LLM never sees raw English or raw characters directly. It only operates on discrete integer IDs:
    raw text  ──(encode)──>  token IDs  ──(model)──>  predicted IDs  ──(decode)──>  text

This file provides two implementations:
1. CharTokenizer: The simplest possible tokenizer. Excellent for intuition, debugging,
   and seeing every step clearly without subword merge complexity.
2. ByteLevelBPETokenizer: A production-style Byte-Pair Encoding tokenizer implemented from
   scratch. It treats any UTF-8 byte as base vocabulary so it can encode any text with zero <unk> tokens.
"""

import json
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Dict, Tuple, Optional


class BaseTokenizer(ABC):
    """Abstract interface all NexLM tokenizers must satisfy."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        pass


# =====================================================================
# 1. Character-Level Tokenizer (Intuitive & Debug-Friendly)
# =====================================================================

class CharTokenizer(BaseTokenizer):
    """
    Character-level Tokenizer.
    
    Example:
        text = "hello"
        tokens = ['h', 'e', 'l', 'l', 'o']
        ids    = [7,   4,   11,  11,  14]
    """

    def __init__(self, vocab_str: Optional[str] = None):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._eos = "<|eos|>"
        self._pad = "<|pad|>"

        if vocab_str is not None:
            self.train(vocab_str)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def eos_token_id(self) -> int:
        return self.stoi[self._eos]

    @property
    def pad_token_id(self) -> int:
        return self.stoi[self._pad]

    def train(self, text: str) -> None:
        """Build character vocabulary from raw training text."""
        chars = sorted(list(set(text)))
        specials = [self._pad, self._eos]
        all_chars = specials + chars
        self.stoi = {ch: i for i, ch in enumerate(all_chars)}
        self.itos = {i: ch for i, ch in enumerate(all_chars)}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert string to list of integer token IDs."""
        tokens = [self.stoi.get(c, self.pad_token_id) for c in text]
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """Convert list of token IDs back into string."""
        chars = []
        for t in tokens:
            ch = self.itos.get(t, "")
            if skip_special_tokens and ch in (self._pad, self._eos):
                continue
            chars.append(ch)
        return "".join(chars)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        return tok


# =====================================================================
# 2. Byte-Level BPE Tokenizer (Production Grade, Zero <unk> Tokens)
# =====================================================================

class ByteLevelBPETokenizer(BaseTokenizer):
    """
    Byte-level Byte Pair Encoding (BPE) Tokenizer.
    
    Starts with 256 byte tokens (0..255) and iteratively merges the most
    frequent adjacent token pairs in the corpus.
    """

    SPECIAL_TOKENS = {
        "<|endoftext|>": 256,
        "<|pad|>": 257,
    }

    def __init__(self, vocab_size: int = 4096):
        self.target_vocab_size = vocab_size
        self._special_tokens = dict(self.SPECIAL_TOKENS)
        self._reverse_special = {v: k for k, v in self._special_tokens.items()}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self._special_tokens) + len(self.merges)

    @property
    def eos_token_id(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    @property
    def pad_token_id(self) -> int:
        return self._special_tokens["<|pad|>"]

    def train(self, text: str, verbose: bool = False) -> None:
        """Extract frequent merges from text until target_vocab_size is reached."""
        raw_bytes = list(text.encode("utf-8"))
        if not raw_bytes:
            return

        current_tokens = list(raw_bytes)
        num_merges = self.target_vocab_size - 256 - len(self._special_tokens)
        next_token_id = 256 + len(self._special_tokens)

        self.merges = {}
        self._init_base_vocab()

        for step in range(num_merges):
            pair_counts: Counter[Tuple[int, int]] = Counter()
            for i in range(len(current_tokens) - 1):
                pair = (current_tokens[i], current_tokens[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            self.merges[best_pair] = next_token_id
            self.vocab[next_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            new_tokens = []
            i = 0
            while i < len(current_tokens):
                if (
                    i < len(current_tokens) - 1
                    and (current_tokens[i], current_tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(next_token_id)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1

            current_tokens = new_tokens
            next_token_id += 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = list(text.encode("utf-8"))
        if not tokens:
            return [self.eos_token_id] if add_special_tokens else []

        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            mergeable = [(self.merges[p], p) for p in pairs if p in self.merges]
            if not mergeable:
                break

            target_id, target_pair = min(mergeable, key=lambda x: x[0])
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == target_pair:
                    new_tokens.append(target_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        byte_chunks = []
        for token in tokens:
            if token in self._reverse_special:
                if not skip_special_tokens:
                    byte_chunks.append(self._reverse_special[token].encode("utf-8"))
            elif token in self.vocab:
                byte_chunks.append(self.vocab[token])
            elif token < 256:
                byte_chunks.append(bytes([token]))
            else:
                byte_chunks.append(b"")
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        data = {
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self._special_tokens,
            "merges": [{"pair": list(k), "token_id": v} for k, v in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data.get("target_vocab_size", 4096))
        tok._special_tokens = data.get("special_tokens", cls.SPECIAL_TOKENS)
        tok._reverse_special = {v: k for k, v in tok._special_tokens.items()}
        tok.merges = {tuple(m["pair"]): m["token_id"] for m in data.get("merges", [])}
        tok._init_base_vocab()
        return tok
