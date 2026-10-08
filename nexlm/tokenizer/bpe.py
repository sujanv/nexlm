import json
from collections import Counter
from typing import List, Dict, Tuple, Optional

from nexlm.tokenizer.base import BaseTokenizer


class ByteLevelBPETokenizer(BaseTokenizer):
    """
    Byte-level Byte Pair Encoding (BPE) Tokenizer implemented from scratch.
    
    Properties:
    - Base vocabulary is all 256 individual byte values (0-255).
    - Guarantees zero out-of-vocabulary (OOV / <unk>) tokens for any valid or invalid UTF-8 string.
    - Special tokens: <|endoftext|>, <|pad|>.
    - Learnable merge rules extracted from any raw text training corpus.
    """

    SPECIAL_TOKENS = {
        "<|endoftext|>": 256,
        "<|pad|>": 257,
    }

    def __init__(self, vocab_size: int = 4096):
        self.target_vocab_size = vocab_size
        self._special_tokens = dict(self.SPECIAL_TOKENS)
        self._reverse_special = {v: k for k, v in self._special_tokens.items()}

        # Merge rules: (token_a, token_b) -> new_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Inverse lookup: new_token_id -> (token_a, token_b)
        self.vocab: Dict[int, bytes] = {}

        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        # Base byte vocab (0..255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Merged tokens expand to their underlying bytes
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

    def train(self, text: str, verbose: bool = True) -> None:
        """
        Train BPE merge rules on raw text up to self.target_vocab_size.
        """
        # Convert input text to list of raw bytes (integers 0..255)
        raw_bytes = list(text.encode("utf-8"))
        if not raw_bytes:
            return

        current_tokens = list(raw_bytes)
        num_merges = self.target_vocab_size - 256 - len(self._special_tokens)
        next_token_id = 256 + len(self._special_tokens)

        self.merges = {}
        self._init_base_vocab()

        for step in range(num_merges):
            # Count adjacent pairs
            pair_counts: Counter[Tuple[int, int]] = Counter()
            for i in range(len(current_tokens) - 1):
                pair = (current_tokens[i], current_tokens[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                # No more repeated pairs to merge
                break

            # Register merge
            self.merges[best_pair] = next_token_id
            self.vocab[next_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to current token stream
            new_tokens: List[int] = []
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

            if verbose and (step + 1) % 500 == 0:
                print(f"BPE Training: Step {step + 1}/{num_merges} | Vocab size: {self.vocab_size}")

        if verbose:
            print(f"BPE Training Complete. Final vocab size: {self.vocab_size} (Learned {len(self.merges)} merges)")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode string into list of token IDs."""
        tokens = list(text.encode("utf-8"))
        if not tokens:
            return [self.eos_token_id] if add_special_tokens else []

        # Iteratively merge pairs according to learned merge rules
        while len(tokens) >= 2:
            # Find candidate pairs and their merge IDs
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            # Find the pair with the earliest/lowest merge id (highest priority)
            mergeable = [(self.merges[p], p) for p in pairs if p in self.merges]
            if not mergeable:
                break

            target_id, target_pair = min(mergeable, key=lambda x: x[0])

            # Apply the selected merge across the tokens
            new_tokens: List[int] = []
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
        """Decode list of token IDs back into string."""
        byte_chunks: List[bytes] = []

        for token in tokens:
            if token in self._reverse_special:
                if not skip_special_tokens:
                    byte_chunks.append(self._reverse_special[token].encode("utf-8"))
            elif token in self.vocab:
                byte_chunks.append(self.vocab[token])
            elif token < 256:
                byte_chunks.append(bytes([token]))
            else:
                # Fallback for unrecognized token ID
                byte_chunks.append(b"")

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Save vocabulary and merge rules to JSON."""
        data = {
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self._special_tokens,
            "merges": [
                {"pair": list(pair), "token_id": token_id}
                for pair, token_id in self.merges.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPETokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data.get("target_vocab_size", 4096))
        tokenizer._special_tokens = data.get("special_tokens", cls.SPECIAL_TOKENS)
        tokenizer._reverse_special = {v: k for k, v in tokenizer._special_tokens.items()}

        tokenizer.merges = {}
        for item in data.get("merges", []):
            pair = tuple(item["pair"])
            tokenizer.merges[pair] = item["token_id"]

        tokenizer._init_base_vocab()
        return tokenizer


class CharTokenizer(BaseTokenizer):
    """
    Simple Character-level Tokenizer for instant training & debugging.
    """

    def __init__(self, vocab_str: Optional[str] = None):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._eos = "<|endoftext|>"
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
        chars = sorted(list(set(text)))
        specials = [self._eos, self._pad]
        all_chars = specials + chars
        self.stoi = {ch: i for i, ch in enumerate(all_chars)}
        self.itos = {i: ch for i, ch in enumerate(all_chars)}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [self.stoi.get(c, self.pad_token_id) for c in text]
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        res = []
        for t in tokens:
            ch = self.itos.get(t, "")
            if skip_special_tokens and ch in (self._eos, self._pad):
                continue
            res.append(ch)
        return "".join(res)

    def save(self, path: str) -> None:
        data = {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        return tok
