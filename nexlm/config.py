import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class NexLMConfig:
    """Configuration class for NexLM GPT-style transformer."""

    vocab_size: int = 4096
    max_seq_len: int = 256
    dim: int = 192
    n_layers: int = 6
    n_heads: int = 6
    dropout: float = 0.1
    bias: bool = True
    norm_eps: float = 1e-5
    norm_type: str = "layernorm"  # "layernorm" or "rmsnorm"
    tie_word_embeddings: bool = True

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, (
            f"Hidden dimension {self.dim} must be divisible by n_heads {self.n_heads}"
        )
        assert self.norm_type in ("layernorm", "rmsnorm"), (
            f"Unsupported norm_type: {self.norm_type}"
        )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NexLMConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "NexLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def tiny(cls) -> "NexLMConfig":
        """Tiny preset (~4-5M parameters) for quick iteration & testing."""
        return cls(
            vocab_size=2048,
            max_seq_len=256,
            dim=128,
            n_layers=4,
            n_heads=4,
            dropout=0.1,
            bias=True,
            norm_type="layernorm",
        )

    @classmethod
    def small(cls) -> "NexLMConfig":
        """Small preset (~15M parameters) ideal for TinyStories training."""
        return cls(
            vocab_size=4096,
            max_seq_len=256,
            dim=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            bias=True,
            norm_type="layernorm",
        )

    @classmethod
    def medium(cls) -> "NexLMConfig":
        """Medium preset (~35M parameters)."""
        return cls(
            vocab_size=8192,
            max_seq_len=512,
            dim=384,
            n_layers=8,
            n_heads=12,
            dropout=0.1,
            bias=True,
            norm_type="layernorm",
        )
