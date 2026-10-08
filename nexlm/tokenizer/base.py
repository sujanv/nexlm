from abc import ABC, abstractmethod
from typing import List, Optional


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers in NexLM."""

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

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTokenizer":
        pass
