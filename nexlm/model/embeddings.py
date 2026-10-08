import torch
import torch.nn as nn
from typing import Optional


class PositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (GPT-2 style).
    Maps position indices 0, 1, ..., max_seq_len - 1 to vectors of size dim.
    """

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (seq_len,) or (batch_size, seq_len)
        return self.embedding(positions)


class TransformerEmbedding(nn.Module):
    """
    Combined Token + Position Embedding module with Dropout.
    """

    def __init__(self, vocab_size: int, max_seq_len: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = PositionalEmbedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            start_pos: integer offset for sequence position (used during KV cache autoregressive inference)
        """
        B, T = input_ids.shape
        assert start_pos + T <= self.max_seq_len, (
            f"Cannot exceed max_seq_len {self.max_seq_len}, requested up to {start_pos + T}"
        )

        device = input_ids.device
        positions = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=device)

        tok_emb = self.token_embedding(input_ids)      # (B, T, dim)
        pos_emb = self.position_embedding(positions)    # (T, dim)

        x = tok_emb + pos_emb
        return self.drop(x)
