"""
Milestone 5: The Full GPT Model Architecture

Data flow:
    token IDs ──> Token Embedding ──(+)──> Positional Embedding
                                     │
                             Transformer Block 1
                                     │
                             Transformer Block 2
                                     │
                                    ...
                                     │
                             Transformer Block N
                                     │
                                 LayerNorm
                                     │
                        Linear Head ──> Vocabulary Logits
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import TransformerBlock, LayerNorm


@dataclass
class GPTConfig:
    vocab_size: int = 5000
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    tie_weights: bool = True


class GPT(nn.Module):
    """
    GPT-style autoregressive decoder-only language model.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and positional embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final normalization and LM projection head
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: reuse token embedding weights in lm_head
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Weight initialization (GPT-2 standard: N(0, 0.02))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            targets: (batch_size, seq_len) optional next-token labels for cross-entropy loss
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Compute position indices
        positions = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=device)

        # Combine embeddings
        tok_embeddings = self.tok_emb(input_ids)
        pos_embeddings = self.pos_emb(positions)
        x = self.drop(tok_embeddings + pos_embeddings)

        new_kv_caches = [] if use_cache else None

        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache and updated_cache is not None:
                new_kv_caches.append(updated_cache)

        # Final LayerNorm
        x = self.ln_f(x)

        # Project to vocabulary logits: (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Language modeling next-token cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_kv_caches
