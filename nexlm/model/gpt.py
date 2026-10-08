import math
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from nexlm.config import NexLMConfig
from nexlm.model.embeddings import TransformerEmbedding
from nexlm.model.transformer import TransformerBlock
from nexlm.model.normalization import get_norm_layer


class NexLM(nn.Module):
    """
    NexLM: Full GPT-style decoder-only transformer language model.
    """

    def __init__(self, config: NexLMConfig):
        super().__init__()
        self.config = config

        self.embeddings = TransformerEmbedding(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            dim=config.dim,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                bias=config.bias,
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = get_norm_layer(
            config.norm_type,
            config.dim,
            eps=config.norm_eps,
            bias=config.bias,
        )

        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share parameters between token embedding and final linear projection
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return parameter count in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.position_embedding.embedding.weight.numel()
        return n_params

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
            input_ids: (batch_size, seq_len) token indices
            targets: (batch_size, seq_len) target indices for language modeling loss
            kv_cache: List of per-layer (key, value) cache tuples
            use_cache: Whether to return updated KV cache
            start_pos: Position index offset for positional embeddings during KV generation
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: Cross-entropy scalar loss (if targets is not None else None)
            new_kv_cache: List of updated (k, v) tuples per layer (if use_cache is True else None)
        """
        B, T = input_ids.shape
        x = self.embeddings(input_ids, start_pos=start_pos)

        new_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache and updated_cache is not None:
                new_kv_cache.append(updated_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Shift happens in data loader or caller; compute cross entropy across flattened tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, new_kv_cache

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 6e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        """
        Filter parameters: apply weight decay to 2D tensors (weights),
        disable weight decay on 1D tensors (biases, layer norms).
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if supported and on CUDA
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )
        return optimizer
