"""
Milestone 34: Multi-Token Prediction (MTP) Architecture

Instead of the classical next-token prediction task where the model predicts only token t+1:
    x_1, ..., x_t  ──>  predict x_{t+1}

Multi-Token Prediction (Gloeckle et al., Meta / DeepSeek-V3) trains the network
to simultaneously predict multiple future tokens (t+1, t+2, ..., t+M) using auxiliary heads:
    Head 1: predict x_{t+1} (primary loss)
    Head 2: predict x_{t+2} (auxiliary loss)
    ...
    Head M: predict x_{t+M} (auxiliary loss)

Key advantages:
1. Forces representations to encode multi-step causal dynamics and future planning.
2. Enables self-speculative decoding: at inference, the model generates multiple tokens per step
   using its own MTP heads without requiring a separate draft model.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt import GPT, GPTConfig


class MTPModule(nn.Module):
    """
    Auxiliary projection block for predicting the k-th future token.
    Combines representation h_{t} with token embedding and projects to vocab.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden state from base model or previous MTP head (B, T, D)
        Returns:
            Logits for future token (B, T, vocab_size)
        """
        norm_h = self.norm(h)
        h = h + self.mlp(norm_h)
        logits = self.head(h)
        return logits


class MultiTokenPredictionModel(nn.Module):
    """
    Wraps a base GPT model with M auxiliary future-token prediction heads.
    """

    def __init__(self, base_model: GPT, num_future_tokens: int = 2, aux_loss_weight: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.num_future_tokens = num_future_tokens
        self.aux_loss_weight = aux_loss_weight

        # Create M auxiliary heads for k = 2 ... M+1
        self.aux_heads = nn.ModuleList([
            MTPModule(
                d_model=base_model.config.d_model,
                vocab_size=base_model.config.vocab_size,
            )
            for _ in range(num_future_tokens)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass computing primary logits and optional multi-token losses.
        """
        B, T = input_ids.shape
        cfg = self.base_model.config

        # Extract embeddings and hidden states through transformer blocks
        tok_emb = self.base_model.tok_emb(input_ids)
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.base_model.pos_emb(pos)
        x = self.base_model.drop(tok_emb + pos_emb)

        for block in self.base_model.blocks:
            x, _ = block(x)

        norm_x = self.base_model.ln_f(x)
        primary_logits = self.base_model.lm_head(norm_x)

        total_loss = None
        loss_dict: Dict[str, torch.Tensor] = {}

        if targets is not None:
            # Primary loss: predict t+1 (standard CE loss)
            primary_loss = F.cross_entropy(
                primary_logits.view(-1, cfg.vocab_size),
                targets.view(-1),
            )
            loss_dict["primary_loss"] = primary_loss
            total_loss = primary_loss

            # Multi-token auxiliary losses for k = 2, 3, ...
            for k_idx, aux_head in enumerate(self.aux_heads, start=2):
                aux_logits = aux_head(norm_x)

                # For k-th token prediction, target at position t is targets[t + (k - 1)]
                shift = k_idx - 1
                if T > shift:
                    pred_logits = aux_logits[:, :-shift, :].contiguous()
                    future_targets = targets[:, shift:].contiguous()

                    aux_loss = F.cross_entropy(
                        pred_logits.view(-1, cfg.vocab_size),
                        future_targets.view(-1),
                    )
                    loss_dict[f"aux_loss_k{k_idx}"] = aux_loss
                    total_loss = total_loss + (self.aux_loss_weight * aux_loss)

            loss_dict["total_loss"] = total_loss

        return primary_logits, total_loss, loss_dict

    def predict_multi_tokens(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Predicts next token and future tokens simultaneously for speculative self-drafting.
        Returns:
            List of greedy predicted token IDs for [t+1, t+2, ..., t+1+M]
        """
        self.eval()
        with torch.no_grad():
            B, T = input_ids.shape
            tok_emb = self.base_model.tok_emb(input_ids)
            pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            pos_emb = self.base_model.pos_emb(pos)
            x = self.base_model.drop(tok_emb + pos_emb)

            for block in self.base_model.blocks:
                x, _ = block(x)

            norm_x = self.base_model.ln_f(x)
            primary_logits = self.base_model.lm_head(norm_x)
            primary_token = torch.argmax(primary_logits[:, -1, :], dim=-1)

            predictions = [primary_token]
            for aux_head in self.aux_heads:
                aux_logits = aux_head(norm_x)
                aux_token = torch.argmax(aux_logits[:, -1, :], dim=-1)
                predictions.append(aux_token)

            return predictions
