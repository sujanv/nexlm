"""
Enhancement: Sparsely-Gated Mixture of Experts (MoE)

Why MoE?
In standard dense transformers, every single parameter is computed for every single token.
Mixture of Experts (Shazeer et al., 2017; Mixtral 8x7B, 2024) replaces the dense MLP
with N parallel "Expert" MLPs, where a gating router selects only the Top-K experts per token.

Result:
A model can have 4x-8x larger parameter capacity while keeping FLOPs per token constant!

Formulation:
    Gating logits: g(x) = x @ W_gate
    Top-K router: Select top-K values, softmax normalize over top-K.
    Output: y = sum_{i in TopK} p_i * Expert_i(x)
    Auxiliary load-balancing loss prevents expert starvation.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import MLP


class TopKRouter(nn.Module):
    """
    Top-K gating router with auxiliary load-balancing loss.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        assert top_k <= num_experts, "top_k cannot exceed num_experts"
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            topk_weights: (B, T, top_k) normalized routing probabilities
            topk_indices: (B, T, top_k) expert index assignments
            aux_loss: Scalar auxiliary loss for expert load-balancing
        """
        B, T, C = x.shape
        # Compute router logits: (B * T, num_experts)
        flat_x = x.view(-1, C)
        router_logits = self.gate(flat_x)

        # Full softmax for load balancing loss computation
        router_probs = F.softmax(router_logits, dim=-1)

        # Pick top-k experts per token
        topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        # Reshape to (B, T, top_k)
        topk_weights = topk_weights.view(B, T, self.top_k)
        topk_indices = topk_indices.view(B, T, self.top_k)

        # Auxiliary load balancing loss (Switch Transformer / GShard formulation):
        # f_i = fraction of tokens routed to expert i
        # P_i = average probability allocated to expert i
        # Loss = num_experts * sum(f_i * P_i)
        expert_mask = F.one_hot(topk_indices.view(-1), num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.mean(dim=0)  # f_i
        mean_probs = router_probs.mean(dim=0)         # P_i
        aux_loss = self.num_experts * torch.sum(tokens_per_expert * mean_probs)

        return topk_weights, topk_indices, aux_loss


class MoEFeedForward(nn.Module):
    """
    Sparsely-Gated Mixture of Experts layer containing N parallel MLP experts.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model=d_model, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList([
            MLP(d_model=d_model, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Routes tokens to their top-k experts and linearly blends expert outputs.
        """
        B, T, C = x.shape
        topk_weights, topk_indices, aux_loss = self.router(x)

        final_output = torch.zeros_like(x)

        # Accumulate outputs across selected experts
        for k in range(self.top_k):
            # For each token, get weight and expert index for the k-th selection
            weight_k = topk_weights[:, :, k].unsqueeze(-1)  # (B, T, 1)
            idx_k = topk_indices[:, :, k]                   # (B, T)

            for expert_idx in range(self.num_experts):
                token_mask = (idx_k == expert_idx)
                if token_mask.any():
                    expert_input = x[token_mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    final_output[token_mask] += weight_k[token_mask] * expert_output

        return final_output, aux_loss
