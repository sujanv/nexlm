"""
Milestone 53: DeepSeek-V3 Style Mixture of Experts (Shared + Routed Experts)

Standard MoE models route every token strictly to top-K experts. However, this creates
instability and redundant parameter duplication across experts for common language tokens.

DeepSeek-V3 MoE architecture introduces:
1. Shared Experts (N_shared):
   Dedicated feed-forward experts that are ALWAYS active for every token, capturing general
   syntax, grammar, and foundational reasoning.
2. Routed Experts (N_routed):
   A pool of specialized experts where a Top-K router selects only the most relevant experts.
3. Auxiliary Load-Balancing Loss:
   Encourages uniform distribution of tokens across routed experts:
       L_balance = alpha * N * sum_{i=1}^N (f_i * P_i)
4. Combined Output:
   y = sum_{j in Shared} FFN_j(x) + sum_{k in TopK} g_k(x) * FFN_k(x)
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.swiglu import SwiGLU


class DeepSeekV3MoE(nn.Module):
    """
    DeepSeek-V3 style MoE module combining shared experts and routed top-k experts.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: Optional[int] = None,
        num_shared_experts: int = 1,
        num_routed_experts: int = 4,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim or int(4 * d_model * 2 / 3)
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = min(top_k, num_routed_experts)
        self.aux_loss_weight = aux_loss_weight

        # 1. Shared Experts: always computed for every token
        self.shared_experts = nn.ModuleList([
            SwiGLU(d_model=d_model, hidden_dim=self.hidden_dim, dropout=dropout)
            for _ in range(num_shared_experts)
        ])

        # 2. Routed Experts: sparsely activated via router
        self.routed_experts = nn.ModuleList([
            SwiGLU(d_model=d_model, hidden_dim=self.hidden_dim, dropout=dropout)
            for _ in range(num_routed_experts)
        ])

        # 3. Router gating projection
        self.gate = nn.Linear(d_model, num_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, T, d_model)
        Returns:
            output: (B, T, d_model)
            aux_loss: Scalar auxiliary load balancing loss
        """
        B, T, D = x.shape
        flat_x = x.view(-1, D)  # (B*T, D)
        num_tokens = flat_x.shape[0]

        # 1. Compute Shared Experts output
        shared_out = torch.zeros_like(flat_x)
        for shared_exp in self.shared_experts:
            shared_out = shared_out + shared_exp(flat_x)

        # 2. Compute Router gating scores
        gate_logits = self.gate(flat_x)  # (num_tokens, num_routed_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (num_tokens, num_routed_experts)

        # Select Top-K routed experts per token
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Re-normalize topk weights so they sum to 1.0
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # 3. Execute Routed Experts
        routed_out = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            expert_indices = topk_indices[:, k]  # (num_tokens,)
            weights = topk_weights[:, k].unsqueeze(-1)  # (num_tokens, 1)

            # Dispatch tokens to selected experts
            for exp_id, exp_module in enumerate(self.routed_experts):
                mask = expert_indices == exp_id
                if mask.any():
                    token_subset = flat_x[mask]
                    exp_result = exp_module(token_subset)
                    routed_out[mask] += weights[mask] * exp_result

        # 4. Compute Auxiliary Load-Balancing Loss
        # f_i: fraction of tokens routed to expert i
        # P_i: average probability assigned to expert i
        tokens_per_expert = torch.zeros(self.num_routed_experts, device=x.device)
        for exp_id in range(self.num_routed_experts):
            tokens_per_expert[exp_id] = (topk_indices == exp_id).sum().float()
        fraction_tokens = tokens_per_expert / (num_tokens * self.top_k)
        avg_prob_per_expert = gate_probs.mean(dim=0)

        # L_balance = weight * N * sum(f_i * P_i)
        aux_loss = self.aux_loss_weight * self.num_routed_experts * torch.sum(fraction_tokens * avg_prob_per_expert)

        # 5. Combine Shared + Routed
        total_out = (shared_out + routed_out).view(B, T, D)
        return total_out, aux_loss
