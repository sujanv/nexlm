"""
Enhancement: Direct Preference Optimization (DPO) for LLM Alignment

Why DPO?
Traditional RLHF (Reinforcement Learning from Human Feedback) requires training a separate
reward model and using complex actor-critic PPO loops, which are unstable and memory-heavy.
Rafailov et al. (NeurIPS 2023) proved that the optimal policy under the Bradley-Terry preference
model can be extracted analytically:
    L_DPO(pi_theta; pi_ref) = -E_{(x, y_w, y_l)} [ log sigma( beta * ( log(pi(y_w|x)/pi_ref(y_w|x))
                                                                       - log(pi(y_l|x)/pi_ref(y_l|x)) ) ) ]
where:
    y_w is the winning (chosen) completion
    y_l is the losing (rejected) completion
    beta is the temperature parameter controlling KL divergence constraint to reference model.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt import GPT


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Computes sequence log-probabilities under the model for each batch element.
    Args:
        input_ids: (B, T)
        labels: (B, T) with -100 for ignored prefix tokens
    Returns:
        log_probs: (B,) sum of log probabilities over valid label tokens
    """
    logits, _, _ = model(input_ids)
    # Log softmax over vocabulary
    log_probs = F.log_softmax(logits, dim=-1)

    # Shift labels: predict token t from index t-1
    # input_ids[:, :-1] predicts labels[:, 1:]
    shift_log_probs = log_probs[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = (shift_labels != -100).float()
    safe_labels = shift_labels.clone()
    safe_labels[shift_labels == -100] = 0

    # Gather log prob of true labels
    token_log_probs = torch.gather(shift_log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    # Sum over active sequence tokens
    seq_log_probs = (token_log_probs * mask).sum(dim=-1)
    return seq_log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Direct Preference Optimization (DPO) loss.

    Returns:
        loss: Scalar DPO loss
        chosen_rewards: Implicit reward for chosen completions
        rejected_rewards: Implicit reward for rejected completions
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Implicit rewards for logging
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    return loss, chosen_rewards, rejected_rewards


class DPOTrainer:
    """Trainer orchestrator for Direct Preference Optimization."""

    def __init__(
        self,
        policy_model: GPT,
        ref_model: GPT,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.beta = beta

        # Ref model is always frozen
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Single DPO optimization step on a preference batch."""
        self.policy_model.train()

        # 1. Forward policy model
        policy_chosen_logps = compute_log_probs(self.policy_model, chosen_ids, chosen_labels)
        policy_rejected_logps = compute_log_probs(self.policy_model, rejected_ids, rejected_labels)

        # 2. Forward reference model (no gradients)
        with torch.no_grad():
            ref_chosen_logps = compute_log_probs(self.ref_model, chosen_ids, chosen_labels)
            ref_rejected_logps = compute_log_probs(self.ref_model, rejected_ids, rejected_labels)

        # 3. Compute DPO loss
        loss, chosen_r, rejected_r = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=self.beta,
        )

        # 4. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        reward_accuracy = (chosen_r > rejected_r).float().mean().item()

        return {
            "loss": loss.item(),
            "chosen_reward": chosen_r.mean().item(),
            "rejected_reward": rejected_r.mean().item(),
            "reward_margin": (chosen_r - rejected_r).mean().item(),
            "accuracy": reward_accuracy,
        }
