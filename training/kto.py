"""
Milestone 46: Kahneman-Tversky Optimization (KTO) Alignment Engine

Unlike DPO (which requires pairwise preferences (y_w, y_l)), KTO (Ethayarajh et al., 2024)
directly optimizes language models from unpaired binary feedback (thumbs up / thumbs down):
    x: prompt
    y: completion
    label: bool (True for desirable/reward, False for undesirable/penalty)

Rooted in Daniel Kahneman & Amos Tversky's Prospect Theory:
    Humans are loss-averse (losses hurt more than equivalent gains please).
    Value function:
        r(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )
        z_0 = reference point (estimated KL divergence / reward offset)
        v(r) = 1 - sigmoid(lambda_D * (r - z_0))    if y is desirable
        v(r) = 1 - sigmoid(lambda_U * (z_0 - r))    if y is undesirable
    where lambda_D and lambda_U govern loss aversion (typically lambda_U > lambda_D).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt import GPT


@dataclass
class KTOBatch:
    prompt_ids: torch.Tensor       # (B, T_p)
    completion_ids: torch.Tensor   # (B, T_c)
    labels: torch.Tensor           # (B,) boolean (1 = desirable, 0 = undesirable)


def compute_sequence_log_probs(
    model: GPT,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Computes sum of log probabilities for the completion tokens conditioned on prompt:
        log pi(y | x) = sum_{t=1}^{T_c} log P(y_t | x, y_{<t})
    """
    full_seq = torch.cat([prompt_ids, completion_ids], dim=1)
    B, T_full = full_seq.shape
    T_prompt = prompt_ids.shape[1]
    T_comp = completion_ids.shape[1]

    logits, _, _ = model(full_seq)

    # Shift logits and targets for next-token prediction
    # logits at position i predict token at position i+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full_seq[:, 1:].contiguous()

    # Log probabilities over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log prob of actual target tokens
    gathered_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    # Slice only completion tokens (starting from index T_prompt - 1 in shift_labels)
    comp_start = T_prompt - 1
    comp_log_probs = gathered_log_probs[:, comp_start : comp_start + T_comp]

    # Sum log probabilities over completion tokens
    return comp_log_probs.sum(dim=-1)


class KTOLoss(nn.Module):
    """
    Kahneman-Tversky Optimization (KTO) loss module.
    """

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,    # lambda_D
        undesirable_weight: float = 1.33,  # lambda_U (loss aversion factor > 1)
    ):
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        labels: torch.Tensor,
        ref_point: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes KTO loss.
        Args:
            policy_log_probs: (B,) log pi_theta(y|x)
            ref_log_probs: (B,) log pi_ref(y|x)
            labels: (B,) bool or int tensor (1 = desirable, 0 = undesirable)
            ref_point: Optional reference anchor z_0 (KL divergence baseline).
                       If None, computed dynamically as mean of rewards in batch.
        """
        # Implicit reward: r(x, y) = beta * (log pi - log pi_ref)
        delta_log_probs = policy_log_probs - ref_log_probs
        rewards = self.beta * delta_log_probs

        if ref_point is None:
            # Anchor point z_0: detached mean reward
            z_0 = rewards.detach().mean()
        else:
            z_0 = ref_point

        labels_bool = labels.bool()

        # Desirable loss: 1 - sigmoid(lambda_D * (r - z_0))
        # Note: -log(sigmoid(lambda_D * (r - z_0)))
        desirable_mask = labels_bool
        undesirable_mask = ~labels_bool

        loss_desirable = torch.tensor(0.0, device=rewards.device)
        loss_undesirable = torch.tensor(0.0, device=rewards.device)

        if desirable_mask.any():
            r_pos = rewards[desirable_mask] - z_0
            # v(r) = sigmoid(lambda_D * r_pos)
            loss_desirable = -F.logsigmoid(self.desirable_weight * r_pos).mean()

        if undesirable_mask.any():
            r_neg = z_0 - rewards[undesirable_mask]
            # v(r) = sigmoid(lambda_U * r_neg)
            loss_undesirable = -F.logsigmoid(self.undesirable_weight * r_neg).mean()

        total_loss = loss_desirable + loss_undesirable

        metrics = {
            "kto_loss": total_loss.item(),
            "loss_desirable": loss_desirable.item() if desirable_mask.any() else 0.0,
            "loss_undesirable": loss_undesirable.item() if undesirable_mask.any() else 0.0,
            "mean_reward": rewards.mean().item(),
            "anchor_z0": z_0.item() if isinstance(z_0, torch.Tensor) else float(z_0),
        }

        return total_loss, metrics


class KTOEngine:
    """
    High-level engine for Kahneman-Tversky Optimization (KTO) alignment.
    """

    def __init__(
        self,
        policy_model: GPT,
        ref_model: GPT,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.33,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.policy_model = policy_model.to(self.device)
        self.ref_model = ref_model.to(self.device)
        self.ref_model.eval()  # Reference model stays frozen
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = optimizer
        self.loss_fn = KTOLoss(
            beta=beta,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
        )

    def train_step(self, batch: KTOBatch) -> Dict[str, float]:
        """Executes a single KTO optimization step."""
        self.policy_model.train()
        self.optimizer.zero_grad()

        prompt_ids = batch.prompt_ids.to(self.device)
        completion_ids = batch.completion_ids.to(self.device)
        labels = batch.labels.to(self.device)

        # 1. Compute policy log probs (with gradients)
        policy_log_probs = compute_sequence_log_probs(self.policy_model, prompt_ids, completion_ids)

        # 2. Compute reference log probs (frozen)
        with torch.no_grad():
            ref_log_probs = compute_sequence_log_probs(self.ref_model, prompt_ids, completion_ids)

        # 3. Compute KTO loss
        loss, metrics = self.loss_fn(policy_log_probs, ref_log_probs, labels)

        # 4. Backprop and step
        loss.backward()
        self.optimizer.step()

        return metrics
