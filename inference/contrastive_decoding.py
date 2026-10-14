"""
Milestone 41: Contrastive Decoding & Classifier-Free Guidance (CFG) for LLMs

1. Contrastive Decoding (Li et al., 2023):
   Contrasts an Expert Model against an Amateur Model to subtract out repetitive,
   generic, or vacuous tokens:
       L_CD = L_expert - alpha * L_amateur
   Subject to the Adaptive Plausibility Constraint (APC):
       V_head = { w in V | P_expert(w) >= beta * max_v P_expert(v) }

2. Classifier-Free Guidance (CFG) for Autoregressive LLMs (Sanchez et al., 2023):
   Contrasts conditioned prompt logits against unconditioned logits:
       L_CFG = L_uncond + gamma * (L_cond - L_uncond)
   When gamma > 1.0, forces strict compliance with system prompts and formatting rules.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt import GPT
from inference.sampler import sample_next_token


def apply_contrastive_penalty(
    expert_logits: torch.Tensor,
    amateur_logits: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Applies Contrastive Decoding logit subtraction with Adaptive Plausibility Constraint.
    """
    # 1. Compute expert probabilities
    expert_probs = F.softmax(expert_logits, dim=-1)
    max_expert_prob = expert_probs.max(dim=-1, keepdim=True).values

    # 2. Plausibility mask: only tokens with P_expert >= beta * max(P_expert)
    plausible_mask = expert_probs >= (beta * max_expert_prob)

    # 3. Contrastive logit subtraction
    diff_logits = expert_logits - (alpha * amateur_logits)

    # 4. Mask out unplausible tokens
    filtered_logits = torch.where(plausible_mask, diff_logits, torch.full_like(diff_logits, float("-inf")))
    return filtered_logits


def apply_cfg_guidance(
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    cfg_scale: float = 1.5,
) -> torch.Tensor:
    """
    Applies Classifier-Free Guidance across conditioned and unconditioned logits:
        L_CFG = L_uncond + cfg_scale * (L_cond - L_uncond)
    """
    if cfg_scale == 1.0:
        return cond_logits
    return uncond_logits + cfg_scale * (cond_logits - uncond_logits)


@torch.no_grad()
def generate_contrastive(
    expert_model: GPT,
    amateur_model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    alpha: float = 0.5,
    beta: float = 0.1,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
) -> torch.Tensor:
    """
    Autoregressive generation using Contrastive Decoding between expert and amateur models.
    """
    expert_model.eval()
    amateur_model.eval()
    tokens = input_ids.clone()

    for _ in range(max_new_tokens):
        idx_cond = tokens[:, -expert_model.config.max_seq_len :]
        
        # Expert forward
        expert_logits, _, _ = expert_model(idx_cond)
        exp_next = expert_logits[:, -1, :]

        # Amateur forward
        amateur_logits, _, _ = amateur_model(idx_cond)
        ama_next = amateur_logits[:, -1, :]

        # Apply contrastive penalty
        cd_logits = apply_contrastive_penalty(exp_next, ama_next, alpha=alpha, beta=beta)

        # Sample next token
        next_token = sample_next_token(cd_logits, temperature=temperature, top_k=top_k)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


@torch.no_grad()
def generate_cfg(
    model: GPT,
    prompt_ids: torch.Tensor,
    uncond_ids: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.5,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
) -> torch.Tensor:
    """
    Autoregressive generation using Classifier-Free Guidance (CFG).
    """
    model.eval()
    cond_tokens = prompt_ids.clone()
    
    # If no unconditioned prompt provided, use single BOS/EOS/PAD token
    if uncond_ids is None:
        uncond_tokens = torch.tensor([[0]], dtype=torch.long, device=prompt_ids.device)
    else:
        uncond_tokens = uncond_ids.clone()

    for _ in range(max_new_tokens):
        cond_cond = cond_tokens[:, -model.config.max_seq_len :]
        uncond_cond = uncond_tokens[:, -model.config.max_seq_len :]

        cond_logits, _, _ = model(cond_cond)
        uncond_logits, _, _ = model(uncond_cond)

        guided_logits = apply_cfg_guidance(
            cond_logits[:, -1, :],
            uncond_logits[:, -1, :],
            cfg_scale=cfg_scale,
        )

        next_token = sample_next_token(guided_logits, temperature=temperature, top_k=top_k)
        cond_tokens = torch.cat([cond_tokens, next_token], dim=1)
        uncond_tokens = torch.cat([uncond_tokens, next_token], dim=1)

    return cond_tokens
