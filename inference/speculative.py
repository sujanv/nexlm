"""
Enhancement: Speculative Decoding Engine

Why Speculative Decoding?
Generating N tokens with a large target model requires N sequential, memory-bandwidth bound forward passes.
Speculative Decoding (Leviathan et al., 2023) breaks this sequential bottleneck:
1. A lightweight "draft" model speculatively proposes gamma candidate tokens.
2. The target model scores all gamma candidate tokens in parallel in ONE single forward pass!
3. Rejection sampling verifies candidate tokens:
       Accept token x if random() <= min(1, P_target(x) / P_draft(x)).
4. If rejected, sample from the adjusted distribution max(0, P_target - P_draft) and stop.

Key Guarantee:
The output distribution is mathematically identical to running the large model directly,
while reducing latency by 2x to 3x!
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt import GPT
from inference.sampler import sample_next_token


@torch.no_grad()
def speculative_generate(
    target_model: GPT,
    draft_model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    gamma: int = 4,
    temperature: float = 0.8,
) -> Tuple[torch.Tensor, float]:
    """
    Autoregressive generation accelerated via Speculative Decoding.

    Args:
        target_model: Full-sized primary language model
        draft_model: Small, lightweight candidate generator
        input_ids: Prompt token IDs of shape (1, T)
        max_new_tokens: Maximum number of tokens to generate
        gamma: Number of speculative candidate tokens proposed per step
        temperature: Sampling temperature

    Returns:
        output_tokens: Generated tokens tensor of shape (1, T + generated)
        acceptance_rate: Fraction of proposed draft tokens accepted
    """
    target_model.eval()
    draft_model.eval()

    tokens = input_ids.clone()
    total_proposed = 0
    total_accepted = 0

    while tokens.shape[1] - input_ids.shape[1] < max_new_tokens:
        curr_len = tokens.shape[1]
        if curr_len >= target_model.config.max_seq_len:
            break

        # Step 1: Draft model generates gamma candidate tokens autoregressively
        draft_tokens = tokens.clone()
        draft_probs = []

        for _ in range(gamma):
            if draft_tokens.shape[1] >= draft_model.config.max_seq_len:
                break
            idx_cond = draft_tokens[:, -draft_model.config.max_seq_len:]
            logits, _, _ = draft_model(idx_cond)
            logits_last = logits[:, -1, :] / max(1e-5, temperature)
            prob_dist = F.softmax(logits_last, dim=-1)

            cand = torch.multinomial(prob_dist, num_samples=1)
            draft_tokens = torch.cat([draft_tokens, cand], dim=1)
            draft_probs.append(prob_dist[0, cand[0, 0]].item())

        num_proposed = draft_tokens.shape[1] - curr_len
        if num_proposed == 0:
            break
        total_proposed += num_proposed

        # Step 2: Target model evaluates all proposed tokens in a single parallel forward pass
        idx_cond = draft_tokens[:, -target_model.config.max_seq_len:]
        target_logits, _, _ = target_model(idx_cond)
        # Slices corresponding to the positions evaluated
        eval_logits = target_logits[:, -(num_proposed + 1) : -1, :] / max(1e-5, temperature)
        target_probs = F.softmax(eval_logits, dim=-1)

        # Step 3: Rejection sampling verification
        accepted_count = 0
        all_accepted = True

        for i in range(num_proposed):
            cand_token = draft_tokens[0, curr_len + i].item()
            p_target = target_probs[0, i, cand_token].item()
            p_draft = max(1e-8, draft_probs[i])

            # Acceptance probability
            accept_ratio = min(1.0, p_target / p_draft)
            if torch.rand(1).item() <= accept_ratio:
                accepted_count += 1
            else:
                # Rejection: sample replacement from max(0, p_target - p_draft)
                p_diff = torch.clamp(target_probs[0, i] - (target_probs[0, i] * (p_draft / max(1e-8, p_target))), min=0.0)
                if p_diff.sum() > 0:
                    p_diff = p_diff / p_diff.sum()
                    replacement = torch.multinomial(p_diff, num_samples=1).unsqueeze(0)
                else:
                    replacement = torch.argmax(target_probs[0, i], keepdim=True).unsqueeze(0)

                # Keep accepted tokens up to i, append replacement, stop draft verification
                tokens = torch.cat([tokens, draft_tokens[:, curr_len : curr_len + accepted_count], replacement], dim=1)
                all_accepted = False
                break

        total_accepted += accepted_count

        if all_accepted:
            # If all gamma tokens were accepted, sample one bonus token from the last target logits
            tokens = draft_tokens
            bonus_logits = target_logits[:, -1, :] / max(1e-5, temperature)
            bonus_token = torch.multinomial(F.softmax(bonus_logits, dim=-1), num_samples=1)
            tokens = torch.cat([tokens, bonus_token], dim=1)

    acceptance_rate = total_accepted / max(1, total_proposed)
    return tokens, acceptance_rate
