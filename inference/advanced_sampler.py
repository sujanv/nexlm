"""
Milestone 48: Advanced Sampling Engine (Min-P, Typical Sampling, Frequency & Presence Penalties)

Provides modern sampling strategies that outperform traditional Top-K and Top-P:
1. Min-P Sampling:
   Filters tokens with probability below min_p * max_prob:
       p_threshold = min_p * max(P)
       Keeps tokens where P >= p_threshold.
   Dynamically adapts to model confidence, preventing hallucinations in flat distributions
   without truncating viable alternatives in peaky distributions.

2. Locally Typical Sampling (Meister et al., 2022):
   Selects tokens whose information content -log P(x) is closest to the conditional entropy H(X):
       score(x) = |-log P(x) - H(X)|
   Sorted by typicality score until cumulative mass reaches tau.

3. Frequency & Presence Penalties:
   logit(t) = logit(t) - alpha_freq * count(t) - alpha_presence * (1 if count(t) > 0 else 0)
"""

from typing import List, Optional
import torch
import torch.nn.functional as F


def apply_frequency_presence_penalties(
    logits: torch.Tensor,
    context_tokens: List[int],
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Applies frequency and presence penalties based on prior token occurrences.
    Args:
        logits: (batch_size, vocab_size) or (vocab_size,)
        context_tokens: list of preceding token IDs
        frequency_penalty: penalty proportional to token frequency count
        presence_penalty: flat penalty for any token that appeared at least once
    """
    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits
    if not context_tokens:
        return logits

    penalized_logits = logits.clone()
    vocab_size = logits.size(-1)

    # Count occurrences of tokens in context
    counts = torch.bincount(
        torch.tensor(context_tokens, dtype=torch.long, device=logits.device),
        minlength=vocab_size,
    ).float()

    has_occurred = (counts > 0).float()
    penalty = (frequency_penalty * counts) + (presence_penalty * has_occurred)

    if penalized_logits.dim() == 2:
        penalized_logits = penalized_logits - penalty.unsqueeze(0)
    else:
        penalized_logits = penalized_logits - penalty

    return penalized_logits


def sample_min_p(
    logits: torch.Tensor,
    min_p: float = 0.05,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Min-P sampling:
    Filters out any token whose probability is less than min_p * max_prob.
    """
    if temperature <= 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / max(temperature, 1e-5)
    probs = F.softmax(scaled_logits, dim=-1)

    # Compute max probability per batch item
    max_probs = probs.max(dim=-1, keepdim=True).values  # (B, 1)
    threshold = min_p * max_probs

    # Mask out tokens with prob < threshold
    mask = probs < threshold
    filtered_logits = scaled_logits.masked_fill(mask, float("-inf"))
    filtered_probs = F.softmax(filtered_logits, dim=-1)

    return torch.multinomial(filtered_probs, num_samples=1)


def sample_typical(
    logits: torch.Tensor,
    typical_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Locally Typical Sampling:
    Selects tokens whose information content is close to the conditional distribution entropy.
    """
    if temperature <= 1e-4 or typical_p >= 1.0:
        scaled_logits = logits / max(temperature, 1e-5) if temperature > 1e-4 else logits
        return torch.argmax(scaled_logits, dim=-1, keepdim=True)

    scaled_logits = logits / max(temperature, 1e-5)
    probs = F.softmax(scaled_logits, dim=-1)

    # 1. Compute entropy: H(X) = - sum(p * log p)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)  # (B, 1)

    # 2. Information content difference: |-log p(x) - H(X)|
    info_diff = torch.abs(-log_probs - entropy)

    # 3. Sort tokens by ascending difference (most typical first)
    sorted_diff, sorted_indices = torch.sort(info_diff, dim=-1, descending=False)
    sorted_probs = torch.gather(probs, dim=-1, index=sorted_indices)

    # 4. Cumulative probability of sorted typical tokens
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 5. Mask out tokens beyond typical_p
    cutoff_mask = cumulative_probs > typical_p
    # Keep at least the single most typical token
    cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
    cutoff_mask[..., 0] = False

    indices_to_remove = cutoff_mask.scatter(dim=-1, index=sorted_indices, src=cutoff_mask)
    filtered_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))
    filtered_probs = F.softmax(filtered_logits, dim=-1)

    return torch.multinomial(filtered_probs, num_samples=1)


def advanced_sample(
    logits: torch.Tensor,
    context_tokens: Optional[List[int]] = None,
    temperature: float = 0.8,
    min_p: Optional[float] = None,
    typical_p: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Unified advanced sampler supporting Min-P, Typical, Penalties, Top-P, and Top-K.
    """
    if context_tokens is not None:
        logits = apply_frequency_presence_penalties(
            logits,
            context_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    if min_p is not None and 0.0 < min_p < 1.0:
        return sample_min_p(logits, min_p=min_p, temperature=temperature)
    elif typical_p is not None and 0.0 < typical_p < 1.0:
        return sample_typical(logits, typical_p=typical_p, temperature=temperature)
    else:
        # Standard fallback with temperature, top_k, top_p
        if temperature <= 1e-4:
            return torch.argmax(logits, dim=-1, keepdim=True)
        scaled = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled[scaled < v[..., -1, None]] = float("-inf")
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            indices_to_remove = remove_mask.scatter(dim=-1, index=sorted_indices, src=remove_mask)
            scaled = scaled.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)
