import torch
import torch.nn.functional as F
from typing import Optional


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Scale logits by temperature (higher = more diverse, lower = more focused)."""
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Filter logits to keep only top_k highest values; set others to -infinity.
    """
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    min_top_v = v[:, -1].unsqueeze(-1)
    return torch.where(logits < min_top_v, torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: filter logits to keep smallest set of tokens
    whose cumulative probability exceeds top_p.
    """
    if top_p >= 1.0:
        return logits

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    # Shift right so we keep at least the first token exceeding the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter mask back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample the next token given logits for the last position.
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        temperature: Temperature parameter (>0). If ~0 or greedy, picks argmax.
        top_k: Top-k filtering threshold
        top_p: Top-p (nucleus) filtering threshold (0.0 < top_p <= 1.0)
        
    Returns:
        next_token: Tensor of shape (batch_size, 1)
    """
    # Greedy decoding if temperature is very small
    if temperature is not None and temperature < 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if temperature is not None and temperature != 1.0:
        logits = apply_temperature(logits, temperature)

    if top_k is not None and top_k > 0:
        logits = apply_top_k(logits, top_k)

    if top_p is not None and top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
