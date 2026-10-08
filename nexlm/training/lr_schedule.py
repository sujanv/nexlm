import math


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    1. Linear warmup from 0 to max_lr for step < warmup_steps
    2. Cosine decay from max_lr to min_lr for step in [warmup_steps, max_steps]
    3. Flat min_lr after max_steps
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / max(1, warmup_steps)

    if step > max_steps:
        return min_lr

    # Cosine decay down to min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
