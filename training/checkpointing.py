"""
Enhancement: Gradient Checkpointing (Activation Checkpointing)

Why Gradient Checkpointing?
In standard training, every activation tensor generated in the forward pass must be retained
in VRAM until the backward pass reaches it to compute gradients.
For deep transformers or long context lengths, activation memory dwarfs parameter memory!

Gradient Checkpointing (Chen et al., 2016; PyTorch torch.utils.checkpoint):
- Discards intermediate activations during the forward pass.
- Re-evaluates (recomputes) intermediate activations on-demand during the backward pass.
- Reduces activation memory from O(L) to O(sqrt(L)) for an L-layer model, allowing
  2x-3x larger batch sizes or sequence lengths to fit in memory.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerBlockList(nn.Module):
    """
    Wraps a list of TransformerBlock modules with activation checkpointing.
    """

    def __init__(self, blocks: nn.ModuleList, enabled: bool = True):
        super().__init__()
        self.blocks = blocks
        self.enabled = enabled

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        new_caches = [] if use_cache else None

        for block in self.blocks:
            if self.enabled and self.training:
                # Custom forward helper compatible with torch.utils.checkpoint
                def create_custom_forward(module):
                    def custom_forward(tensor):
                        out, _ = module(tensor, kv_cache=None, use_cache=False)
                        return out
                    return custom_forward

                # Discards activations and recomputes them during backward pass
                x = checkpoint(
                    create_custom_forward(block),
                    x,
                    use_reentrant=False,
                )
            else:
                x, cache = block(x, kv_cache=None, use_cache=use_cache)
                if use_cache:
                    new_caches.append(cache)

        return x, new_caches
