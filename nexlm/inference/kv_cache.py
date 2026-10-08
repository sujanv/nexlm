from typing import List, Tuple, Optional
import torch


class KVCache:
    """
    Manages per-layer Key and Value tensors across autoregressive decoding steps.
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

    def get_layer(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.cache[layer_idx]

    def update_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.cache[layer_idx] = (k, v)

    def update_all(self, new_cache: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.cache = list(new_cache)

    @property
    def current_seq_len(self) -> int:
        """Length of cached sequence tokens (from first non-empty layer)."""
        for item in self.cache:
            if item is not None:
                # Key shape: (B, n_heads, seq_len, head_dim)
                return item[0].shape[2]
        return 0

    def reset(self) -> None:
        self.cache = [None] * self.n_layers
