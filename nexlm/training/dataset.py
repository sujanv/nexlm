from typing import List, Tuple, Union, Optional
import torch
from torch.utils.data import Dataset
import numpy as np

from nexlm.tokenizer.base import BaseTokenizer


class TokenizedDataset(Dataset):
    """
    PyTorch Dataset for language modeling.
    Extracts fixed-length chunks (x, y) where y is x shifted by 1 token.
    """

    def __init__(self, token_ids: Union[List[int], np.ndarray, torch.Tensor], seq_len: int):
        self.seq_len = seq_len
        if isinstance(token_ids, list):
            self.tokens = torch.tensor(token_ids, dtype=torch.long)
        elif isinstance(token_ids, np.ndarray):
            self.tokens = torch.from_numpy(token_ids.astype(np.int64))
        else:
            self.tokens = token_ids.to(dtype=torch.long)

        assert len(self.tokens) > seq_len, (
            f"Dataset length ({len(self.tokens)}) must be greater than seq_len ({seq_len})"
        )

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def prepare_dataset_from_text(
    text: str,
    tokenizer: BaseTokenizer,
    seq_len: int,
    val_split: float = 0.1,
) -> Tuple[TokenizedDataset, Optional[TokenizedDataset]]:
    """
    Tokenizes raw text and returns train and validation TokenizedDataset instances.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= seq_len + 1:
        # Repeat tokens if text is very short
        repeats = ((seq_len + 2) // len(tokens)) + 1
        tokens = tokens * repeats

    split_idx = int(len(tokens) * (1.0 - val_split))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    if len(train_tokens) <= seq_len + 1:
        train_tokens = tokens

    train_ds = TokenizedDataset(train_tokens, seq_len=seq_len)
    val_ds = TokenizedDataset(val_tokens, seq_len=seq_len) if len(val_tokens) > seq_len + 1 else None
    return train_ds, val_ds
