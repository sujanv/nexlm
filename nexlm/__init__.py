"""
NexLM: A small GPT-style language model built from scratch in PyTorch.
"""

from nexlm.config import NexLMConfig
from nexlm.model import (
    LayerNorm,
    RMSNorm,
    PositionalEmbedding,
    TransformerEmbedding,
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    NexLM,
)
from nexlm.tokenizer import BaseTokenizer, ByteLevelBPETokenizer, CharTokenizer
from nexlm.inference import generate, generate_stream, KVCache, sample_next_token
from nexlm.training import NexLMTrainer, TrainingConfig

__version__ = "0.1.0"

__all__ = [
    "NexLMConfig",
    "NexLM",
    "LayerNorm",
    "RMSNorm",
    "PositionalEmbedding",
    "TransformerEmbedding",
    "CausalSelfAttention",
    "MLP",
    "TransformerBlock",
    "BaseTokenizer",
    "ByteLevelBPETokenizer",
    "CharTokenizer",
    "generate",
    "generate_stream",
    "KVCache",
    "sample_next_token",
    "NexLMTrainer",
    "TrainingConfig",
]
